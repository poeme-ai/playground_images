import json
import os
import time
import re
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import google.generativeai as genai
import anthropic
import logging

MODEL_PROVIDER_MAP = {
    "gpt-4o": "openai",
    "gpt-4o-2024-05-13": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4-turbo-2024-04-09": "openai",
    "gpt-4-0125-preview": "openai",
    "gpt-4-turbo-preview": "openai",
    "gpt-4-1106-preview": "openai",
    "gpt-4-1106-vision-preview": "openai",
    "gpt-4": "openai",
    "gpt-4-32k": "openai",
    "gpt-3.5-turbo-0125": "openai",
    "gpt-3.5-turbo-instruct": "openai",
    "gemini-pro": "gemini",
    #"gemini-ultra": "gemini",   ainda em preview, não disponível na API
    "claude-3-opus-20240229": "claude",
    "claude-3-sonnet-20240229": "claude",
    "claude-3-haiku-20240307": "claude",
}
MODELS_TOKEN_LIMIT = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-1106-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-instruct": 4096,
    "gemini-pro": 30720,
    # "gemini-ultra": , # Define token limit once available
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
}

OPEN_AI_ASSISTANT_TOOLS = [{"type": "code_interpreter"}, {"type": "retrieval"}]

load_dotenv()

class RequestAttemptsExceededError(Exception):
    def __init__(self, attempts, exception, message="API request attempts exceeded"):
        self.attempts = attempts
        self.exception_message = str(exception)
        self.message = f"{message}: Failed after {attempts} attempts. Exception: {self.exception_message}"
        super().__init__(self.message)

class APIAdapter(ABC):
    """
    Adapter Interface for dependency injection
    A generic class to adapt different API services for use in ChatApp.
    OpenAIAdapter, GoogleGeminiAdapter and ClaudeAdapter are implementing all the abstract methods.
    """

    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    def add_message_to_context(self, role, message, messages):
        pass

    @abstractmethod
    def chat(self, message, messages, system_prompt = None, max_tokens=300, temperature=1, request_attempts=1):
        pass

    @abstractmethod
    def single_message_completion(self, user_message, system_message=None, output_type=None, max_tokens=4000, temperature=1, request_attempts=1):
        pass

    @abstractmethod
    def calculate_cost(self, input_tokens, output_tokens):
        pass

    @abstractmethod
    def count_tokens(self, string: str):
        pass


class OpenAIAdapter(APIAdapter):
    """
    Implements the APIAdapter interface to provide a concrete adapter for the OpenAI API.
    This class facilitates communication with the OpenAI's models, abstracting the API's interaction details.
    
    Attributes:
        client (OpenAI): The client instance to communicate with OpenAI's API.
        model (str): The model identifier to be used for requests to the OpenAI API.
    """
    def __init__(self, api_key, model, logger):
        super().__init__(logger)
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def add_message_to_context(self, role, message, messages):
        """
        Appends a new message to the conversation history and returns the updated history.
        """
        messages.append({"role": role, "content": message})
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

    def chat(self, message, messages, system_prompt = None, max_tokens=300, temperature=1, request_attempts=1):
        """
        Sends a message and a chat history to the OpenAI API and returns the model's response. Updates messages.
        """
        if system_prompt:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        attempts_count = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, 
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                break
            except Exception as e:
                attempts_count +=1
                self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                if attempts_count >= request_attempts:
                    self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                    raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)


        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content

    def single_message_completion(self, user_message, system_message=None, output_type=None, max_tokens=4000, temperature=1, request_attempts=1):
        """
        Generates a response from the model based on a single input user message and optional system message.
        """
        one_time_generate_messages = []
        if system_message:
            one_time_generate_messages.append({"role": "system", "content": system_message})

        one_time_generate_messages.append({"role": "user", "content": user_message})

        if output_type == "json":
            attempts_count = 0
            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=one_time_generate_messages,
                        temperature=temperature,
                        response_format={ "type": "json_object" },
                    )
                    break
                except Exception as e:
                    attempts_count +=1
                    self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                    if attempts_count >= request_attempts:
                        self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                        raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)
        else:
            attempts_count = 0
            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model, 
                        messages=one_time_generate_messages, 
                        max_tokens=max_tokens, 
                        temperature=temperature
                    )
                    break
                except Exception as e:
                    attempts_count +=1
                    self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                    if attempts_count >= request_attempts:
                        self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                        raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)
                    
        return response.choices[0].message.content
    
    def calculate_cost(self, input_tokens, output_tokens):
        """
        Calculates the estimated cost of a request based on token usage.
        """
        prices_per_1000_tokens = {
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-2024-05-13": (0.005, 0.015),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-4-turbo-2024-04-09": (0.01, 0.03),
            "gpt-4-0125-preview": (0.01, 0.03),
            "gpt-4-turbo-preview": (0.01, 0.03),
            "gpt-4-1106-preview": (0.01, 0.03),
            "gpt-4-1106-vision-preview": (0.01, 0.03),
            "gpt-4": (0.03, 0.06),
            "gpt-4-32k": (0.06, 0.12),
            "gpt-3.5-turbo-0125": (0.0005, 0.0015),
            "gpt-3.5-turbo-instruct": (0.0015, 0.0020)
        }
        input_price, output_price = prices_per_1000_tokens.get(self.model, (0, 0))
        total_cost_usd = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
        cotacao_dolar_real = 5
        return total_cost_usd * cotacao_dolar_real

    def count_tokens(self, string: str):
        """
        Counts the number of tokens in a given string using the model's tokenizer.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
class GoogleGeminiAdapter(APIAdapter):
    """
    Implements the APIAdapter interface to provide a concrete adapter for the Google Gemini API.
    This class facilitates communication with the Gemini's models, abstracting the API's interaction details.
    
    Attributes:
        model (str): The model identifier to be used for requests to the Google API.

    """
    def __init__(self, api_key, model, logger):
        super().__init__(logger)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(f'models/{model}')
        # Initialize any Google Gemini specific setup here

    def adapt_messages_to_gemini_format(self, messages):
        """
        Converts messages from the standard format to Gemini's format, excluding system messages, since they are no supported by Google Gemini.

        :param messages: List of message dictionaries in the format [{"role": role, "content": message}]
        :return: Adapted messages in Gemini's format.
        """
        gemini_messages = []
        for message in messages:
            if message["role"] != "system":
                gemini_message = {
                    "parts": [{"text": message["content"]}],
                    "role": "user" if message["role"] == "user" else "model"
                }
                gemini_messages.append(gemini_message)
            
        return gemini_messages

    def adapt_response_from_gemini_format(self, gemini_messages):
        """
        Converts messages from Gemini's format back to the standard format.

        :param gemini_messages: Messages in Gemini's format.
        :return: Messages in the standard format.
        """
        standard_messages = []
        for gemini_message in gemini_messages:
            text = " ".join(part["text"] for part in gemini_message["parts"])
            role = "user" if gemini_message["role"] == "user" else "assistant"
            standard_messages.append({"role": role, "content": text})
        return standard_messages

    def add_message_to_context(self, role, message, messages):
        """
        Appends a new message to the conversation history and returns the updated history.
        """
        messages.append({"role": role, "content": message})
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

    def chat(self, message, messages, system_prompt = None, max_tokens=300, temperature=1, request_attempts=1):
        """
        Sends a message and a chat history to the Google Gemini API and returns the model's response. Updates messages.
        """

        attempts_count = 0
        while True:
            try:
                gemini_message = self.adapt_messages_to_gemini_format(messages)
                chat = self.model.start_chat(history=gemini_message,)
                chat
                if messages == [] and system_prompt:
                    message = f'Nesta conversa, você deve assumir o seguinte papel e as seguintes regras:\n{system_prompt}\n\nAgora, estou mandando a primeira mensagem: {message}'
                response = chat.send_message(message)
                break
            except Exception as e:
                attempts_count +=1
                self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                if attempts_count >= request_attempts:
                    self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                    raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)
        
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": response.text})

        return response.text

    def single_message_completion(self, user_message, system_message=None, output_type=None, max_tokens=4000, temperature=1, request_attempts=1):
        """
        Generates a response from the model based on a single input user message.
        """

        input_message_in_format = {
            "parts": [
                {"text": user_message}
            ]
        }
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        attempts_count = 0
        while True:
             try:
                response = self.model.generate_content(
                    input_message_in_format, 
                    generation_config=generation_config
                )
                break
             except Exception as e:
                attempts_count +=1
                self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                if attempts_count >= request_attempts:
                    self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                    raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)

        return response.text

    def calculate_cost(self, input_tokens, output_tokens):
        """
        Calculates the estimated cost of a request based on token usage.
        """
        prices_per_1000_tokens = {
            "gemini-pro": (0.00, 0.00),
            #"gemini-ultra": (0.0x, 0.0x), anda em preview, sem acesso pelo API
        }
        input_price, output_price = prices_per_1000_tokens.get(self.model, (0, 0))
        total_cost_usd = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
        cotacao_dolar_real = 5
        return total_cost_usd * cotacao_dolar_real

    def count_tokens(self, string: str):
        """
        Counts the number of tokens in a given string using the model's tokenizer.
        """
        # Assuming the tokenization process is the same for Google Gemini, for now, using tiktoken (best approximation available)
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        num_tokens = len(encoding.encode(string))
        return num_tokens

class ClaudeAdapter(APIAdapter):
    """
    Implements the APIAdapter interface to provide a concrete adapter for the Claude API.
    This class is specifically tailored for interacting with Claude models, encapsulating
    the logic required for handling system prompts and generating responses.

    Attributes:
        client (Anthropic): Client instance for communicating with the Claude API.
        model (str): Identifier of the Claude model to use for generating responses.

    """
    def __init__(self, api_key, model, logger):
        super().__init__(logger)
        self.client = anthropic.Anthropic(api_key=api_key,)
        self.model = model
    
    def extract_system_prompt(self, messages):
        """
        Extracts and removes the system prompt from the messages list.
        Needed because Claude API demands the system prompt as a parameter, not in the context.

        :param messages: The list of message dictionaries.
        :return: The extracted system prompt or an empty string if not found.
        """
        for i, message in enumerate(messages):
            if message["role"] == "system":
                return messages.pop(i)["content"]
        return ""  # Return an empty string if no system prompt is found

    def replace_system_prompt(self, messages, system_prompt):
        """
        Replaces or inserts the system prompt at the beginning of the messages list.

        :param messages: The list of message dictionaries.
        :param system_prompt: The system prompt content to insert.
        """
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

    def add_message_to_context(self, role, message, messages):
        """
        Appends a new message to the conversation history and returns the updated history.
        """
        messages.append({"role": role, "content": message})
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    
    def chat(self, message, messages, system_prompt = None, max_tokens=300, temperature=1, request_attempts=1):
        """
        Sends a message and a chat history to the Claude API and returns the model's response. Updates messages.
        """

        if system_prompt:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})

        system_prompt = self.extract_system_prompt(messages)
        messages.append({"role": "user", "content": message})
        attempts_count = 0
        while True:
            try:
                response = self.client.messages.create(
                    model=self.model, 
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature
                )
                break
            except Exception as e:
                attempts_count +=1
                self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                if attempts_count >= request_attempts:
                    self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                    raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)

        response_text = response.content[0].text    
        messages.append({"role": "assistant", "content": response_text})
        self.replace_system_prompt(messages, system_prompt)
        return response_text

    def single_message_completion(self, user_message, system_message=None,  output_type=None, max_tokens=4000, temperature=1, request_attempts=1):
        """
        Generates a response from the model based on a single input user message.
        """
        one_time_generate_messages = []
        one_time_generate_messages.append({"role": "user", "content": user_message})

        if system_message is None:
            attempts_count = 0
            while True:
                try:
                    response = self.client.messages.create(
                        model=self.model, 
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=one_time_generate_messages,
                    )
                    break
                except Exception as e:
                    attempts_count +=1
                    self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                    if attempts_count >= request_attempts:
                        self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                        raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)
        else:
            attempts_count = 0
            while True:
                try:
                    response = self.client.messages.create(
                        model=self.model, 
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_message,
                        messages=one_time_generate_messages,
                    )
                    break
                except Exception as e:
                    attempts_count +=1
                    self.logger.error(f"Falha ao chamar API do modelo {self.model}: {e}")
                    if attempts_count >= request_attempts:
                        self.logger.error(f"Número de tentativas totais ({attempts_count}) de chamada de API para {self.model} chegou ao limite ({request_attempts})")
                        raise RequestAttemptsExceededError(attempts=request_attempts, exception=e)
                    
        return response.content[0].text

    def calculate_cost(self, input_tokens, output_tokens):
        """
        Calculates the estimated cost of a request based on token usage.
        """
        prices_per_1000_tokens = {
            "claude-3-opus-20240229": (0.015, 0.075),
            "claude-3-sonnet-20240229": (0.003, 0.015),
        }
        input_price, output_price = prices_per_1000_tokens.get(self.model, (0, 0))
        total_cost_usd = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
        cotacao_dolar_real = 5
        return total_cost_usd * cotacao_dolar_real

    def count_tokens(self, string: str):
        """
        Counts the number of tokens in a given string using the model's tokenizer.
        """
        # Assuming the tokenization process is the same for Claude models, for now, using tiktoken (best approximation available)
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        num_tokens = len(encoding.encode(string))
        return num_tokens

class AssistantOpenAi:
    """
    This class provides an interface to interact with OpenAI's Assistant API, allowing the creation of 
    an AI assistant, managing threads, sending messages, and processing responses.

    Attributes:
        client (OpenAI): An instance of the OpenAI client, initialized with the user's API key.
        model (str): The identifier of the model used by the Assistant.
    """

    def __init__(self, api_key, model):
        """
        Initializes the AssistantOpenAi with an API key and a model.

        Parameters:
            api_key (str): The API key to authenticate requests to OpenAI.
            model (str): The model identifier to be used with the Assistant.
        """

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_assistant(self, name, instructions, tools=None, file_ids=None):
        """
        Creates an assistant with a specified name, instructions, and optional tools.

        Parameters:
            name (str): The name of the assistant.
            instructions (str): Instructions that define the behavior of the assistant.
            tools (list): Optional; tools to enable for the assistant (e.g., Code Interpreter).

        Returns:
            str: The ID of the created assistant.
        """

        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools or [],
            model=self.model,
            file_ids=file_ids or []
        )

        return assistant.id

    def create_thread(self):
        """
        Creates a new conversation thread.

        Returns:
            str: The ID of the created thread.
        """

        thread = self.client.beta.threads.create()
        return thread.id

    def add_message_to_thread(self, thread_id, content, role="user", file_ids=None):
        """
        Adds a message to the specified thread.

        Parameters:
            thread_id (str): The ID of the thread to which the message should be added.
            content (str): The content of the message.
            role (str): The role of the message sender (e.g., "user", "assistant").

        Returns:
            str: The ID of the added message.
        """
        params = {
            'thread_id': thread_id,
            'role': role,
            'content': content,
        }
        if file_ids:
            params['file_ids']=file_ids

        message = self.client.beta.threads.messages.create(
            **params
        )
        return message.id

    def run_assistant(self, thread_id, assistant_id, instructions=None):
        """
        Executes the assistant to process the messages in the thread and generate a response.

        Parameters:
            thread_id (str): The ID of the thread to be processed.
            assistant_id (str): The ID of the assistant to run.
            instructions (str): Optional; additional instructions for this specific run.

        Returns:
            str: The ID of the run.
        """

        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions
        )
        return run.id

    def get_run_status(self, thread_id, run_id):
        """
        Retrieves the status of a specific run of the assistant.

        Parameters:
            thread_id (str): The ID of the thread associated with the run.
            run_id (str): The ID of the run to check the status of.

        Returns:
            str: The status of the run.
        """

        run = self.client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        return run.status

    def get_messages_from_thread(self, thread_id):
        """
        Retrieves all messages from a specified thread.

        Parameters:
            thread_id (str): The ID of the thread from which to retrieve messages.

        Returns:
            list: A list of messages from the thread.
        """

        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data
    
    def get_last_message_reponse_from_assistant_chat(self, assistant_messages_history):
        """
        Retrieves last messages from assistant from a specified assistant_messages_history.

        Parameters:
            assistant_messages_history (list): List of messages between assitant and user in the open ai thread format.

        Returns:
            string: A string with the last assistant messages.
        """

        assistant_responses = []
        for message in assistant_messages_history:
            if message.role != 'assistant':
                break

            content = message.content[0]
            assistant_response = content.text.value 
            assistant_responses.append(assistant_response)

        return '\n\n'.join(reversed(assistant_responses))
    
    def save_ids(self, assistant_id, thread_id, filename="assistant_thread_ids.json"):
        """
        Saves the assistant and thread IDs to a JSON file.
        """
        #TODO EM AMBIENTE DE PRODUCAO, PODEMOS FAZER EM BANCO DE DADOS
        ids = {
            "assistant_id": assistant_id,
            "thread_id": thread_id
        }
        with open(filename, "w") as file:
            json.dump(ids, file)
        print(f"Assistant and thread IDs saved to {filename}.")

    def load_ids(self, filename="assistant_thread_ids.json"):
        """
        Loads the assistant and thread IDs from a JSON file.
        """
        #TODO EM AMBIENTE DE PRODUCAO, PODEMOS FAZER EM BANCO DE DADOS
        with open(filename, "r") as file:
            ids = json.load(file)
        self.assistant_id = ids.get("assistant_id")
        self.thread_id = ids.get("thread_id")
        #TODO A TROCAR PRINT POR LOGS
        print(f"Assistant and thread IDs loaded from {filename}.")

    def read_generated_file(self, fileId: str):
        """
        Retrieves the content of a file stored on OpenAI's servers using the provided file ID,
        and returns the data directly. This method is useful for processing or using data
        retrieved from the assistant's output immediately in the application without the need
        to write it to a local file system.

        Parameters:
            fileId (str): The unique identifier of the file on OpenAI's server. This ID is
                        typically obtained after a file has been uploaded or generated by
                        the assistant as part of its output.

        Returns:
            bytes: The content of the file as a byte string. This can be directly used or
                converted into other formats depending on the content type of the file.
        """
        file = self.client.files.content(fileId)
        data = file.read()
        return data
 

class ChatApp:
    """
    A class to interact with LLM's models for chat or assistant applications. It supports direct chat
    functionalities with various models and can also integrate with the OpenAI Assistant API to manage
    conversations using specific assistants.

    Attributes:
        model (str): The model used for chat completions.
        messages (list): A list of message dicts representing the conversation history.
        total_cost (float): Total accumulated cost of the interactions (for billing purposes).
        api_provider (str): The provider of the model, determining which API to use.
        assistant_on (bool): Initialize Open AI assistant (only for Open AI Models)
        api_adapter (APIAdapter): The adapter corresponding to the selected API provider.
        assistant_ai (AssistantOpenAi): Interface to interact with the OpenAI Assistant API.
        assistant_id (str): ID of the initialized assistant.
        thread_id (str): ID of the created conversation thread.
        assistant_id (str, optional): The ID of the initialized assistant. If provided, it will use the existing assistant; otherwise, it will create a new one.
        thread_id (str, optional): The ID of the created conversation thread. If provided, it will use the existing thread; otherwise, it will create a new one.
    """

    def __init__(self, model="gpt-4-0125-preview", api_key = None, messages = [], load_file='', assistant_on = False, assistant_id=None, thread_id=None):
        """
        Initializes the ChatApp with a specific model and optionally loads a chat history.

        :param model: The model to be used for chat completions, default is "gpt-4.0-turbo".
        :param load_file: Path to a file to load previous chat history, default is an empty string.
        :param assistant_on: Initialize Open AI Assistant API
        """

        self.logger = logging.getLogger('ChatApp')
        self.logger.setLevel(logging.INFO)
        self.set_logs()

        self.model = model
        self.messages = messages
        self.total_cost = 0
        self.assistant_on = assistant_on

        if load_file:
            self.load(load_file)

        self.api_provider = MODEL_PROVIDER_MAP.get(self.model)
        if self.api_provider is None:
            raise ValueError(f"The model '{model}' is not supported or is not mapped to an API provider.")
           
            
        if self.api_provider == 'gemini':
            if api_key is None:
                api_key = os.environ['GOOGLE_GEMINI_API_KEY']
            self.api_adapter = GoogleGeminiAdapter(
                api_key,
                model,
                self.logger
            )
        elif self.api_provider == 'claude':
            if api_key is None:
                api_key = os.environ['ANTHROPIC_API_KEY']
            self.api_adapter = ClaudeAdapter(
                api_key,
                model,
                self.logger
            )
        else:
            if api_key is None:
                api_key = os.environ['OPENAI_API_KEY']
            self.api_adapter = OpenAIAdapter(
                api_key,
                model,
                self.logger
            )
            if assistant_on:
                self.assistant_ai = AssistantOpenAi(api_key, self.model)
                self.assistant_id = assistant_id
                self.thread_id = thread_id
                self.file_ids =[]

    def __del__(self):
        """
        Método chamado ao destruir a instância da classe.
        """
        print(f"Custo total: {round(self.total_cost, 2)}")
        self.logger.info(f"Encerrando chamadas (método __del__). Cuto total {self.total_cost}")

        if self.assistant_on and self.api_provider =='openai':
            for file_id in self.file_ids:
                try:
                    file_deletion_status = self.api_adapter.client.beta.assistants.files.delete(assistant_id=self.assistant_id,file_id=file_id)
                    self.logger.info(f'file deletion for {file_id}: {file_deletion_status}')
                except Exception as e:
                    self.logger.info(f'Erro ao deletar arquivo: {e}')

    def set_logs(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def add_message_to_context(self, role, message):
        """
        Appends a new message to the conversation history and returns the updated history.
        """
        if message == "exit":
            self.save()
            os._exit(1)
        elif message == "save":
            self.save()
            return "(saved)"
        return self.api_adapter.message(role, message, self.messages)
    
    def context_size_control(self, message: str):
        """
        Controls the size of the context to ensure it's within the model's token limit.
        Removes the oldest messages (except system messages) if the limit is exceeded.

        :param message: The message to be added to the conversation.
        """
        SAFETY_RESPONSE_INDEX = 0.85

        current_token_count = self.api_adapter.count_tokens(message)
        for msg in self.messages:
            current_token_count += self.api_adapter.count_tokens(msg['content'])
        
        token_limit = MODELS_TOKEN_LIMIT.get(self.model, 2048)

        while current_token_count > (token_limit*SAFETY_RESPONSE_INDEX) and self.messages:
            if self.messages[0]['role'] == 'system':
                if len(self.messages) > 1:
                    current_token_count -= self.api_adapter.count_tokens(self.messages[1]['content'])
                    del self.messages[1]
                else:
                    break
            else:
                current_token_count -= self.api_adapter.count_tokens(self.messages[0]['content'])
                del self.messages[0]

    def chat(self, message, messages=None, system_prompt=None, max_tokens=300, temperature=1, request_attempts=1):
        """
        Handles a chat message, saves or exits if special commands are used.

        :param message: The message from the user.
        :return: The response from the model.
        """
        if messages is None:
            messages = self.messages
        else:
            self.messages = messages
            
        if message == "exit":
            self.save()
            os._exit(1)
        elif message == "save":
            self.save()
            return "(saved)"
        

        self.context_size_control(message)

        input_token_count = self.api_adapter.count_tokens(message)
        self.logger.info(f"Input tokens for '{message}': {input_token_count}")
        
        self.logger.info(f"Calling {self.api_provider} API in chat method, with message: {message}")
        response = self.api_adapter.chat(
            message, 
            self.messages, 
            system_prompt, 
            max_tokens=max_tokens,
            temperature=temperature, 
            request_attempts=request_attempts
        )
        self.logger.info(f"Received response in chat method from {self.api_provider} API: {response}")
        response_token_count = self.api_adapter.count_tokens(response)
        self.logger.info(f"Response tokens for '{response}': {response_token_count}")

        cost = self.calculate_cost(input_token_count, response_token_count)
        self.total_cost += cost
        self.logger.info(f"Cost for this interaction: {cost}")
        
        return response

    def single_message_completion(self, user_message, system_message=None, output_type=None, max_tokens=4000, temperature=1, request_attempts=1):
        """
        Generate Response without history

        :param user_message: The message from the user.
        :return: The response from the model.
        """

        if output_type is not None and self.api_provider != 'openai':
            raise ValueError(f"The model '{self.model}' does not support custom output_type. Only Open AI models do.")
        
        if system_message is not None and self.api_provider == 'gemini':
            raise ValueError(f"The model '{self.model}' does not support system prompts")
        
        self.logger.info(f"Calling {self.api_provider} API in generate method, with message: {user_message}")
        response = self.api_adapter.single_message_completion(user_message, system_message, output_type, max_tokens, temperature, request_attempts)
        self.logger.info(f"Received response in generate method from {self.api_provider} API: {response}")

        return response


    def calculate_cost(self, input_tokens, output_tokens, print_cost=False):
        """
        Calculates the cost of a message in Brazilian Real (R$) based on the number of input and output tokens.

        :param input_tokens: The number of input tokens in the message.
        :param output_tokens: The number of output tokens in the response.
        :return: None, prints the cost directly in R$.
        """

        cost = self.api_adapter.calculate_cost(input_tokens, output_tokens)

        self.logger.info(f"Token usage: {input_tokens} input tokens, {output_tokens} output tokens")
        self.logger.info(f"Cost: R${cost:.4f}")

        if print_cost:
            print(f"CUSTO: R${cost:.4f} para {input_tokens} tokens de entrada e {output_tokens} tokens de saída")
        return cost
   
    def count_tokens(self, string: str, pricing = None):
        """Returns the number of tokens in a text string."""
        num_tokens = self.api_adapter.count_tokens(string)
        if pricing:
            cost = self.calculate_cost(num_tokens, 0) 
            return num_tokens, cost
        else:
            return num_tokens

    def save(self):
        """Saves the chat history to a JSON file."""
        try:
            ts = time.time()
            json_object = json.dumps(self.messages, indent=4)
            filename_prefix = self.messages[0]['content'][0:30]
            filename_prefix = re.sub('[^0-9a-zA-Z]+', '-', f"{filename_prefix}_{ts}")
            with open(f"models/chat_model_{filename_prefix}.json", "w") as outfile:
                outfile.write(json_object)
        except Exception as e:
            self.logger.error(f"Error while saving chat history: {e}")
            os._exit(1)

    def load(self, load_file):
        """Loads chat history from a file.

        :param load_file: Path to the file containing the chat history.
        """
        with open(load_file) as f:
            data = json.load(f)
            self.messages = data

    def clear_chat(self):
        """Clears the chat history."""
        self.messages = []
       
    def export_chat_to_txt(self, filename):
        """Exports the chat history to a text file.

        :param filename: The name of the text file to export to.
        """
        with open(filename, 'w') as file:
            for message in self.messages:
                role = message['role']
                content = message['content']
                file.write(f"{role}: {content}\n")
        print(f"Chat history exported to {filename}.")

    def get_chat_history(self):
        """Returns the chat history as a formatted string.

        :return: A string representing the chat history.
        """
        history = ""
        for message in self.messages:
            role = message['role']
            content = message['content']
            history += f"{role}: {content}\n"
        return history

    def set_model(self, model):
        """Sets the model used for chat completions.

        :param model: The model to be used.
        """
        self.model = model
        print(f"Model set to {model}.")

    def init_assistant(self, name, instructions):
        """
        Initializes the assistant by creating an assistant instance with the given name and instructions.
        This method is specific to the OpenAI provider and prepares the assistant for conversation.

        Parameters:
            name (str): The name of the assistant to be created.
            instructions (str): The instructions that define the behavior of the assistant.
        """

        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")
        if not self.assistant_on:
            raise Exception(f"Assistant not initialized. Try calling ChatApp with assistant_on as True")
        if not self.assistant_id:
            try:
                self.assistant_id = self.assistant_ai.create_assistant(name, instructions, tools=OPEN_AI_ASSISTANT_TOOLS)
                self.logger.info(f"Assistente Open AI ID: {self.assistant_id} Iniciado")
            except Exception as e:
                self.logger.error(f"Erro ao iniciar assistente: {e}")
                raise e(f"Erro ao iniciar assistente: {e}")
            
    def upload_file_to_assistant(self, file_path: str, purpose: str = 'assistants'):
        """
        Uploads a file to OpenAI and returns the file ID.

        Parameters:
            file_path (str): The path to the file to upload.
            purpose (str): The purpose of the file upload (default is 'assistants').

        Returns:
            str: The file ID of the uploaded file.
        """
        try:
            client = self.api_adapter.client
            file = client.files.create(
                file = open(f'./{file_path}', 'rb'),
                purpose=purpose,
            )
            return file.id
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            raise e


    def init_assistant_with_file(self, file_path: 'str | list[str]', name: str, instructions: str):
        """
        Handles the entire flow of creating an assistant with a specific file.

        Parameters:
            file_path (str | list[str]): The local path to the file to be uploaded. Or a list of local paths.
            name (str): Name of the assistant.
            instructions (str): Instructions for the assistant.
        """

        self.file_ids =[]
        if type(file_path) is str:
            file_path = [file_path]

        for file in file_path:
            file_id = self.upload_file_to_assistant(file)
            self.file_ids.append(file_id)
    
        if self.assistant_on:
            try:
                self.assistant_id = self.assistant_ai.create_assistant(name, instructions, tools=OPEN_AI_ASSISTANT_TOOLS, file_ids=self.file_ids)
                self.logger.info(f"Assistant created with ID: {self.assistant_id}")
                return file_id, self.assistant_id
            except Exception as e:
                self.logger.error(f"Error creating assistant with file: {e}")
                raise e
        else:
            raise Exception("Assistant functionality is not initialized.")

    def start_conversation_with_openai_assistant(self):
        """
        Initializes a new conversation thread. This method is specific to the OpenAI provider and
        must be called before sending messages to ensure that there is a thread in which to operate.
        """

        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")
        if not self.thread_id:
            try:
                self.thread_id = self.assistant_ai.create_thread()
                self.logger.info(f"Iniciando thread {self.thread_id} no assistant ID: {self.assistant_id}")
                return self.thread_id
            except Exception as e:
                self.logger.error(f"Erro ao iniciar thread no assistant ID: {self.assistant_id}: {e}")
                raise e(f"Erro ao iniciar thread: {e}")

    def send_message_to_assistant_return_thread_messages(self, message, max_checks = 1000, file_ids=None):
        """
        Sends a message to the initialized assistant within the created thread and retrieves the response.
        This method is specifically for interacting with the OpenAI Assistant API.

        Parameters:
            message (str): The message to send to the assistant.

        Returns:
            list: A list of messages including the assistant's response, if the run is completed.
        """

        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")
        if self.thread_id and self.assistant_id:
            self.assistant_ai.add_message_to_thread(self.thread_id, message, role='user', file_ids = file_ids)
            run_id = self.assistant_ai.run_assistant(self.thread_id, self.assistant_id)
            status = self.assistant_ai.get_run_status(self.thread_id, run_id)
            self.logger.info(f"Mensagem nova adicionada a thread {self.thread_id} no assistant ID: {self.assistant_id}: message")

            checks = 0
            while True and checks < max_checks:
                if status == 'completed':
                    response = self.assistant_ai.get_messages_from_thread(self.thread_id)
                    self.logger.info(f"Resposta nova adicionada a thread {self.thread_id} no assistant ID: {self.assistant_id}: {response}")
                    return response
                else:
                    time.sleep(1)
                    status = self.assistant_ai.get_run_status(self.thread_id, run_id)
                    checks +=1
        else:
            self.logger.error(f"Erro ao adicionar nova mensagem a thread de assistant: IDs de Assistente ou Thread não existem")
            raise Exception("Assistant or thread not initialized.")
        
    def get_last_message_response_from_assistant_chat(self, assistant_messages_history):
        """
        Retrieves last messages from assistant from a specified assistant_messages_history.

        Parameters:
            assistant_messages_history (list): List of messages between assistant and user in the open ai thread format.

        Returns:
            string: A string with the last assistant messages.
        """

        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")

        return self.assistant_ai.get_last_message_reponse_from_assistant_chat(assistant_messages_history)
    
    def save_assistant_ids(self, filename="assistant_thread_ids.json"):
        """
        Saves the assistant and thread IDs to a JSON file.
        """

        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")
        
        self.assistant_ai.save_ids(self.assistant_id, self.thread_id, filename)

    def load_assistant_ids(self, filename="assistant_thread_ids.json"):
        """
        Loads the assistant and thread IDs from a JSON file.
        """

        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")
        
        self.assistant_ai.load_ids(filename)

    def read_assistant_file(self, fileId:str):
        """
        Retrieves the content, in bytes, of a file stored on OpenAI's servers
        """
        if self.api_provider != 'openai':
            raise Exception(f"Only OpenAI models support assistant function. You are using {self.api_provider}")
        
        return self.assistant_ai.read_generated_file(fileId)
        