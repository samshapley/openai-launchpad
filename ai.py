import base64
import io
import json
import os
from datetime import datetime
from typing import List
import sounddevice as sd
import soundfile as sf
import openai
import tiktoken
import numpy as np
from numpy import dot
from numpy.linalg import norm
from PIL import Image as PilImage
from scipy.io.wavfile import write
from typing import Union, List
from collections import defaultdict
from helpers import log
import helpers as h
from openai import BadRequestError
from datetime import timedelta
import requests
import tool_manager as tm

## get openai api key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

logging = True

class Chat:

    def __init__(self, system="", model="gpt-3.5-turbo"):
        """
        Initializes the Chat class with a system message, model, and OpenAI client.

        Args:
            system (str): A system message to start the conversation.
            model (str): The model to be used for chat completions. Defaults to 'gpt-3.5-turbo'.
        """
        self.system = system
        self.openai = openai
        self.messages = [{"role": "system", "content": system}]
        self.model = model
        self.encoding = tiktoken.encoding_for_model(self.model)
        log(logging, f"Initalized Chat class with model {model}", "green")

    def update_messages(self, messages: List[dict]) -> None:
        """
        Updates the messages list with a new list of messages.

        Args:
            messages (List[dict]): A list of message objects.
        """
        self.messages = messages
    
    @staticmethod
    def _augment_phrases(phrases: List[str], augment: bool) -> List[str]:
        """
        Augments phrases by adding spaces before and after, changing case, and ensuring uniqueness.
        Used within logit bias to suppress variations of provided phrases.

        Args:
            phrases (List[str]): A list of phrases to augment.
            augment (bool): A flag to determine whether to augment phrases or not.

        Returns:
            List[str]: A list of unique augmented phrases.
        """
        if not augment:
            return phrases

        def _iter():
            for p in phrases:
                yield from (" " + p, p + " ", p.lower(), p.upper(), p.capitalize(), p.title())
 
        return list(set(_iter()))

    def create_logit_bias(self, suppressed_phrases, bias=None, augment: bool = False) -> dict:
        """
        Creates a logit bias dictionary for suppressing phrases in the chat completion.

        Args:
            suppressed_phrases (Union[List[str], Dict[str, int]]): Phrases to be suppressed with a global bias, or a dict of phrases with individual biases.
            bias (int, optional): The global amount of bias to apply for suppression if suppressed_phrases is a list. Ignored if suppressed_phrases is a dict.
            augment (bool, optional): Whether to augment the phrases for suppression. Defaults to False.

        Returns:
            logit_bias_dict (dict): A dictionary with tokens as keys and the bias as values.
        """
        if isinstance(suppressed_phrases, dict):
            phrases = list(suppressed_phrases.keys())
        else:
            phrases = self._augment_phrases(suppressed_phrases, augment)

        logit_bias_dict = {}
        for phrase in phrases:
            tokens = list(set([t for t in self.encoding.encode(phrase)]))
            phrase_bias = suppressed_phrases[phrase] if isinstance(suppressed_phrases, dict) else bias
            for t in tokens:
                logit_bias_dict[t] = phrase_bias

        return logit_bias_dict

    def chat_completion(
        self, 
        prompt: str = "", 
        memories: bool = True, 
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = None, 
        json: bool = False, 
        seed: int = None,
        logit_bias: dict = {},
        return_logprobs: Union[bool, None] = False,
        top_logprobs: Union[int, None] = None,
        model: str = None,
        speak: bool = False,
        stream: bool = False,
        messages: List[dict] = None,
        tools: List[dict] = None,
        tool_choice: Union[str, dict] = "auto",
        return_tool_calls: bool = False,
        return_messages: bool = False,
        ) -> dict:
        """
        Generates a chat completion using the OpenAI API. Full control over all aspects of the API is native to this method.

        Args:
            prompt (str): The user's input prompt for the chat.
            memories (bool, optional): Whether to retain conversation context. Defaults to True.
            temperature (float, optional): Sampling temperature to use. Defaults to 1.0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 1.0.
            frequency_penalty (float, optional): Adjusts frequency of token usage. Defaults to 0.0.
            presence_penalty (float, optional): Adjusts presence of token usage. Defaults to 0.0.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to None.
            json (bool, optional): Whether to return the response in JSON format. Defaults to False.
            seed (int, optional): A seed for deterministic completions. Defaults to None.
            logit_bias (dict, optional): A dictionary of logit biases. Defaults to None.
            return_logprobs (Union[bool, None], optional): Whether to return log probabilities of the output tokens. Defaults to False.
            top_logprobs (Union[int, None], optional): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
            model (str, optional): The model to use for the chat completion if different from initialization. Defaults to None.
            speak (bool, optional): Whether to speak the chat completion. Defaults to False.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            messages (List[dict], optional): A list of messages to use for the request, i.e if you wish to overwrite. Defaults to None.
            tools (List[dict], optional): A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. Defaults to None.
            tool_choice (Union[str, dict], optional): Controls which (if any) function is called by the model. "none" means the model will not call a function and instead generates a message. "auto" means the model can pick between generating a message or calling a function. Specifying a particular function via {"type": "function", "function": {"name": "my_function"}} forces the model to call that function. "none" is the default when no functions are present. Defaults to "auto".
            return_tool_calls (bool, optional): Whether to return tool calls in output dict. Defaults to False.
            return_messages (bool, optional): Whether to return messages in output dict. Defaults to False.

        Returns:
            completion_dict (dict): A dictionary containing all the information about the completion with the following keys:
                - 'response' (str): The text of the chat completion.
                - 'prompt' (str): The original prompt provided by the user.
                - 'model' (str): The model used for the chat completion.
                - 'parameters' (dict): A dictionary of the parameters used for the chat completion.
                    - 'temperature' (float): Sampling temperature used.
                    - 'top_p' (float): Nucleus sampling parameter used.
                    - 'frequency_penalty' (float): Frequency penalty used.
                    - 'presence_penalty' (float): Presence penalty used.
                    - 'max_tokens' (int): Maximum number of tokens used.
                    - 'seed' (int): The seed used for deterministic completions.
                - 'logit_bias' (dict): A dictionary of logit biases used in the completion.
                - 'id' (str): The unique identifier for the completion.
                - 'created' (int): The Unix timestamp when the completion was created.
                - 'system_fingerprint' (str): The backend configuration that the model runs with.
                - 'finish_reason' (str): The reason the completion ended.
                - 'logprobs' (list, optional): A list of log probabilities of the output tokens, if requested.
                - 'tool_calls' (list, optional): A list of tool calls in the output dict, if requested.
                - 'messages' (list, optional): A list of messages in the output dict, if requested.
                    - Each message (dict) contains:
                        - 'role' (str): The role of the message sender ('user', 'assistant', etc.).
                        - 'content' (list): A list of content objects, each with a 'type' key (e.g., 'text', 'image_url') and corresponding content.
        """
        # Check if return_tool_calls is True but tools is None
        if return_tool_calls and tools is None:
            raise ValueError("return_tool_calls cannot be True if tools is None")

        # Set up model if provided, otherwise use self.model.
        model = model or self.model
        # Create the messages, use self.messages if messages is not provided.
        self.messages = messages or self.messages
        # Set the response format based on the json flag.
        response_format = {"type": "json_object"} if json else {"type": "text"}


        # Create the user message with the prompt
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }

        # Append the user message to the messages list
        self.messages.append(user_message)

            # Prepare the API call arguments
        api_call_args = {
            "model": model,
            "messages": self.messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
            "logit_bias": logit_bias,
            "logprobs": return_logprobs,
            "top_logprobs": top_logprobs,
            "stream": stream,
        }

        # Include 'tools' only if it is present
        if tools is not None:
            api_call_args["tools"] = tools
            api_call_args["tool_choice"] = tool_choice


        # Make the API call
        log(logging, "Making Chat Completion API call...", "purple")
        
        completion = self.openai.chat.completions.create(**api_call_args)

        completion_content = ""
        
        if stream:
            current_tool_call = None
            tool_calls = []
            id = None
            finish_reason = None
            system_fingerprint = None
            created = None

            # Initialize logprobs list to accumulate logprobs from all chunks
            logprobs = []

            for chunk in completion:

                id = chunk.id                                  # Store the completion id
                created = chunk.created                        # Store the completion created timestamp (in seconds)
                system_fingerprint = chunk.system_fingerprint  # Store the system fingerprint
                finish_reason = chunk.choices[0].finish_reason # Store the completion finish reason

                # Extract logprobs if available and requested
                if return_logprobs:
                    chunk_logprobs = h.to_dict(chunk.choices[0].logprobs)
                    if chunk_logprobs:
                        for logprob in chunk_logprobs['content']:
                                logprobs.append(logprob)

                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end='')
                    completion_content += delta.content  # Accumulate the content

                if delta.tool_calls:
                    # Handle tool calls
                    for tc in delta.tool_calls:
                        if tc.id:
                            # Start of a new tool call grouping
                            if current_tool_call:
                                tool_calls.append(current_tool_call)
                            current_tool_call = {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            log(logging, f"\ntool call: {tc.function.name}\n", "blue")
                        else:
                            # Continuation of the current tool call
                            if current_tool_call:
                                current_tool_call["function"]["arguments"] += tc.function.arguments
                            else:
                                # This is a case where there's no ID and no current tool call
                                pass

            print("\n")

            # Append the last tool call if it exists
            if current_tool_call:
                tool_calls.append(current_tool_call)

        else: 
            id = completion.id                                                  # Extract the completion id
            created = completion.created                                        # The Unix timestamp (in seconds) when the completion was created.
            system_fingerprint = completion.system_fingerprint                  # This fingerprint represents the backend configuration that the model runs with. Useful in conjuction with seed to understand determinism.
            finish_reason = completion.choices[0].finish_reason                 # The reason the completion ended.
            completion_content = completion.choices[0].message.content or ""    # The content of the completion.
            tool_calls = h.to_dict(completion.choices[0].message.tool_calls)  # Extract the completion tool calls to dict.
            if return_logprobs:
                logprobs   = h.to_dict(completion.choices[0].logprobs.content)  # Extract the completion logprobs to dict

        if memories:
            # Construct a new message dictionary to hold the assistant's response.
            new_message = {"role": "assistant", "content": completion_content if completion_content else None}
            # If there are tool calls, add them to the message
            if tool_calls:
                new_message["tool_calls"] = tool_calls
            # Append the new message to the messages list
            self.messages.append(new_message)
        else: 
            # If memories is False, reset the messages list after each call
            self.messages = messages or [{"role": "system", "content": self.system}]

        # Construct the output dictionary dynamically based on the flags. Set parameters to defaults if not provided.
        completion_dict = {
            "response": completion_content,
            "prompt": prompt,
            "model": model,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "seed": seed,
                        },
            "logit_bias": logit_bias,
            "id": id,
            "created": created,
            "system_fingerprint": system_fingerprint,
            "finish_reason": finish_reason,
        }

        if return_logprobs:
            completion_dict["logprobs"] = logprobs

        if return_tool_calls:
            completion_dict["tool_calls"] = tool_calls

        if return_messages:
            completion_dict["messages"] = self.messages

        # If speak is True, generate audio from the completion text
        if speak:
            log(logging, f"Generating speech...", "purple")
            audio = Audio()
            audio.speak(completion_content)

        return completion_dict
    
    def use_tools(self, available_functions):
        self.available_functions = available_functions
        messages = tm.use_tools(self.messages, available_functions)
        self.messages = messages
        return self.messages
class Vision:
    def __init__(self, model="gpt-4-vision-preview", system=""):
        """
        Initializes the Vision class with a model, system message, and OpenAI client.

        Args:
            model (str): The model to be used for vision tasks. Defaults to 'gpt-4-vision-preview'.
            system (str): A system message to start the conversation.
        """
        self.openai = openai
        self.model = model
        self.messages = [{"role": "system", "content": system}]
        self.system = system
        log(logging, f"Initalized Vision class with model {model}.", "green")

    def _encode_image(self, image_path):
        """
        Encodes an image to a base64 string.

        Args:
            image_path (str): The path to the image file to be encoded.

        Returns:
            str: The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    def _construct_image_message(self, image_path: str = None, image_url: str = None, detail: str = "auto") -> dict:
        """
        Constructs the message object for a single image path using the encode_images, or is a url is provided.

        Args:
            image_path (str): The path to the image file.
            image_url (str): The URL of the image.
            detail (str): The level of detail requested for the image. Options are 'low', 'high', or 'auto':
                - 'low': The model processes a low-res 512px x 512px version of the image, using a budget of 65 tokens. 
                        This setting is faster and uses fewer tokens, suitable for use cases that don't require high detail.
                - 'high': The model first sees the low-res image, then processes detailed 512px square crops of the input image. 
                        Each crop uses a budget of 129 tokens, allowing for a more detailed understanding of the image.
                - 'auto': The model decides whether to use 'low' or 'high' based on the input image size. This is the default setting.
            Defaults to "auto".

        Returns:
            image_message: The message object for the image.
                - 'type' (str): The type of message, i.e. "image_url".
                - 'image_url' (dict): A dictionary containing the URL of the image and the detail level.
                    - 'url' (str): The URL of the image.
                    - 'detail' (str): The level of detail requested for the image.
        """

        if image_path:
            encoded_string = self._encode_image(image_path)
            image_data = f"data:image/jpeg;base64,{encoded_string}"
        elif image_url:
            image_data = image_url
        else:
            raise ValueError("Either image_path or image_url must be provided.")

        image_message = {
            "type": "image_url",
            "image_url": {
                "url": image_data,
                "detail": detail
            }
        }
               
        return image_message
    
    def vision_completion(
        self, 
        prompt: str, 
        image_paths: List[str] = None,
        image_urls: List[str] = None,
        detail: str = "auto",
        memories: bool = True, 
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 250,
        seed: int = None,
        # logit_bias: dict = {}, # Currently not supported for vision completions.
        # return_logprobs: Union[bool, None] = False, # Currently not supported for vision completions.
        # top_logprobs: Union[int, None] = None, # Currently not supported for vision completions.
        stream: bool = False,
        speak: bool = False,
        messages: List[dict] = None,
        return_messages: bool = False,
        ) -> dict:
        """
        Generates a vision completion using the OpenAI API.

        Args:
            prompt (str): The user's input prompt for the vision task.
            image_paths (List[str], optional): A list of image paths to include in the request.
            image_urls (List[str], optional): A list of image URLs to include in the request.
            detail (str): The level of detail requested for the images. Options are 'low', 'high', or 'auto':
                - 'low': The model processes a low-res 512px x 512px version of the image, using a budget of 65 tokens. 
                        This setting is faster and uses fewer tokens, suitable for use cases that don't require high detail.
                - 'high': The model first sees the low-res image, then processes detailed 512px square crops of the input image. 
                        Each crop uses a budget of 129 tokens, allowing for a more detailed understanding of the image.
                - 'auto': The model decides whether to use 'low' or 'high' based on the input image size. This is the default setting.
            Defaults to "auto".
            memories (bool, optional): Whether to retain conversation context. Defaults to True.
            temperature (float, optional): Sampling temperature to use. Defaults to 1.0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 1.0.
            frequency_penalty (float, optional): Adjusts frequency of token usage. Defaults to 0.0.
            presence_penalty (float, optional): Adjusts presence of token usage. Defaults to 0.0.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 250.
            seed (int, optional): A seed for deterministic completions. Defaults to None.
            speak (bool, optional): Whether to speak the vision completion. Defaults to False.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            messages (List[dict], optional): A list of messages to use for the request, i.e if you wish to overwrite. Defaults to None.
            return_messages (bool, optional): Whether to return messages in output dict. Defaults to False.

        Returns:
            dict: A dictionary containing all the information about the completion with the following keys:
                - 'response' (str): The text of the vision completion.
                - 'prompt' (str): The original prompt provided by the user.
                - 'model' (str): The model used for the vision completion.
                - 'parameters' (dict): A dictionary of the parameters used for the vision completion.
                    - 'temperature' (float): Sampling temperature used.
                    - 'top_p' (float): Nucleus sampling parameter used.
                    - 'frequency_penalty' (float): Frequency penalty used.
                    - 'presence_penalty' (float): Presence penalty used.
                    - 'max_tokens' (int): Maximum number of tokens used.
                    - 'seed' (int): The seed used for deterministic completions.
                - 'id' (str): The unique identifier for the completion.
                - 'created' (int): The Unix timestamp when the completion was created.
                - 'system_fingerprint' (str): The backend configuration that the model runs with.
                - 'finish_reason' (str): The reason the completion ended.
                - 'messages' (list, optional): A list of messages in the output dict, if requested.
                    - Each message (dict) contains:
                        - 'role' (str): The role of the message sender ('user', 'assistant', etc.).
                        - 'content' (list): A list of content objects, each with a 'type' key (e.g., 'text', 'image_url') and corresponding content.
        """

        # Use the provided messages if available, otherwise use self.messages
        messages = messages or self.messages

        # Create the user message with the prompt
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }

        # If image paths are provided, construct the image messages and add them to the user message content
        if image_paths:
            for image_path in image_paths:
                image_message = self._construct_image_message(image_path, detail)
                user_message["content"].append(image_message)

        # If image URLs are provided, construct the image messages and add them to the user message content
        if image_urls:
            for image_url in image_urls:
                image_message = self._construct_image_message(image_url=image_url, detail=detail)
                user_message["content"].append(image_message)

        # Append the user message to the messages list
        messages.append(user_message)

        # Prepare the API call arguments
        api_call_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            # "logit_bias": logit_bias,
            # "logprobs": return_logprobs,
            # "top_logprobs": top_logprobs,
            "stream": stream,
        }

        # Make the API call
        log(logging, "Making Vision Completion API call...", "purple")
        completion = self.openai.chat.completions.create(**api_call_args)

        completion_content = ""
        id = None
        created = None
        system_fingerprint = None
        finish_reason = None
        # logprobs = []

        if stream:
            for chunk in completion:
                id = chunk.id
                created = chunk.created
                system_fingerprint = chunk.system_fingerprint
                finish_reason = chunk.choices[0].finish_reason

                # if return_logprobs:
                #     chunk_logprobs = h.to_dict(chunk.choices[0].logprobs)
                #     if chunk_logprobs:
                #         logprobs.extend(chunk_logprobs['content'])

                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end='')
                    completion_content += delta.content
        else:
            id = completion.id
            created = completion.created
            system_fingerprint = completion.system_fingerprint
            finish_reason = completion.choices[0].finish_reason
            completion_content = completion.choices[0].message.content or ""
            # if return_logprobs:
            #     logprobs = h.to_dict(completion.choices[0].logprobs.content)

        print("\n")

        if memories:
            # Construct a new message dictionary to hold the assistant's response.
            new_message = {"role": "assistant", "content": completion_content if completion_content else None}

            # Append the new message to the messages list
            self.messages.append(new_message)
        else: 
            # If memories is False, reset the messages list after each call
            self.messages = messages or [{"role": "system", "content": self.system}]

        completion_dict = {
            "response": completion_content,
            "prompt": prompt,
            "model": self.model,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "seed": seed,
            },
            # "logit_bias": logit_bias,
            "id": id,
            "created": created,
            "system_fingerprint": system_fingerprint,
            "finish_reason": finish_reason,
        }

        # if return_logprobs:
        #     completion_dict["logprobs"] = logprobs

        if return_messages:
            completion_dict["messages"] = self.messages

        # If speak is True, generate audio from the completion text
        if speak:
            audio = Audio()
            audio.speak(completion_content)

        return completion_dict
 
class Embeddings:
    def __init__(self, embedding_model="text-embedding-ada-002", encoding_format=None):
        """
        Initializes the Embeddings class with an OpenAI client, embedding model, and encoding format.

        Args:
            embedding_model (str): The model to be used for generating embeddings. Defaults to 'text-embedding-ada-002'.
            encoding_format (str, optional): The encoding format to be used for the embeddings. Defaults to None.
        """
        self.openai = openai
        self.embedding_model = embedding_model
        self.encoding_format = encoding_format
        log(logging, f"Initalized Embeddings class with model {embedding_model}.", "green")

    def create_embeddings(self, texts):
        """
        Creates embeddings for a list of texts using the specified embedding model.

        Args:
            texts (list): A list of strings for which to generate embeddings.

        Returns:
            list: A list of dictionaries, each containing the original text and its corresponding embedding.
        """
        # Ensure texts is a list
        if not isinstance(texts, list):
            raise ValueError("Input should be a list of strings.")

        # Create embeddings.
        embeddings = self.openai.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format=self.encoding_format
        )
        # Combine original texts and their embeddings into a list of dictionaries
        embeddings_objects = [{"text": text, "embedding": embedding.embedding} for text, embedding in zip(texts, embeddings.data)]

        return embeddings_objects
     
    @staticmethod
    def cosine_similarity(vector1, vector2):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vector1 (list): The first vector.
            vector2 (list): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return dot(vector1, vector2) / (norm(vector1) * norm(vector2))

    def string_similarity(self, string1, string2):
        """
        Calculate the cosine similarity between the embeddings of two strings.

        Args:
            string1 (str): The first string to compare.
            string2 (str): The second string to compare.

        Returns:
            float: The cosine similarity between the embeddings of the two strings.
        """
        # Create embeddings for the strings
        embedding1 = self.create_embeddings([string1])[0]["embedding"]
        embedding2 = self.create_embeddings([string2])[0]["embedding"]

        # Calculate and return the cosine similarity
        return self.cosine_similarity(embedding1, embedding2)
    
class Images:
    def __init__(self, model="dall-e-3"):
        """
        Initializes the Image class with an OpenAI client and model.

        Args:
            model (str): The model to be used for image generation. Defaults to 'dall-e-3'.
        """
        self.openai = openai
        self.model = model
        self.valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        log(logging, f"Initalized Images class with model {model}.", "green")

    def _check_size(self, size):
        """
        Checks if the provided size is valid for the model.

        Args:
            size (str): The size to be checked.

        Raises:
            ValueError: If the size is not in the list of valid sizes.
        """
        if size not in self.valid_sizes:
            raise ValueError(f"Invalid size. Must be one of {self.valid_sizes} for {self.model} model.")

    @staticmethod
    def _display_image(image_data_or_url, display_image=True, save_image=True, save_dir='media/images', image_name='image.png'):
        """
        Displays and/or saves the image from the provided base64-encoded image data or URL.

        Args:
            image_data_or_url (str): The base64-encoded image data or the URL of the image.
            display_image (bool): Whether to display the image. Defaults to True.
            save_image (bool): Whether to save the image to the filesystem. Defaults to True.
            save_dir (str): The directory where the image will be saved. Defaults to 'media/images'.
            image_name (str): The name of the image file. Defaults to 'image.png'.
        """
        image_path = os.path.join(save_dir, image_name)
        if image_data_or_url.startswith('http'):  # If it's a URL
            response = requests.get(image_data_or_url)
            image_data = response.content
        else:  # Base64 data
            image_data = base64.b64decode(image_data_or_url)

        with io.BytesIO(image_data) as image_buffer, PilImage.open(image_buffer) as image:
            if display_image:
                image.show()
            if save_image:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                image.save(image_path)

    def generate_image(self, prompt, fix_prompt=False, size="1024x1024", quality=None , style=None , response_format="url", display_image=False, save_image=True, image_name=None, save_dir='media/images', return_b64_json=False):
        """
        Generates an image based on the provided prompt and parameters.

        Args:
            prompt (str): The prompt to generate the image from.
            fix_prompt (bool): If True, DALLE-3 will be instructed to use the prompt as-is, i.e no auto revision of prompt. Defaults to False.
            size (str): The size of the generated images. Defaults to "1024x1024".
            quality (str, optional): The quality of the generated images. Defaults to None. "hd" creates images with finer details and greater consistency across the image.
            style (str, optional): The style of the generated images. Defaults to None. Must be one of "vivid" (hyper-real images) or natural (realistic images).
            response_format (str): The format of the response. Defaults to "url".
            display_image (bool): Whether to display the generated images. Defaults to False.
            save_image (bool): Whether to save the generated images. Defaults to True.
            image_name (str, optional): The name of the image file. If none is provided, defaults to timestamp.png.
            save_dir (str): The directory where the image will be saved. Defaults to 'media/images'.
            return_b64_json (bool): Whether to return the base64-encoded JSON string of the generated image. Defaults to False as this is a very large string.

        Returns:
            dict: A JSON object containing the following keys:
                - 'url' (str, optional): The URL of the generated image if response_format is "url".
                - 'b64_json' (str, optional): The base64-encoded JSON string of the generated image if response_format is "b64_json" and return_b64_json is True.
                - 'user_prompt' (str): The original prompt provided by the user.
                - 'revised_prompt' (str): The prompt that was used to generate the image, if there was any revision to the prompt.
                - 'prompt_modified' (bool): A flag indicating whether the prompt was modified or not.
                - 'size' (str): The size of the generated image.
                - 'quality' (str): The quality of the generated image, if any.
                - 'style' (str): The style of the generated image, if any.
                - 'path_to_image' (str): The file system path to the saved image if save_image is True.
                - 'timestamp' (str): The ISO 8601 timestamp when the image was generated.
        """
        self._check_size(size)

        if return_b64_json and response_format == "url":
            raise ValueError("return_b64_json cannot be True if response_format is 'url'.")

        user_prompt = prompt

        if fix_prompt:
            prompt += "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: "

        log(logging, "Making Image Generation API call...", "purple")

        api_call_args = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": response_format,
        }

        if quality:
            api_call_args["quality"] = quality
        if style:
            api_call_args["style"] = style

        try:
            response = self.openai.images.generate(**api_call_args)

            created = response.created # The Unix timestamp when the image was generated

            # After the API call
            if save_image or display_image:
                if not image_name:
                    image_name = f'{created}.png'
                image_path = os.path.join(save_dir, image_name)

                if response_format == "b64_json":
                    image_data_or_url = response.data[0].b64_json
                elif response_format == "url":
                    image_data_or_url = response.data[0].url

                self._display_image(image_data_or_url, display_image, save_image, save_dir=save_dir, image_name=image_name)

            revised_prompt = response.data[0].revised_prompt # The prompt that was used to generate the image, if there was any revision to the prompt.

            prompt_modified = revised_prompt != user_prompt # Whether the prompt was modified or not.

            image_generation = {
                "url": None if response_format == "b64_json" else response.data[0].url,
                "user_prompt": user_prompt,
                "revised_prompt": revised_prompt,
                "prompt_modified": prompt_modified,
                "size": size,
                "quality": quality,
                "style": style,
                "model": self.model,
                "timestamp": created,
                "status": "success"
            }

            if save_image:
                image_generation["path_to_image"] = image_path

            if return_b64_json and response_format == "b64_json":
                image_generation["b64_json"] = response.data[0].b64_json

            return image_generation

        except BadRequestError as e:
            # Access the error details correctly
            error_details = e.args[0]
            if 'content_policy_violation' in error_details:
                print("Content policy violation:", error_details)

            else:
                print("BadRequestError:", error_details)

            created = datetime.now().isoformat()

            image_generation = {
                "url": None,
                "user_prompt": user_prompt,
                "revised_prompt": None,
                "prompt_modified": None,
                "size": size,
                "quality": quality,
                "style": style,
                "path_to_image": None,
                "timestamp": created,
                "status": "error"
            }

            image_generation["content_policy_violation"] = True if 'content_policy_violation' in error_details else False

            return image_generation

class Audio:
    def __init__(self):
        self.openai = openai
        log(logging, f"Initalized Audio class.", "green")

    def speak(self, text, model="tts-1", voice="onyx", response_format="opus", speed=1.0, play_audio = True, save_audio = False, file_name=None, save_dir='media/audio', return_audio=False):
        """
        Generates audio from the input text and plays it in real-time. Optionally saves the audio to a file.

        Args:
            text (str): The text to generate audio for. The maximum length is 4096 characters.
            model (str): One of the available TTS models: tts-1 or tts-1-hd. Defaults to 'tts-1'.
            voice (str): The voice to use when generating the audio. Supported voices are alloy, echo, fable, onyx, nova, and shimmer.
            response_format (str): The format to audio in. Supported formats are mp3, opus, aac, and flac. Defaults to 'opus'.
            speed (float): The speed of the generated audio. Select a value from 0.25 to 4.0. Defaults to 1.0.
            play_audio (bool): Whether to play the generated audio out loud. Defaults to True.
            save_audio (bool): Whether to save the generated audio. Defaults to False.
            file_name (str): The name of the audio file to save. If none is provided, defaults to timestamp.{response_format}.
            save_dir (str): The directory where the audio will be saved. Defaults to 'media/audio'.
            return_audio (bool): Whether to return the audio as a numpy array. Defaults to False.

        Returns:
            speech_dict: Information about the generated speech with the following keys:
                - 'text' (str): The original text provided for speech generation.
                - 'speed' (float): The speed at which the audio was generated.
                - 'voice' (str): The voice used to generate the audio.
                - 'model' (str): The TTS model used for generating the audio.
                - 'response_format' (str): The format of the generated audio file.
                - 'file_path' (str, optional): The file path where the audio was saved, if saving was requested.
                - 'start_time' (str): The ISO 8601 timestamp when the audio generation started.
                - 'end_time' (str): The ISO 8601 timestamp when the audio generation ended.
                - 'duration_ms' (float): The duration of the generated audio in milliseconds.
                - 'words_per_minute' (float): The number of words spoken per minute in the generated audio.
                - 'audio' (numpy array, optional): The audio as a numpy array, if return_audio is True.
                - 'sample_rate' (int, optional): The sample rate of the audio, if return_audio is True.
        """

        log(logging, "Making Speech API call...", "purple")

        spoken_response = self.openai.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
            speed=speed
        )
        
        buffer = io.BytesIO()
        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)

        # Calculate the length of the audio in milliseconds
        with sf.SoundFile(buffer, 'r') as sound_file:
            data = sound_file.read(dtype='int16')
            duration_ms = len(data) / sound_file.samplerate * 1000
            start_time = datetime.now()
            end_time = start_time + timedelta(milliseconds=duration_ms)
            words = len(text.split())
            words_per_minute = words / (duration_ms / 60000.0)

        # Save the audio file if requested
        if save_audio:
            if not file_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f'{timestamp}.{response_format}'
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            path_to_recording = os.path.join(save_dir, file_name)

            with open(path_to_recording, 'wb') as f:
                f.write(buffer.getvalue())

        # Play the audio if requested
        if play_audio:
            sd.play(data, sound_file.samplerate)
            sd.wait()

        speech_dict = {
            "text": text,
            "speed": speed,
            "voice": voice,
            "model": model,
            "response_format": response_format,
            "file_path": path_to_recording if save_audio else None,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_ms": duration_ms,
            "words_per_minute": words_per_minute
        }
    
        if return_audio:
            speech_dict["audio"] = data
            speech_dict["sample_rate"] = sound_file.samplerate
   
        return speech_dict
    
    def transcribe(self, file_path, model="whisper-1", language=None, prompt=None, response_format="json", temperature=0):
        """
        Transcribes audio into the input language using the specified model.

        Args:
            file_path (str): The path to the audio file to transcribe.
            model (str): ID of the model to use. Defaults to 'whisper-1'.
            language (str, optional): The language of the input audio in ISO-639-1 format.
            prompt (str, optional): An optional text to guide the model's style.
            response_format (str, optional): The format of the transcript output. Defaults to 'json'.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            The transcribed text.
        """

        log(logging, "Making Speech Transcription API call...", "purple")
        with open(file_path, "rb") as audio_file:
            transcription = self.openai.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )
        return transcription.text    

    def translate(self, file_path, model="whisper-1", prompt=None, response_format="json", temperature=0):
        """
        Translates audio into English using the specified model.

        Args:
            file_path (str): The path to the audio file to translate.
            model (str): ID of the model to use. Defaults to 'whisper-1'.
            prompt (str, optional): An optional text to guide the model's style or continue a previous audio segment.
            response_format (str, optional): The format of the transcript output. Defaults to 'json'.
            temperature (float, optional): The sampling temperature. Defaults to 0.

        Returns:
            The translated text.
        """

        log(logging, "Making Speech Translation API call...", "purple")
        with open(file_path, "rb") as audio_file:
            translation = self.openai.audio.translations.create(
                model=model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )
        return translation.text
    
    def record_from_microphone(self, sample_rate=44100, file_path='recording_temp.wav', threshold=0.02, silence_duration=2, min_duration=2):
        """
        Records audio from the microphone, stopping when the sound drops below the threshold for a specified duration after a minimum recording time has passed.

        Args:
            sample_rate (int): The sample rate of the recording. Defaults to 44100 Hz.
            file_path (str): The file path where the audio will be saved. Defaults to 'recording_temp.wav'.
            threshold (float): The sound level threshold to stop recording. Defaults to 0.01.
            silence_duration (int): The duration of silence in seconds to stop recording. Defaults to 2.
            min_duration (int): The minimum duration of the recording in seconds. Defaults to 2.

        Returns:
            The file path of the recorded audio.
        """
        # Prepare the recording stream
        with sd.InputStream(samplerate=sample_rate, channels=1) as stream:
            log(logging, "Recording started. Speak into the microphone.", "blue")
            buffer = []
            silence_counter = 0
            start_time = datetime.now()
            
            while True:
                data, overflowed = stream.read(sample_rate * silence_duration)
                if overflowed:
                    raise RuntimeError("Overflowed")
                buffer.append(data)
                current_time = datetime.now()
                elapsed_time = (current_time - start_time).total_seconds()
                
                if np.abs(data).mean() < threshold:
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # Check if the minimum duration has passed and if the silence duration has been reached
                if elapsed_time > min_duration and silence_counter >= silence_duration:
                    break
            
            log(logging, "Recording ended.", "red")
            recording = np.concatenate(buffer, axis=0)

        # Save the recording to a file
        write(file_path, sample_rate, recording)  # Save as WAV file
        return file_path

    def record_and_transcribe(self, model="whisper-1", language=None, prompt=None, response_format="json", temperature=0):
        """
        Records audio from the microphone and transcribes it.

        Args:
            model (str): ID of the model to use for transcription. Defaults to 'whisper-1'.
            language (str, optional): The language of the input audio in ISO-639-1 format.
            prompt (str, optional): An optional text to guide the model's style.
            response_format (str, optional): The format of the transcript output. Defaults to 'json'.
            temperature (float, optional): The sampling temperature. Defaults to 0.
        
        Returns:
            dict: The transcription result in the specified format.
        """
        file_path = self.record_from_microphone()
        
        transcript = self.transcribe(file_path, model, language, prompt, response_format, temperature)

        # delete the temporary recording file
        os.remove(file_path)

        return transcript
    
    def record_and_translate(self, model="whisper-1", prompt=None, response_format="json", temperature=0):
        """
        Records audio from the microphone and translates it.

        Args:
            model (str): ID of the model to use for translation. Defaults to 'whisper-1'.
            prompt (str, optional): An optional text to guide the model's style.
            response_format (str, optional): The format of the translation output. Defaults to 'json'.
            temperature (float, optional): The sampling temperature. Defaults to 0.
        
        Returns:
            The translation result in the specified format.
        """
        file_path = self.record_from_microphone()
        
        translation = self.translate(file_path, model, prompt, response_format, temperature)

        # delete the temporary recording file
        os.remove(file_path)

        return translation
  
class Files:
    def __init__(self):
        self.openai = openai
        log(logging, f"Initalized Files class.", "green")

    def upload_file(self, file_path, purpose):
        """
        Uploads a file for a specific purpose.

        Args:
            file_path (str): The path to the file to be uploaded.
            purpose (str): The purpose of the file. Supported values are 'fine-tune', 'assistants', etc.

        Returns:
            The uploaded file.
        """
        try:
            log(logging, f"Uploading file {file_path}...", "purple")
            with open(file_path, "rb") as file:
                return self.openai.files.create(file=file, purpose=purpose)
        except Exception as e:
            log(logging, f"Failed to upload file {file_path}: {e}", "red")
            return None
        
    def list_files(self, purpose=None):
        """
        Lists all files for a specific purpose (or all files if purpose is not provided).

        Args:
            purpose (str, optional): The purpose of the files to list. Supported values are 'fine-tune', 'assistants', etc.

        Returns:
            A list of file objects.
        """
        try:
            log(logging, f"Listing files...", "purple")
            return self.openai.files.list(purpose=purpose)
        except Exception as e:
            log(logging, f"Failed to list files: {e}", "red")
            return None

    def delete_file(self, file_id):
        """
        Deletes a specific file.

        Args:
            file_id (str): The ID of the file to delete.

        Returns:
            The deletion status of the file.

        {
        "id": "file-abc123",
        "object": "file",
        "deleted": true
        }
       
        """
        try:
            log(logging, f"Deleting file {file_id}.", "purple")
            return self.openai.files.delete(file_id)
        except Exception as e:
            log(logging, f"Failed to delete file {file_id}: {e}", "red")
            return None

    def retrieve_file(self, file_id):
        """
        Retrieves a specific file.

        Args:
            file_id (str): The ID of the file to retrieve.

        Returns:
            The specified file object.
        """
        try:
            log(logging, f"Retrieving file {file_id}.", "purple")
            return self.openai.files.retrieve(file_id)
        except Exception as e:
            log(logging, f"Failed to retrieve file {file_id}: {e}", "red")
            return None

    def retrieve_file_content(self, file_id):
        """
        Retrieves the content of a specific file.

        Args:
            file_id (str): The ID of the file to retrieve.

        Returns:
            The content of the specified file.
        """
        try:
            log(logging, f"Retrieving content of file {file_id}.", "purple")
            return self.openai.files.retrieve(file_id, "content")
        except Exception as e:
            log(logging, f"Failed to retrieve content of file {file_id}: {e}", "red")
            return None  
    
class FineTuner:
    def __init__(self):
        self.openai = openai
        self.files = Files()
        log(logging, f"Initalized FineTuner class.", "green")

    @staticmethod
    def validate_finetuning_format(file_path):
        """
        Validates the format of the dataset for fine-tuning.

        Args:
            file_path (str): The path to the file to be validated for fine-tuning.
            perform_validation (bool): Flag to perform validation. Defaults to True.

        Returns:
            bool: True if the format is correct, False otherwise.
            dict: A dictionary containing counts of each type of error found.
        """

        format_errors = defaultdict(int)

        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name", "function_call") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant", "function"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if not any(message.get("role", None) == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

        is_valid = len(format_errors) == 0
        return is_valid, format_errors

    def upload_finetune_file(self, file_path, file_validation=True):
        """
        Uploads a file for fine-tuning, performing validation if specified.

        Args:
            file_path (str): The path to the file to be uploaded.
            validation (bool): Flag to perform validation. Defaults to True.

        Returns:
            The uploaded file.
        """
        if file_validation:
            is_valid, format_errors = self.validate_finetuning_format(file_path)
            if not is_valid:
                raise ValueError("File format is invalid. Please fix the following errors:\n" + "\n".join([f"{k}: {v}" for k, v in format_errors.items()]))
            
        return self.files.upload_file(file_path, purpose='fine-tune')

    def finetune_model(self, file_path, model_name, suffix, n_epochs, learning_rate_multiplier=None, batch_size=None, file_validation=True):
        """
        Fine-tunes a model with the provided parameters.

        Args:
            file_path (str): The path to the file to be used for fine-tuning.
            model_name (str): The name of the model to be fine-tuned.
            suffix (str): The suffix to be added to the fine-tuned model's name.
            n_epochs (int): The number of epochs for which the model should be fine-tuned.
            price_check (bool, optional): Whether to check the estimated cost of fine-tuning before proceeding. Defaults to True.
            learning_rate_multiplier (float, optional): The multiplier for the learning rate. If not provided, OpenAI will use defaults/auto.
            batch_size (int, optional): The batch size for fine-tuning. If not provided, OpenAI will use defaults/auto.

        Returns:
            The fine-tuning job.
        """

        file = self.upload_finetune_file(file_path, file_validation)

        #estimated_cost = estimate_finetune_cost(file_path, n_epochs, model_name)['cost']

        # if price_check:
        #     # Estimate the cost of fine-tuning
            
        #     print(f"Estimated cost of fine-tuning: ${estimated_cost}")

        #     # Check if the user wants to continue
        #     if input("Continue? (y/n): ") != "y":
        #         print("Exiting...")
        #         return

        # Construct hyperparameters dictionary based on provided inputs, if not provided OpenAI will use defaults/auto.
        hyperparameters = {}
        if n_epochs is not None:
            hyperparameters["n_epochs"] = n_epochs
        if learning_rate_multiplier is not None:
            hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier
        if batch_size is not None:
            hyperparameters["batch_size"] = batch_size

        log(logging, "Making Fine-Tuning API call...", "purple")
        fine_tuning_job = self.openai.fine_tuning.jobs.create(
            training_file=file.id, 
            model=model_name, 
            suffix=suffix, 
            hyperparameters=hyperparameters
        )

        return fine_tuning_job
    
    def list_finetuning_jobs(self):
        """
        Lists all fine-tuning jobs.
        
        Returns:
            A list of fine-tuning jobs.
        """
        return self.openai.fine_tuning.jobs.list()

    def retrieve_finetuning_job(self, job_id):
        """
        Retrieves a specific fine-tuning job.
        
        Args:
            job_id (str): The ID of the fine-tuning job to retrieve.
        
        Returns:
            The specified fine-tuning job.
        """
        return self.openai.fine_tuning.jobs.retrieve(job_id)

    def cancel_finetuning(self, job_id):
        """
        Cancels a specific fine-tuning job.
        
        Args:
            job_id (str): The ID of the fine-tuning job to cancel.
        
        Returns:
            The cancelled fine-tuning job.
        """
        return self.openai.fine_tuning.jobs.cancel(job_id)

    def list_finetuning_events(self, job_id, limit=2):
        """
        Lists events for a specific fine-tuning job.
        
        Args:
            job_id (str): The ID of the fine-tuning job to list events for.
            limit (int, optional): The maximum number of events to return. Defaults to 2.
        
        Returns:
            A list of events for the specified fine-tuning job.
        """
        return self.openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
    
    def delete_finetuned_model(self, model_id):
        """
        Deletes a specific fine-tuned model.
        
        Args:
            model_id (str): The ID of the fine-tuned model to delete.
        
        Returns:
            The deletion status of the fine-tuned model.
        """
        return self.openai.models.delete(model_id)
    
    def list_finetuned_models(self):
        """
        Lists all fine-tuned models that are not owned by OpenAI or system.
        
        Returns:
            A list of fine-tuned models.
        """
        all_models = self.openai.models.list()
        finetuned_models = [model for model in all_models.data if 'openai' not in model.owned_by.lower() and 'system' not in model.owned_by.lower()]
        return finetuned_models