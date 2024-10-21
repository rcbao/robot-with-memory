# /src/language_processor.py

import os
import json
from openai import OpenAI
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def clean_code_block(s: str) -> str:
    return s.removeprefix("```json").removesuffix("```").strip()

class LanguageProcessor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LanguageProcessor with OpenAI API key.

        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, it will be read from the environment variable 'OPENAI_API_KEY'.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set the 'OPENAI_API_KEY' environment variable."
            )

    def interpret_user_prompt(
        self, user_prompt: str
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Interpret the user prompt to extract action, object name, and details using GPT-4.

        Args:
            user_prompt (str): The input command from the user.

        Returns:
            Tuple containing:
                - relevancy (bool): Whether the prompt is relevant (remember, recall, fetch).
                - action (Optional[str]): The action to perform ('remember', 'recall', 'fetch').
                - object_name (Optional[str]): The name of the object.
                - detail (Optional[str]): Additional details about the object.
        """
        prompt = (
            f"Parse the following user command and extract the action, object name, and detail.\n\n"
            f'Command: "{user_prompt}"\n\n'
            f"Format the response as JSON with keys: relevancy (true/false), action, object_name, detail.\n"
            f"If the command is not relevant to remembering, recalling, or fetching objects, set relevancy to false."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that parses user commands related to object memory and fetching.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0,
            )

            response_content = response.choices[0].message.content.strip()
            print(f"LLM Response:\n----------\n{response_content}\n----------\n")
            response_content = clean_code_block(response_content)
            parsed = json.loads(response_content)

            relevancy = parsed.get("relevancy", False)
            action = parsed.get("action")
            object_name = parsed.get("object_name")
            detail = parsed.get("detail")

            return relevancy, action, object_name, detail

        except Exception as e:
            print(f"Error interpreting user prompt: {e}")
            return False, None, None, None

    def generate_response(self, context: str) -> str:
        """
        Generate a natural language response based on the given context using GPT-4.

        Args:
            context (str): The context or message to respond to.

        Returns:
            str: The generated response.
        """
        prompt = (
            f"Generate a friendly and helpful response to the following message:\n\n"
            f'"{context}"'
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly and helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.7,
            )

            

            response_content = response.choices[0].message.content.strip()
            print(f"LLM Response:\n----------\n{response_content}\n----------\n")
            return response_content

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process that request."

    def are_the_same_object(self, object_name: str, object_detail: str, other_object_name: str, other_object_detail: str) -> bool:
        """
        Decide whether the two descriptions of an object are for the same item.

        Arguments:
            object_name (str): object name.
            object_detail (str): object descriptors. 
            other_object_name (str): the name of the object you want to compare to.
            other_object_name (str): the description of the object you want to compare to.

        Returns:
            bool: whether the objects are the same (true) or not (false).
        """
        prompt = (
            f"Determine if the two objects are the same object or not, based off of the object name and details:\n\n"
            f'Object 1 name: {object_name}'
            f'Object 1 details: {object_detail}'
            f'Object 2 name: {other_object_name}'
            f'Object 2 details: {other_object_detail}'
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Reply strictly with a boolean value (either 'True' or 'False'). Use common sense.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.7,
            )

            response_content = response.choices[0].message.content.strip()
            bool_value = eval(response_content)
            return bool_value

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process that request."
