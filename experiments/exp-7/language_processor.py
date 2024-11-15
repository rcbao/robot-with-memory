import os
import json
from openai import OpenAI
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()



class LanguageProcessor:
    def __init__(self):
        """
        Initialize the LanguageProcessor with OpenAI API key.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            error_message = "OpenAI API key not provided. Set the 'OPENAI_API_KEY' environment variable."
            raise ValueError(error_message)

    def clean_up_json(self, llm_response: str) -> str:
        """
        Remove code block from JSON.
        """
        llm_response = llm_response.removeprefix("```json")
        llm_response = llm_response.removesuffix("```")
        llm_response = llm_response.strip()
        return llm_response

    def get_llm_response(self, system_message: str, user_message: str, max_tokens: int, temperature: float) -> str:
        """
        Interact with the GPT-4o model and return a response.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_content = response.choices[0].message.content.strip()
            print(f"[LLM Response]: {response_content}")
            return response_content

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def parse_user_input(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Interpret the user prompt to extract action, object name, and details using LLM
        """
        system_message = "You are a helpful assistant that parses user commands related to object memory and fetching."
        user_message = (
            f"Parse the following user command and extract the action, object name, and detail.\n\n"
            f'Command: "{user_input}"\n\n'
            f"Format the response as JSON with keys: relevancy (true/false), action, object_name, detail.\n"
            f"If the command is not relevant to remembering, recalling, or fetching objects, set relevancy to false."
        )

        fallback_response = (False, None, None, None)

        response_content = self.get_llm_response(
            system_message=system_message,
            user_message=user_message,
            max_tokens=150,
            temperature=0
        )

        if response_content:
            response_content = self.clean_up_json(response_content)
            try:
                parsed = json.loads(response_content)

                relevancy = parsed.get("relevancy", False)
                action = parsed.get("action")
                object_name = parsed.get("object_name")
                detail = parsed.get("detail")

                return relevancy, action, object_name, detail
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                return fallback_response
        else:
            return fallback_response

    def generate_response(self, context: str) -> str:
        """
        Generate a natural language response based on the given context using GPT-4.
        """
        system_message = "You are a friendly and helpful assistant."
        user_message = (
            f"Generate a friendly and helpful response to the following message:\n\n"
            f'"{context}"'
        )

        return self.get_llm_response(
            system_message=system_message,
            user_message=user_message,
            max_tokens=100,
            temperature=0.7
        )
