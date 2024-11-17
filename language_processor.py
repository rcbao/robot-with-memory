# language_processor.py

import os
import json
from openai import OpenAI
from typing import Tuple, Optional
from dotenv import load_dotenv
from prompt_builder import PromptBuilder

load_dotenv()


class LanguageProcessor:
    def __init__(self):
        """
        Initialize the LanguageProcessor with OpenAI API key.
        """
        self.client = OpenAI()
        self.prompt_builder = PromptBuilder()
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

    def get_llm_response(self, messages: list, max_tokens: int, temperature: float) -> str:
        """
        Interact with the GPT-4o model and return a response.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_content = response.choices[0].message.content.strip()
            # print(f"[LLM Response]: {response_content}")
            return response_content

        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def parse_user_input(self, user_command: str, message_history: list) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Interpret the user prompt to extract action, object name, and details using LLM
        """
        messages = self.prompt_builder.build_input_parser_messages(user_command, message_history)

        fallback_response = (False, None, None, None)

        response_content = self.get_llm_response(
            messages=messages,
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

    def recall_object_info(self, user_question: str, obj_description: str, message_history: list = []) -> Optional[str]:
        """
        Given the user's question, use LLM to generate a response with object information.
        """
        # Build recall messages using PromptBuilder
        messages = self.prompt_builder.build_recall_messages(user_question, obj_description, message_history)

        response_content = self.get_llm_response(
            messages=messages,
            max_tokens=200,
            temperature=0
        )

        if response_content:
            # Optionally, clean up the response if needed
            response_content = response_content.strip()
            return response_content
        else:
            return None

    def write_generic_response(self, user_question: str, message_history: list = []) -> Optional[str]:
        """
        Given the user's question, use LLM to generate a response with object information.
        """
        # Build recall messages using PromptBuilder
        messages = self.prompt_builder.build_generic_messages(user_question, message_history)

        response_content = self.get_llm_response(
            messages=messages,
            max_tokens=200,
            temperature=0
        )

        if response_content:
            # Optionally, clean up the response if needed
            response_content = response_content.strip()
            return response_content
        else:
            return None

if __name__ == "__main__":
    lang = LanguageProcessor()
    output = lang.parse_user_input("what did we do so far?", [])
    print(output)
