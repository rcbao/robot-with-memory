# prompt_builder.py

import re
from file_handler import FileHandler
from constants import prompt_paths


class PromptBuilder:
    def __init__(self):
        self.file_handler = FileHandler()

        self.system_prompt = self.file_handler.read_file(prompt_paths["system"])
        self.user_prompt_dict = prompt_paths["user"]

    def build_input_parser_prompt(self, user_command: str) -> str:
        user_prompt_path = self.user_prompt_dict["parse_input_json"]
        user_prompt = self.file_handler.read_file(user_prompt_path)
        user_prompt = user_prompt.format(user_command=user_command)

        return user_prompt

    def build_recall_prompt(self, user_command: str, object_details: str) -> str:
        recall_prompt_path = self.user_prompt_dict["recall"]
        recall_prompt = self.file_handler.read_file(recall_prompt_path)
        recall_prompt = recall_prompt.format(user_command=user_command, object_details=object_details)

        return recall_prompt

    def build_generic_prompt(self, user_command: str) -> str:
        prompt = f"""Please respond to the following user input: ''' {user_command} '''. Answer in a friendly manner."""
        return prompt

    def build_initial_messages(self, user_prompt: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def build_messages(self, user_prompt: str, message_history: list):
        # Start with initial messages if history is empty
        if not message_history:
            return self.build_initial_messages(user_prompt)
        else:
            # Append new user prompt to the existing history
            message_history.append({"role": "user", "content": user_prompt})
            return message_history

    def build_input_parser_messages(self, user_command: str, message_history: list = []):
        user_prompt = self.build_input_parser_prompt(user_command)
        messages = self.build_messages(user_prompt, message_history)
        return messages
    
    def build_recall_messages(self, user_command: str, object_details: str, message_history: list = []):
        user_prompt = self.build_recall_prompt(user_command=user_command, object_details=object_details)
        messages = self.build_messages(user_prompt, message_history)
        return messages

    def build_generic_messages(self, user_command: str, message_history: list = []):
        user_prompt = self.build_generic_prompt(user_command)
        messages = self.build_messages(user_prompt, message_history)
        return messages
