import re
from file_handler import FileHandler
from constants import prompt_paths


class PromptBuilder:
    def __init__(self):
        self.file_handler = FileHandler()

    def build_input_parser_user_prompt(self, user_command: str) -> str:
        user_prompt = prompt_paths["parse_input_json"].user
        user_prompt = self.file_handler.read_file(user_prompt)

        return user_prompt.format(user_command=user_command)

    def build_input_parser_system_prompt(self) -> str:
        system_prompt = prompt_paths["parse_input_json"].system
        return self.file_handler.read_file(system_prompt)


    def build_messages(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def build_input_parser_messages(self, user_command):
        system_prompt = self.build_input_parser_system_prompt()
        user_prompt = self.build_input_parser_user_prompt(user_command)

        return self.build_messages(system_prompt, user_prompt)