from collections import namedtuple

PromptPath = namedtuple("PromptPath", ["system", "user"])

prompt_paths = {
    "system": "./llm_prompts/general_system.txt",
    "user": {
        "parse_input_json": "./llm_prompts/parse_input_json_user.txt",
        "recall": "./llm_prompts/recall_object_user.txt"
    }
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 400