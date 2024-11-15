from collections import namedtuple

PromptPath = namedtuple("PromptPath", ["system", "user"])

prompt_paths = {
    "parse_input_json": PromptPath(
        system="./llm_prompts/parse_input_json_sys.txt",
        user="./llm_prompts/parse_input_json_user.txt",
    ),
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 400