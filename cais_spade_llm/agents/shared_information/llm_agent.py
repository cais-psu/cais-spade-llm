from __future__ import annotations

import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No API key for OpenAI found in the environment variables.")
openai.api_key = OPENAI_API_KEY

class LlmAgent:
    def __init__(
        self,
        model: str = None,
        name: str = None,
        annotation: str = None,
        instructions: str = None,
        functions_: list = None,
        non_function_model: str = "gpt-4o",
    ) -> None:

        self.model = model or "gpt-4o"
        self.name = name
        self.annotation = annotation
        self.non_function_model = non_function_model

        self.function_info = []
        self.executables = {}
