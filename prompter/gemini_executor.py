import google.generativeai as genai
from schemas import Prompt, OutputMessage
from copy import deepcopy
from typing_extensions import TypeAlias
from typing import Any


class GeminiExecutor:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def execute(self, prompt: Prompt) -> OutputMessage:
        messages = [message.text() for message in prompt.messages]
        response = self.model.generate_content(prompt.system_message)
        return response.text


class GeminiExecution:
    def __init__(self, executor: GeminiExecutor, prompt: Prompt):
        self.executor = executor
        self.prompt = prompt

    def execute(self) -> OutputMessage:
        return self.executor.execute(self.prompt)
