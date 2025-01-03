from abc import ABC, abstractmethod
from typing import Optional, Dict

from .schemas import Prompt, LLMResponse


class BaseLLMExecutor(ABC):
    """Base class for all LLM executors"""

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def execute(self, prompt: Prompt) -> LLMResponse:
        """Execute the prompt and return a response"""
        pass

    @abstractmethod
    def _calculate_cost(self, usage: Dict) -> float:
        """Calculate the cost of the API call based on usage statistics"""
        pass

    @abstractmethod
    def _convert_prompt_to_api_messages(self, prompt: Prompt) -> list:
        """Convert our internal prompt format to the API's expected format"""
        pass

    @abstractmethod
    def _convert_api_response_to_messages(self, response: Dict) -> LLMResponse:
        """Convert the API response to our internal format"""
        pass
