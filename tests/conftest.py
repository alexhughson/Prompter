import pytest
from prompter.openai_executor import OpenAIExecutor
from prompter.anthropic_executor import AnthropicExecutor
from prompter.gemini_executor import GeminiExecutor


@pytest.fixture(params=["openai", "anthropic"])
def llm_executor(request):
    """Provides an LLM executor for each supported provider"""
    if request.param == "openai":
        return OpenAIExecutor()
    elif request.param == "anthropic":
        return AnthropicExecutor()
