import os

import pytest
from openai import OpenAI

from prompter.anthropic_executor import ClaudeExecutor
from prompter.openai_executor import OpenAIExecutor


# from prompter.gemini_executor import GeminiExecutor

# Map provider-model combinations to their model IDs
MODEL_IDS = {
    "gemini-openrouter": "google/gemini-pro-1.5",
    "llama-fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "llama-openrouter": "meta-llama/llama-3.3-70b-instruct",
    "nova-openrouter": "amazon/nova-pro-v1",
    "phi-fireworks": "accounts/fireworks/models/phi-3-vision-128k-instruct",
    "phi-openrouter": "microsoft/phi-3.5-mini-128k-instruct",
    "qwen-fireworks": "accounts/fireworks/models/qwen2p5-72b-instruct",
    "qwen-openrouter": "qwen/qwen-2.5-72b-instruct",
    "openai-openrouter": "openai/gpt-4o-2024-11-20",
    "anthropic-openrouter": "anthropic/claude-3.5-sonnet",
}


@pytest.fixture(
    params=[
        # "anthropic-openai-compat",
        "anthropic",
        # "anthropic-openrouter",
        # "gemini-openai-compat",
        # "gemini-openrouter",
        # "llama-fireworks",
        # "llama-openrouter",
        # "nova-openrouter",
        "openai",
        # "openai-responses",
        # "openai-openrouter",
        # "phi-fireworks",
        # "phi-openrouter",
        # "qwen-fireworks",
        # "qwen-openrouter",
    ]
)
def llm_executor(request):
    """Provides an LLM executor for each supported provider"""
    if request.param == "openai":
        return OpenAIExecutor()
    elif request.param == "anthropic":
        return ClaudeExecutor()
    # elif request.param == "gemini-openai-compat":
    #     client = OpenAI(
    #         base_url="https://generativelanguage.googleapis.com/v1beta/",
    #         api_key=os.getenv("GEMINI_API_KEY"),
    #     )
    #     return OpenAIExecutor(client=client, model="gemini-1.5-flash")
    # elif request.param.endswith("-openrouter"):
    #     client = OpenAI(
    #         base_url="https://openrouter.ai/api/v1",
    #         api_key=os.getenv("OPENROUTER_API_KEY"),
    #     )
    #     return OpenAIExecutor(client=client, model=MODEL_IDS[request.param])
    # elif request.param.endswith("-fireworks"):
    #     from fireworks.client import Fireworks

    #     client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))
    #     # client = OpenAI(
    #     #     base_url="https://api.fireworks.ai/inference/v1",
    #     #     api_key=os.getenv("FIREWORKS_API_KEY"),
    #     # )
    #     return OpenAIExecutor(client=client, model=MODEL_IDS[request.param])
