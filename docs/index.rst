Welcome to Prompter's documentation!
==================================

Prompter is a Python library for structured LLM prompting with support for multiple providers.
It provides a clean, type-safe interface for working with LLMs while handling the complexities
of different provider APIs.

Key Features
-----------
* Provider-agnostic interface supporting OpenAI, Anthropic, and more
* Structured tool/function calling with argument validation
* Image handling for multimodal models
* Conversation history management
* Token usage tracking and cost calculation

Quick Start
----------

Installation
~~~~~~~~~~~

.. code-block:: bash

   pip install prompter[all]  # Install with all providers
   # Or install with specific providers:
   pip install prompter[openai]
   pip install prompter[anthropic]

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from prompter import OpenAIExecutor, Prompt, UserMessage

   executor = OpenAIExecutor()

   prompt = Prompt(
       system_message="You are a helpful assistant.",
       messages=[
           UserMessage(content="What is the capital of France?")
       ]
   )

   response = executor.execute(prompt)
   print(response.text())

Tool Usage
~~~~~~~~~

.. code-block:: python

   from pydantic import BaseModel
   from prompter import Tool, ToolCallMessage, ToolCallResult

   class WeatherArgs(BaseModel):
       location: str
       units: str = "celsius"

   weather_tool = Tool(
       name="get_weather",
       description="Get the current weather for a location",
       argument_schema=WeatherArgs
   )

   prompt = Prompt(
       system_message="You are a weather assistant.",
       messages=[
           UserMessage(content="What's the weather in Paris?")
       ],
       tools=[weather_tool]
   )

   response = executor.execute(prompt)
   tool_call = response.tool_call()

   # Tool arguments are validated against the schema
   args = tool_call.arguments  # Returns WeatherArgs instance
   print(f"Checking weather in {args.location}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`