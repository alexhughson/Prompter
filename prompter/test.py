from .schemas import (
    Prompt,
    UserMessage,
    AssistantMessage,
    Tool,
    ImageMessage,
    ToolPool,
    OutputMessage,
)

# from gemini_executor import GeminiExecutor
from .openai_executor import OpenAIExecutor
from .anthropic_executor import AnthropicExecutor

# from .anthropic_executor import AnthropicExecutor

# from tools import GetWeatherTool, GetNewsTool
from pydantic import BaseModel


URL_OF_CLOUD_PICTURE = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Cumulus_humilis_clouds_in_Ukraine.jpg/1920px-Cumulus_humilis_clouds_in_Ukraine.jpg"


def test_image_url(executor):
    prompt = Prompt(
        system_message="You are a helpful assistant",
        messages=[
            UserMessage(content="What is this?"),
            ImageMessage(
                url=URL_OF_CLOUD_PICTURE,
            ),
        ],
    )

    response = executor.execute(prompt)
    response.raise_for_status()

    print("Cloud test response: " + response.text())
    assert "cloud" in response.text().lower()


class WeatherArgs(BaseModel):
    location: str


def test_use_tool(executor):
    prompt = Prompt(
        system_message="You are a helpful assistant",
        messages=[
            UserMessage(
                content="Please use the get_weather tool to get the weather in toronto"
            ),
        ],
        tools=[
            Tool(
                name="get_weather",
                description="Get the weather in a given location",
                argument_schema=WeatherArgs,
            )
        ],
        tool_use=Prompt.TOOL_USE_REQUIRED,
    )
    response = executor.execute(prompt)
    response.raise_for_status()
    tool_call = response.tool_call()

    print("Tool call: " + str(tool_call.arguments))

    assert tool_call.name == "get_weather"
    assert isinstance(tool_call.arguments, WeatherArgs)
    assert tool_call.arguments.location.lower() == "toronto"


executors = [
    # gemini,
    # OpenAIExecutor(),
    AnthropicExecutor(),
]

for executor in executors:
    # test_image_url(executor)
    test_use_tool(executor)
