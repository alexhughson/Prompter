"""
Core schema definitions for the Prompter library.

This module contains all the data structures used to represent prompts,
messages, and responses in a provider-agnostic way.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel

"""
Inputs
"""


### Messages
class Message(BaseModel):
    """Base class for all message types in a conversation.

    All message types must implement message_type() to identify their type.
    This allows executors to properly handle different kinds of messages
    (text, images, tool calls, etc).
    """

    @abstractmethod
    def message_type(self) -> str:
        """Get the type identifier for this message.

        Returns:
            str: The message type identifier (e.g., "text", "image", "tool_call")
        """
        pass


class UserMessage(Message):
    """A text message from the user to the LLM.

    Attributes:
        content: The text content of the message
        role: Always "user" for user messages
    """

    content: str
    role: str = "user"

    def message_type(self) -> str:
        return "text"


class AssistantMessage(Message):
    """A text response from the LLM.

    Attributes:
        content: The text content of the message
        role: Always "assistant" for LLM responses
    """

    content: str
    role: str = "assistant"

    def message_type(self) -> str:
        return "text"


class ImageMessage(Message):
    """An image with optional caption to be shown to the LLM.

    Attributes:
        url: URL of the image to be processed
        content: Optional caption or description of the image
        role: Always "user" as only users can send images
        media_type: MIME type of the image, defaults to JPEG
    """

    role: str = "user"
    content: Optional[str] = None
    url: str
    media_type: Optional[str] = "image/jpeg"

    def message_type(self) -> str:
        return "image"


@dataclass
class ToolCallResult:
    """The result of executing a tool call.

    Attributes:
        result: The data returned by the tool, can be any JSON-serializable type
        error: Optional error message if the tool call failed
    """

    result: Any
    error: Optional[str] = None


class ToolCallMessage(Message):
    """A record of a tool being called and its result.

    This represents both the request to call a tool and its response in the
    conversation history. It's used to show the LLM what happened when a tool
    was called previously.

    Attributes:
        tool_name: Name of the tool that was called
        tool_call_id: Optional unique identifier for this specific call
        arguments: The arguments that were passed to the tool
        result: The result returned by the tool
        role: Always "assistant" since tools are called by the LLM
    """

    tool_name: str
    tool_call_id: Optional[str] = None
    arguments: Optional[Union[Dict, str]]
    result: ToolCallResult
    role: str = "assistant"

    def message_type(self) -> str:
        return "tool_call"


### Prompt Options
class Tool(BaseModel):
    """Definition of a tool that the LLM can use.

    Tools represent external functions or APIs that the LLM can call to get
    information or perform actions.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        argument_schema: Pydantic model or JSON schema defining the tool's parameters

    Example:
        ```python
        class WeatherArgs(BaseModel):
            location: str
            units: str = "celsius"

        weather_tool = Tool(
            name="get_weather",
            description="Get the current weather for a location",
            argument_schema=WeatherArgs
        )
        ```
    """

    name: str
    description: str
    argument_schema: Union[Type[BaseModel], Dict]

    def get_schema(self) -> Dict:
        """Get the JSON schema for the tool's arguments.

        Returns:
            Dict: JSON Schema object describing the tool's parameters
        """
        if isinstance(self.argument_schema, type) and issubclass(
            self.argument_schema, BaseModel
        ):
            return self.argument_schema.model_json_schema()
        return self.argument_schema


class Prompt:
    """A complete prompt to be sent to an LLM.

    This includes the system message that sets up the LLM's role,
    the conversation history, and any tools the LLM can use.

    Attributes:
        system_message: Instructions that define the LLM's role and behavior
        messages: List of messages in the conversation history
        tools: Optional list of tools the LLM can use

    Example:
        ```python
        prompt = Prompt(
            system_message="You are a helpful weather assistant.",
            messages=[
                UserMessage(content="What's the weather in London?"),
                ToolCallMessage(
                    tool_name="get_weather",
                    arguments={"location": "London"},
                    result=ToolCallResult(result={"temp": 20, "conditions": "sunny"})
                )
            ],
            tools=[weather_tool]
        )
        ```
    """

    system_message: str
    messages: List[Message]
    tools: List[Tool] = field(default_factory=list)

    def __init__(
        self,
        system_message: str,
        messages: List[Message],
        tools: List[Tool] = None,
        **kwargs,
    ):
        self.system_message = system_message
        self.messages = messages
        self.tools = tools or []


"""
Outputs
"""


class OutputMessage:
    """
    Base class for all types of LLM output messages.

    All message types must implement:
    - text(): Returns a string representation of the message
    - is_tool_call(): Returns whether this message represents a tool call
    """

    type: str

    @abstractmethod
    def text(self) -> str:
        pass

    @abstractmethod
    def is_tool_call(self) -> bool:
        pass

    @abstractmethod
    def is_text(self) -> bool:
        return not self.is_tool_call()

    @abstractmethod
    def raise_on_parse_error(self) -> None:
        pass


class TextOutputMessage(OutputMessage):
    content: str
    type: str = "text"

    def __init__(self, content: str):
        self.content = content

    def text(self) -> str:
        return self.content

    def is_tool_call(self) -> bool:
        return False

    def raise_on_parse_error(self) -> None:
        pass

    def is_text(self) -> bool:
        return True


# TODO
# class ImageOutputMessage(OutputMessage):
#     image_url: str
#     type: str = "image"

#     def text(self) -> str:
#         return f"Image: {self.image_url}"


class ToolCallOutputMessage(OutputMessage):
    """
    Represents a tool call requested by the LLM.

    Attributes:
        name (str): Name of the tool to be called
        type (str): Always "tool_call"

    Methods:
        set_schema(schema): Set the Pydantic model for argument validation
        is_valid(): Check if arguments match the provided schema
        parsed_arguments(): Get validated arguments as a Pydantic model
        raw_arguments(): Get unvalidated arguments as a dict
    """

    name: str
    _arguments: Any
    _schema: Optional[Type[BaseModel]] = None

    def __init__(self, name: str, arguments: str, schema=None):
        self.name = name
        self._arguments = arguments
        self._schema = schema

    def to_input_message(self, result: str) -> ToolCallMessage:
        return ToolCallMessage(
            name=self.name, arguments=self._arguments, tool_call_id=uuid.uuid()
        )

    @property
    def arguments(self) -> Union[BaseModel, Dict, str]:
        return self.parse()

    def parse(self) -> Union[BaseModel, Dict, str]:
        """Parse the arguments into the appropriate type"""
        # First ensure we have a dict
        args_dict = (
            self._arguments
            if isinstance(self._arguments, dict)
            else json.loads(self._arguments)
        )

        # If we have a schema, parse into the model
        if self._schema is not None:
            return self._schema.model_validate(args_dict)
        return args_dict

    def text(self) -> str:
        parsed_args = self.parse()
        if isinstance(parsed_args, BaseModel):
            args_str = parsed_args.model_dump_json()
        else:
            args_str = json.dumps(parsed_args)
        return f"Tool {self.name} called with arguments {args_str}"

    def is_tool_call(self) -> bool:
        return True

    def raise_on_parse_error(self) -> None:
        pass

    def raw_arguments(self):
        return self._arguments

    def parsing_failed(self) -> bool:
        return False

    def is_text(self) -> bool:
        return False


class ModelPricing(BaseModel):
    """Pricing information for a specific model"""

    input_price: float  # Cost per 1K input tokens
    output_price: float  # Cost per 1K output tokens


class UsageInfo(BaseModel):
    """Token usage information from an API call"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMResponse:
    """
    Complete response from an LLM, including all outputs and usage information.

    Attributes:
        messages (List[OutputMessage]): All messages in the response
        usage (UsageInfo): Token usage information
        cost (float): Cost of the API call

    Methods:
        parsed_result() -> T:
            Parse the entire response according to the provided schema.
            Raises ValidationError if parsing fails.

        text(include_tool_calls: bool = True) -> str:
            Get all text content concatenated.
            Optional parameter to include/exclude tool call descriptions.

        tool_calls(strict: bool = True) -> List[ToolCallOutputMessage]:
            Get all tool calls.
            If strict=True, only returns calls with valid arguments.

        single_tool_call() -> ToolCallOutputMessage:
            Get exactly one tool call.
            Raises ValueError if there isn't exactly one.

        messages_iter() -> Iterator[OutputMessage]:
            Iterate through all messages in order.

        tool_calls_iter(strict: bool = True) -> Iterator[ToolCallOutputMessage]:
            Iterate through tool calls in order.
            If strict=True, only yields calls with valid arguments.

    Properties:
        result_type: The type of response (TEXT_ONLY, TOOL_CALLS, or MIXED)
        has_tool_calls: Whether the response contains any tool calls

    Example:
        ```python
        response = executor.execute[WeatherResponse](prompt)

        # Get structured data
        if response.result_type == ResultType.TEXT_ONLY:
            weather = response.parsed_result()
            print(f"Temperature: {weather.temperature}")

        # Process tool calls
        for tool_call in response.tool_calls_iter():
            try:
                args = tool_call.parsed_arguments()
                result = weather_service.get_weather(**args.dict())
            except ValidationError:
                print(f"Invalid args for {tool_call.name}")
        ```
    """

    messages: List[OutputMessage]
    usage: UsageInfo
    cost: float = 0.0

    def __init__(
        self, messages: List[OutputMessage], usage: UsageInfo, cost: float = 0.0
    ):
        self._messages = messages
        self.usage = usage
        self.cost = cost

    @property
    def messages(self) -> List[OutputMessage]:
        return self._messages

    def tool_call(self) -> Optional[ToolCallOutputMessage]:
        all_tool_calls = self.tool_calls()
        if len(all_tool_calls) == 1:
            return all_tool_calls[0]
        elif len(all_tool_calls) == 0:
            raise ValueError("No tool calls found in response")
        else:
            raise ValueError("Multiple tool calls found in response")

    def tool_calls(self) -> List[ToolCallOutputMessage]:
        return [
            message
            for message in self.messages
            if isinstance(message, ToolCallOutputMessage)
        ]

    def text_messages(self) -> List[TextOutputMessage]:
        return [
            message
            for message in self.messages
            if isinstance(message, TextOutputMessage)
        ]

    def text(self) -> str:
        return "\n".join([message.text() for message in self.messages])

    def raise_for_status(self):
        pass


# @dataclass
# class PendingToolCall(Message):
#     """A tool call that has been requested but not yet completed"""

#     tool_name: str
#     arguments: Dict
#     role: str = "assistant"

#
#     def message_type(self) -> str:
#         return "pending_tool_call"
