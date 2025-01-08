"""
Core schema definitions for the Prompter library.

This module contains all the data structures used to represent prompts,
messages, and responses in a provider-agnostic way.
"""

import uuid
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type, Union
from enum import Enum

from pydantic import BaseModel, ValidationError

"""
Inputs
"""


### Messages
class Message:
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

    def __init__(self, content):
        self.content = content
        self.role = "user"
        self.type = "text"

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

    def __init__(self, content):
        self.content = content

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

    def __init__(self, url, content=None, media_type="image/jpeg"):
        self.url = url
        self.content = content
        self.media_type = media_type
        self.role = "user"

    role: str
    content: Optional[str]
    url: str
    media_type: Optional[str]

    def message_type(self) -> str:
        return "image"


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
    result: Any
    role: str = "assistant"

    def __init__(self, tool_name, arguments, result, tool_call_id=None):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.tool_call_id = tool_call_id
        self.role = "assistant"

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
        response_schema: Type[BaseModel] = None,
        **kwargs,
    ):
        self.system_message = system_message
        self.messages = messages
        self.tools = tools or []
        self.response_schema = response_schema


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
        tool_call_id (str): Unique identifier for this tool call
        arguments (SchemaResult): Arguments for the tool call with schema validation
    """

    type = "tool_call"

    def __init__(
        self,
        name: str,
        arguments: Union[str, Dict],
        tool_call_id: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.tool_call_id = tool_call_id or str(uuid.uuid4())
        # Convert arguments to string if they're a dict
        args_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
        self._arguments = SchemaResult(args_str, schema)

    def to_input_message(self, result: Optional[Any] = None) -> ToolCallMessage:
        """Convert this output message to an input message with optional result

        Args:
            result: Optional result from executing the tool call

        Returns:
            ToolCallMessage: Message that can be added to conversation history
        """
        return ToolCallMessage(
            tool_name=self.name,
            arguments=(
                self._arguments.parse_obj() if self._arguments.valid_json() else {}
            ),
            tool_call_id=self.tool_call_id,
            result=result,
        )

    @property
    def arguments(self) -> "SchemaResult":
        """Get the SchemaResult for validating and parsing arguments"""
        return self._arguments

    def text(self) -> str:
        """Get a text representation of this tool call"""
        if self._arguments.valid_json():
            args_str = json.dumps(self._arguments.parse_obj())
        else:
            args_str = self._arguments.raw()
        return f"Tool {self.name} called with arguments {args_str}"

    def is_tool_call(self) -> bool:
        return True

    def is_text(self) -> bool:
        return False


class ModelPricing(BaseModel):
    """Pricing information for a specific model"""

    input_price: float  # Cost per 1K input tokens
    output_price: float  # Cost per 1K output tokens


class UsageInfo(BaseModel):
    """Token usage information from an API call"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseFormat(Enum):
    """Type of response format requested from the LLM"""

    TEXT = "text"
    JSON = "json_object"
    STRUCTURED = "structured"


class CompletionInfo:
    """Information about how the completion finished"""

    class FinishType(Enum):
        """Type of completion finish"""

        SUCCESS = "stop"  # Normal completion
        TOOL_USE = "tool_calls"  # Stopped to make tool calls
        FAIL_LENGTH = "length"  # Hit token limit
        FAIL_FILTER = "content_filter"  # Content was filtered
        FAIL_ERROR = "error"  # Other error occurred

    FAILURE_REASONS = [
        FinishType.FAIL_LENGTH,
        FinishType.FAIL_FILTER,
        FinishType.FAIL_ERROR,
    ]

    def __init__(self, finish_reason: FinishType, refusal: Optional[str] = None):
        self.finish_reason = finish_reason
        self.refusal = refusal


class LLMResponseError(Exception):
    pass


class LLMResponse:
    """Complete response from an LLM, including all outputs and usage information."""

    def __init__(
        self,
        messages: Optional[List[OutputMessage]],
        usage: UsageInfo,
        cost: Optional[float] = 0.0,
        completion_info: Optional[CompletionInfo] = None,
        response_object=None,
    ):
        self._messages = messages
        self.usage = usage
        self.cost = cost
        self.completion_info = completion_info or CompletionInfo()
        self.response_object = response_object
        self._schema_result = None

    def result(self) -> Optional["SchemaResult"]:
        """Get schema validation result if a schema was specified"""
        if self.response_object is None:
            raise Exception("uh, no")
        return self.response_object

    def text(self, include_tools=False):
        return "\n".join(
            [
                message.content
                for message in self._messages
                if message.type == "text" or include_tools
            ]
        )

    def tool_call(self) -> Optional[ToolCallOutputMessage]:
        if len(self.tool_calls()) == 0:
            return None
        if len(self.tool_calls()) > 1:
            raise ValueError(
                "Multiple tool calls in response, use tool_calls() instead to iterate over them"
            )
        return self.tool_calls()[0]

    def messages(self) -> List[OutputMessage]:
        """Get all messages in the response"""
        return self._messages

    def text_messages(self) -> List[TextOutputMessage]:
        """Get all text messages in the response"""
        return [msg for msg in self._messages if msg.type == "text"]

    def tool_calls(self) -> List[ToolCallOutputMessage]:
        """Get all tool messages in the response"""
        return [msg for msg in self._messages if msg.type == "tool_call"]

    def raise_for_status(self):
        if not self.completion_info:
            return
        """Raise an exception if the response status indicates an error"""
        if self.completion_info.finish_reason in CompletionInfo.FAILURE_REASONS:
            raise LLMResponseError(
                f"Response status: {self.completion_info.finish_reason}"
            )


class SchemaResult:
    """Handles validation and parsing of LLM responses against a schema.

    This class provides methods to:
    1. Check if the response is valid JSON
    2. Check if it matches the expected schema
    3. Parse the response into either a dict or schema instance

    Example:
        ```python
        result = response.result()
        if result.valid():
            character = result.parse()  # Returns schema instance
        elif result.valid_json():
            data = result.parse_obj()  # Returns dict, even if schema invalid
        ```
    """

    def __init__(self, input_data: str, schema: Optional[Type[BaseModel]] = None):
        """
        Args:
            input_data: Raw text response from LLM
            schema: Optional Pydantic model to validate against
        """
        self.input_data = input_data
        self._schema = schema
        self._parsed_json = None
        self._json_valid = None

    @classmethod
    def from_parsed(cls, parsed_obj: BaseModel, schema: Type[BaseModel]):
        json_dump = parsed_obj.model_dump_json
        return cls(json_dump, schema)

    def valid(self) -> bool:
        """Check if response is valid according to schema.

        If no schema was specified, checks if response is valid JSON.

        Returns:
            bool: True if valid, False otherwise
        """
        if not self.valid_json():
            return False

        if self._schema is None:
            return True

        try:
            self._schema.model_validate(self._parsed_json)
            return True
        except ValidationError:
            return False

    def valid_json(self) -> bool:
        """Check if response is valid JSON.

        Returns:
            bool: True if valid JSON, False otherwise
        """
        if self._json_valid is not None:
            return self._json_valid

        try:
            self._parsed_json = json.loads(self.input_data)
            self._json_valid = True
            return True
        except json.JSONDecodeError:
            self._json_valid = False
            return False

    def parse_obj(self) -> Union[Dict, List]:
        """Parse response as JSON into dict/list.

        Returns:
            Union[Dict, List]: Parsed JSON data

        Raises:
            JSONDecodeError: If response is not valid JSON
        """
        if not self.valid_json():
            raise json.JSONDecodeError("Invalid JSON", self.input_data, 0)
        return self._parsed_json

    def parse(self) -> BaseModel:
        """Parse response into schema instance.

        Returns:
            BaseModel: Instance of schema class

        Raises:
            JSONDecodeError: If response is not valid JSON
            SchemaValidationError: If response doesn't match schema
            ValueError: If no schema was specified
        """
        if self._schema is None:
            raise ValueError("No schema specified")

        if not self.valid_json():
            raise json.JSONDecodeError("Invalid JSON", self.input_data, 0)

        try:
            return self._schema.model_validate(self._parsed_json)
        except ValidationError as e:
            raise SchemaValidationError(str(e))

    def raw(self) -> str:
        """Get raw response text.

        Returns:
            str: Original response text
        """
        return self.input_data


class SchemaValidationError(Exception):
    pass
