"""
Core schema definitions for the Prompter library.

This module contains all the data structures used to represent prompts,
messages, and responses in a provider-agnostic way.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ValidationError

"""
Inputs
"""


class ToolUse(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NONE = "none"


class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_CALL = "tool_call"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"


@dataclass
class TextMessage:
    """A text message from the user"""

    content: str
    role: Role
    type: Literal[MessageType.TEXT] = MessageType.TEXT

    @classmethod
    def user(cls, content: str) -> "TextMessage":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "TextMessage":
        return cls(role=Role.ASSISTANT, content=content)


@dataclass
class ImageMessage:
    """An image message with optional text caption"""

    url: str
    role: Role
    type: Literal[MessageType.IMAGE] = MessageType.IMAGE
    media_type: str = "image/jpeg"

    @classmethod
    def user(cls, url) -> "ImageMessage":
        return cls(role=Role.USER, url=url)

    @classmethod
    def assistant(cls, url) -> "ImageMessage":

        return cls(role=Role.ASSISTANT, url=url)


@dataclass
class ToolCallMessage:
    """A tool call message with args and optional result"""

    tool_name: str
    arguments: dict
    type: Literal[MessageType.TOOL_CALL] = MessageType.TOOL_CALL
    result: dict | str | None = None
    tool_call_id: str | None = None
    content: str = ""
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT


Message = TextMessage | ImageMessage | ToolCallMessage


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
    argument_schema: Type[BaseModel]

    def get_schema(self) -> Dict:
        """Get the JSON schema for the tool's arguments.

        Returns:
            Dict: JSON Schema object describing the tool's parameters
        """
        # if isinstance(self.argument_schema, type) and issubclass(
        #     self.argument_schema, BaseModel
        # ):
        return self.argument_schema.model_json_schema()
        # return self.argument_schema


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
        tools: List[Tool] | None = None,
        response_schema: Type[BaseModel] | None = None,
        **kwargs,
    ):
        self.system_message = system_message
        self.messages = messages
        self.tools = tools or []
        self.response_schema = response_schema


"""
Outputs
"""


@dataclass
class ThoughtOutputMessage:
    content: str
    type: str = "thinking"

    def __init__(self, content: str):
        self.content = content

    def text(self) -> str:
        return self.content


@dataclass
class TextOutputMessage:
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


@dataclass
class ToolCallOutputMessage:
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
        arguments: dict,
        schema: Type[BaseModel],
        tool_call_id: Optional[str] = None,
    ):
        self.name = name
        self.tool_call_id = tool_call_id or str(uuid.uuid4())
        # Convert arguments to string if they're a dict

        self._arguments = SchemaResult(arguments, schema)

    def __str__(self) -> str:
        return json.dumps(self._arguments.raw())

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


OutputMessage = TextOutputMessage | ToolCallOutputMessage | ThoughtOutputMessage


class ResponseFormat(str, Enum):
    """Type of response format requested from the LLM"""

    TEXT = "text"
    JSON = "json_object"
    STRUCTURED = "structured"


class FinishType(str, Enum):
    """Type of completion finish"""

    SUCCESS = "stop"  # Normal completion
    TOOL_USE = "tool_calls"  # Stopped to make tool calls
    FAIL_LENGTH = "length"  # Hit token limit
    FAIL_FILTER = "content_filter"  # Content was filtered
    FAIL_ERROR = "error"  # Other error occurred


@dataclass
class CompletionInfo:
    """Information about how the completion finished"""

    finish_reason: FinishType

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None

    refusal: Optional[str] = None

    FAILURE_REASONS: ClassVar[list[FinishType]] = [
        FinishType.FAIL_LENGTH,
        FinishType.FAIL_FILTER,
        FinishType.FAIL_ERROR,
    ]


class LLMResponseError(Exception):
    pass


class LLMResponse:
    """Complete response from an LLM, including all outputs and usage information."""

    def __init__(
        self,
        messages: List[OutputMessage],
        cost: Optional[float] = 0.0,
        completion_info: Optional[CompletionInfo] = None,
        response_object=None,
    ):
        self._messages = messages
        self.cost = cost
        self.completion_info = completion_info
        self.response_object = response_object

    def result(self) -> Optional["SchemaResult"]:
        """Get schema validation result if a schema was specified"""
        if self.response_object is None:
            raise Exception("uh, no")
        return self.response_object

    def text(self, include_tools=False, include_thoughts=False):
        def _to_text(message: OutputMessage) -> str:
            if isinstance(message, TextOutputMessage):
                return message.content
            elif isinstance(message, ToolCallOutputMessage):
                return str(message)
            elif isinstance(message, ThoughtOutputMessage):
                return message.content

        def _should_include(message: OutputMessage) -> bool:
            if isinstance(message, TextOutputMessage):
                return True
            elif isinstance(message, ToolCallOutputMessage):
                return include_tools
            elif isinstance(message, ThoughtOutputMessage):
                return include_thoughts

        return "\n".join(
            [
                _to_text(message)
                for message in self._messages
                if _should_include(message)
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
        return [msg for msg in self._messages if isinstance(msg, TextOutputMessage)]

    def tool_calls(self) -> List[ToolCallOutputMessage]:
        """Get all tool messages in the response"""
        return [msg for msg in self._messages if isinstance(msg, ToolCallOutputMessage)]

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

    _parsed_json: dict | None = None

    def __init__(self, input_data: str | dict, schema: Type[BaseModel]):
        """
        Args:
            input_data: Raw text response from LLM
            schema: Optional Pydantic model to validate against
        """
        self.input_data = input_data
        self._schema = schema
        if isinstance(self.input_data, str):
            try:
                self._parsed_json = json.loads(self.input_data)
            except json.JSONDecodeError:
                self._parsed_json = None
        else:
            self._parsed_json = self.input_data

    # @classmethod
    # def from_parsed(
    #     cls: "SchemaResult", parsed_obj: BaseModel, schema: Type[BaseModel]
    # ):
    #     json_dump = parsed_obj.model_dump_json
    #     return cls(json_dump, schema)

    def valid(self) -> bool:
        """Check if response is valid according to schema.

        If no schema was specified, checks if response is valid JSON.

        Returns:
            bool: True if valid, False otherwise
        """
        if self._parsed_json is None:
            return False

        # if not isinstance(self._schema, Type):
        #     # Todo, actually validate against a json schema
        #     return True

        if not issubclass(self._schema, BaseModel):
            raise ValueError("Schema must be a subclass of BaseModel")

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
        return self._parsed_json is not None

    def parse_obj(self) -> dict:
        """Parse response as JSON into dict/list.

        Returns:
            dict | list: Parsed JSON data

        Raises:
            JSONDecodeError: If response is not valid JSON
        """
        if self._parsed_json is None:
            raise json.JSONDecodeError("Invalid JSON", str(self.input_data), 0)
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
        if not isinstance(self._schema, Type):
            raise ValueError("No schema specified")
        if not issubclass(self._schema, BaseModel):
            raise ValueError("Schema must be a subclass of BaseModel")
        if self._parsed_json is None:
            raise json.JSONDecodeError("Invalid JSON", str(self.input_data), 0)

        try:
            return self._schema(**self._parsed_json)
        except ValidationError as e:
            raise SchemaValidationError(str(e))

    def raw(self) -> str | dict | None:
        """Get raw response text.

        Returns:
            str: Original response text
        """
        return self.input_data


class SchemaValidationError(Exception):
    pass
