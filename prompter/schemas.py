from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Type, Union, Any
from abc import ABC, abstractmethod
import json
from pydantic import BaseModel


class Message(BaseModel):
    """Base class for all message types"""

    role: str
    content: str

    @abstractmethod
    def message_type(self) -> str:
        pass


class UserMessage(Message):
    content: str
    role: str = "user"

    def message_type(self) -> str:
        return "text"


class AssistantMessage(Message):
    content: str
    role: str = "assistant"

    def message_type(self) -> str:
        return "text"


class ImageMessage(Message):
    role: str = "user"
    content: Optional[str] = None
    url: str  # URL is required for image messages
    media_type: Optional[str] = "image/jpeg"  # Default to JPEG but can be overridden

    def message_type(self) -> str:
        return "image"


class ModelPricing(BaseModel):
    """Pricing information for a specific model"""

    input_price: float  # Cost per 1K input tokens
    output_price: float  # Cost per 1K output tokens


class UsageInfo(BaseModel):
    """Token usage information from an API call"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Tool(BaseModel):
    """Represents a tool that can be called by the LLM"""

    name: str
    description: str
    argument_schema: Union[
        Type[BaseModel], Dict
    ]  # Can be either a Pydantic model or Dict schema

    def get_schema(self) -> Dict:
        """Get the JSON schema for the tool's arguments"""
        if isinstance(self.argument_schema, type) and issubclass(
            self.argument_schema, BaseModel
        ):
            return self.argument_schema.model_json_schema()
        return self.argument_schema


class ToolPool(BaseModel):
    """Collection of available tools"""

    tools: List[Tool] = field(default_factory=list)

    @property
    def tool_definitions(self) -> List[Dict]:
        """Return the tools in a format suitable for LLM API consumption"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_schema(),
            }
            for tool in self.tools
        ]

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return next((tool for tool in self.tools if tool.name == name), None)


class Prompt:
    """Complete prompt with system message, conversation history, and available tools"""

    TOOL_USE_REQUIRED = "hey"
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


class OutputMessage(BaseModel):
    """Base class for all types of LLM outputs"""

    type: str

    @abstractmethod
    def text(self) -> str:
        pass


class TextOutputMessage(OutputMessage):
    content: str
    type: str = "text"

    def text(self) -> str:
        return self.content


class ImageOutputMessage(OutputMessage):
    image_url: str
    type: str = "image"

    def text(self) -> str:
        return f"Image: {self.image_url}"


class ToolCallArguments(BaseModel):
    """Arguments for a tool call"""

    arguments: Union[Dict, str]  # Can be either a Dict or JSON string
    _schema: Optional[Type[BaseModel]] = None

    def set_schema(self, schema: Type[BaseModel]) -> None:
        """Set the Pydantic model schema for parsing"""
        self._schema = schema

    def parse(self) -> Union[BaseModel, Dict]:
        """Parse the arguments into the appropriate type"""
        # First ensure we have a dict
        args_dict = (
            self.arguments
            if isinstance(self.arguments, dict)
            else json.loads(self.arguments)
        )

        # If we have a schema, parse into the model
        if self._schema is not None:
            return self._schema.model_validate(args_dict)
        return args_dict


class ToolCallOutputMessage:
    name: str
    _arguments: ToolCallArguments

    def __init__(self, name: str, arguments: ToolCallArguments):
        self.name = name
        self._arguments = arguments

    def set_schema(self, schema: Type[BaseModel]) -> None:
        self._arguments.set_schema(schema)

    @property
    def arguments(self) -> ToolCallArguments:
        return self._arguments.parse()

    def text(self) -> str:
        parsed_args = self.arguments.parse()
        if isinstance(parsed_args, BaseModel):
            args_str = parsed_args.model_dump_json()
        else:
            args_str = json.dumps(parsed_args)
        return f"Tool {self.name} called with arguments {args_str}"


class LLMResponse:
    """Complete response from an LLM, including all outputs and usage information"""

    messages: List[OutputMessage]
    usage: UsageInfo
    cost: float = 0.0

    def __init__(
        self, messages: List[OutputMessage], usage: UsageInfo, cost: float = 0.0
    ):
        self.messages = messages
        self.usage = usage
        self.cost = cost

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

    def only_tool_calls(self) -> List[ToolCallOutputMessage]:
        return [
            message
            for message in self.messages
            if isinstance(message, ToolCallOutputMessage)
        ]

    def only_text(self) -> List[TextOutputMessage]:
        return [
            message
            for message in self.messages
            if isinstance(message, TextOutputMessage)
        ]

    def text(self) -> str:
        return "\n".join([message.text() for message in self.messages])

    def raise_for_status(self):
        pass
