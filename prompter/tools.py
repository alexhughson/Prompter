from dataclasses import dataclass
from typing import Dict, Type, get_type_hints
from .schemas import Tool


@dataclass
class WeatherArgs:
    """Arguments for the weather tool"""
    location: str
    units: str = "celsius"


@dataclass
class SearchArgs:
    """Arguments for the search tool"""
    query: str
    max_results: int = 10


def dataclass_to_json_schema(cls: Type) -> Dict:
    """Convert a dataclass to a JSON schema"""
    properties = {}
    required = []

    hints = get_type_hints(cls)
    for field_name, field_type in hints.items():
        # Get the field from the dataclass
        field = cls.__dataclass_fields__[field_name]

        # Determine if field is required (no default value)
        if field.default == field.default_factory == None:
            required.append(field_name)

        # Map Python types to JSON schema types
        if field_type == str:
            field_schema = {"type": "string"}
        elif field_type == int:
            field_schema = {"type": "integer"}
        elif field_type == float:
            field_schema = {"type": "number"}
        elif field_type == bool:
            field_schema = {"type": "boolean"}
        else:
            field_schema = {"type": "string"}  # fallback

        # Add description if available
        if field.metadata.get("description"):
            field_schema["description"] = field.metadata["description"]

        properties[field_name] = field_schema

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


class ToolRegistry:
    """Registry of available tools with their schemas"""

    tools = {
        "get_weather": Tool(
            name="get_weather",
            description="Get the current weather for a location",
            argument_schema=dataclass_to_json_schema(WeatherArgs)
        ),
        "search": Tool(
            name="search",
            description="Search for information on a topic",
            argument_schema=dataclass_to_json_schema(SearchArgs)
        )
    }

    @classmethod
    def get_tool(cls, name: str) -> Tool:
        """Get a tool by name"""
        return cls.tools.get(name)

    @classmethod
    def get_schema(cls, name: str) -> Type:
        """Get the argument schema class for a tool"""
        schema_map = {
            "get_weather": WeatherArgs,
            "search": SearchArgs
        }
        return schema_map.get(name)

    @classmethod
    def all_tools(cls) -> list[Tool]:
        """Get all available tools"""
        return list(cls.tools.values())
