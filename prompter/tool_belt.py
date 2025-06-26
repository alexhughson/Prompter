from typing import Any, Optional

from pydantic import BaseModel

from .schemas import Tool, ToolCall


class ToolBelt:
    def __init__(self, tools: Optional[list[Tool]] = None):
        self.tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.add(tool)

    def add(self, tool: Tool):
        self.tools[tool.name] = tool

    def tool_list(self) -> list[Tool]:
        return list(self.tools.values())

    def parse_call(self, tool_call: ToolCall) -> Any:
        if tool_call.name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_call.name}")

        tool = self.tools[tool_call.name]

        if tool.params is None:
            return tool_call.arguments or {}

        if isinstance(tool.params, type) and issubclass(tool.params, BaseModel):
            return tool.params(**tool_call.arguments)
        else:
            return tool_call.arguments
