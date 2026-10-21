from typing import Callable, Dict, List, Any


class ToolRegistry:
    """Manages registration, schema discovery, and execution of agent tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, tool_func: Callable) -> None:
        """Registers a Python function as a tool."""
        tool_name = tool_func.__name__
        self._tools[tool_name] = tool_func

    def has_tool(self, name: str) -> bool:
        """Returns True if the tool is registered."""
        return name in self._tools

    def get_tool(self, name: str) -> Callable:
        """Retrieves a registered tool by name."""
        return self._tools.get(name)

    def get_tools(self) -> List[Callable]:
        """Returns a list of all registered callable tools."""
        return list(self._tools.values())

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Executes a registered tool by name with arguments dict, returning a string result."""
        tool_func = self.get_tool(tool_name)
        if not tool_func:
            return f"Error: Tool '{tool_name}' is not registered."

        try:
            result = tool_func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"


def get_default_tool_registry() -> ToolRegistry:
    """Returns a pre-configured registry containing calculate and web_search tools."""
    from services.tools.calculator_tool import calculate
    from services.tools.web_search_tool import web_search

    registry = ToolRegistry()
    registry.register(calculate)
    registry.register(web_search)
    return registry
