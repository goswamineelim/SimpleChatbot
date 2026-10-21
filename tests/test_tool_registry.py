import pytest
from services.tools.tool_registry import ToolRegistry


def sample_tool(text: str) -> str:
    """A sample tool for testing."""
    return f"Processed: {text}"


def test_registry_register_and_execute():
    registry = ToolRegistry()
    registry.register(sample_tool)

    assert registry.has_tool("sample_tool")
    result = registry.execute("sample_tool", {"text": "hello"})
    assert result == "Processed: hello"


def test_registry_execute_unregistered():
    registry = ToolRegistry()
    result = registry.execute("unknown_tool", {})
    assert "Error: Tool 'unknown_tool' is not registered." in result


def test_registry_execute_handles_exceptions():
    def broken_tool(val: int) -> str:
        """Broken tool."""
        raise RuntimeError("Something broke!")

    registry = ToolRegistry()
    registry.register(broken_tool)

    result = registry.execute("broken_tool", {"val": 1})
    assert "Error executing tool 'broken_tool'" in result
    assert "Something broke!" in result


def test_registry_get_tools():
    registry = ToolRegistry()
    registry.register(sample_tool)

    tools = registry.get_tools()
    assert len(tools) == 1
    assert tools[0] == sample_tool
