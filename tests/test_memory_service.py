from unittest.mock import Mock
import pytest

from services.memory_service import MemoryService, estimate_tokens, estimate_messages_tokens


def test_estimate_tokens_empty_and_text():
    assert estimate_tokens("") == 0
    # ~4 chars per token heuristic
    assert estimate_tokens("hello world") >= 2


def test_estimate_messages_tokens():
    messages = [
        {"role": "user", "content": "Hello there!"},
        {"role": "assistant", "content": "General Kenobi!"},
    ]
    tokens = estimate_messages_tokens(messages)
    assert tokens > 0
    assert tokens == estimate_tokens("Hello there!") + estimate_tokens("General Kenobi!") + 8


def test_under_token_limit_without_existing_summary_returns_messages_verbatim():
    mock_client = Mock()
    service = MemoryService(client=mock_client)

    messages = [
        {"role": "user", "content": "Short question"},
        {"role": "assistant", "content": "Short answer"},
    ]

    context_messages, updated_summary = service.prepare_context(
        messages=messages,
        current_summary="",
        conv_token_limit=1000,
        summary_budget=200,
    )

    assert context_messages == messages
    assert updated_summary == ""
    mock_client.stream_response.assert_not_called()


def test_under_token_limit_with_existing_summary_prepends_summary():
    mock_client = Mock()
    service = MemoryService(client=mock_client)

    messages = [
        {"role": "user", "content": "Follow up question"},
    ]

    context_messages, updated_summary = service.prepare_context(
        messages=messages,
        current_summary="User likes Python and SQLite.",
        conv_token_limit=1000,
        summary_budget=200,
    )

    assert len(context_messages) == 3
    assert "User likes Python and SQLite." in context_messages[0]["content"]
    assert context_messages[0]["role"] == "user"
    assert context_messages[1]["role"] == "assistant"
    assert context_messages[2] == messages[0]
    assert updated_summary == "User likes Python and SQLite."
    mock_client.stream_response.assert_not_called()


def test_exceeding_token_limit_triggers_summarization():
    mock_client = Mock()
    mock_client.stream_response.return_value = iter([
        {"type": "text", "text": "Condensed summary of conversation."}
    ])
    service = MemoryService(client=mock_client)

    # Create several messages that exceed a small token limit (e.g. 50 tokens)
    messages = [
        {"role": "user", "content": "A" * 100},
        {"role": "assistant", "content": "B" * 100},
        {"role": "user", "content": "C" * 100},
        {"role": "assistant", "content": "D" * 100},
        {"role": "user", "content": "Latest user prompt"},
    ]

    context_messages, updated_summary = service.prepare_context(
        messages=messages,
        current_summary="",
        conv_token_limit=50,
        summary_budget=25,
    )

    mock_client.stream_response.assert_called_once()
    assert updated_summary == "Condensed summary of conversation."
    assert any("Condensed summary of conversation." in m["content"] for m in context_messages)
    # The latest user message must be preserved
    assert context_messages[-1]["content"] == "Latest user prompt"


def test_prepare_context_with_long_term_memories():
    mock_client = Mock()
    service = MemoryService(client=mock_client)

    messages = [
        {"role": "user", "content": "What tools should I use?"},
    ]
    ltm = [
        "User prefers Python over JavaScript",
        "User works on GenAI applications",
    ]

    context_messages, _ = service.prepare_context(
        messages=messages,
        long_term_memories=ltm,
    )

    # First turn should inject long-term memories
    assert len(context_messages) == 3
    assert "User prefers Python over JavaScript" in context_messages[0]["content"]
    assert "User works on GenAI applications" in context_messages[0]["content"]
    assert context_messages[1]["role"] == "assistant"
    assert context_messages[2] == messages[0]


def test_prepare_context_with_summary_and_long_term_memories():
    mock_client = Mock()
    service = MemoryService(client=mock_client)

    messages = [
        {"role": "user", "content": "Next step?"},
    ]
    ltm = ["User lives in Berlin"]
    summary = "Earlier talked about packing bags."

    context_messages, updated_summary = service.prepare_context(
        messages=messages,
        current_summary=summary,
        long_term_memories=ltm,
    )

    assert len(context_messages) == 3
    content = context_messages[0]["content"]
    assert "User lives in Berlin" in content
    assert "Earlier talked about packing bags." in content
    assert updated_summary == summary

