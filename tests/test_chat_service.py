
from unittest.mock import Mock

import pytest

from services.chat_service import ChatService


def test_forwards_history_and_temperature():
    provider = Mock()
    expected_events = [
        {"type": "text", "text": "Hello"},
        {
            "type": "usage",
            "usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
            },
        },
    ]
    provider.stream_response.return_value = iter(expected_events)

    service = ChatService(provider)

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "model", "content": "Hello!"},
        {"role": "user", "content": "What did I say?"},
    ]

    events = list(service.stream_reply(messages, temperature=0.4))

    provider.stream_response.assert_called_once_with(
        messages=messages,
        temperature=0.4,
    )
    assert events == expected_events


def test_preserves_event_order():
    provider = Mock()
    provider.stream_response.return_value = iter([
        {"type": "text", "text": "A"},
        {"type": "text", "text": "B"},
        {"type": "text", "text": "C"},
    ])

    service = ChatService(provider)

    events = list(service.stream_reply(
        [{"role": "user", "content": "Test"}],
        temperature=0.7,
    ))

    assert [e["text"] for e in events] == ["A", "B", "C"]


def test_propagates_provider_errors():
    provider = Mock()
    provider.stream_response.side_effect = RuntimeError("Provider failed")

    service = ChatService(provider)

    with pytest.raises(RuntimeError, match="Provider failed"):
        list(service.stream_reply(
            [{"role": "user", "content": "Test"}],
            temperature=0.7,
        ))
