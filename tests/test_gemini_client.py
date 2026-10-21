
from types import SimpleNamespace
from unittest.mock import Mock

from services.gemini_client import GeminiClient


def make_chunk(text=None, usage=None):
    return SimpleNamespace(
        text=text,
        usage_metadata=usage,
    )


def test_streams_text_chunks_in_order():
    sdk = Mock()
    sdk.models.generate_content_stream.return_value = iter([
        make_chunk("Hello"),
        make_chunk(" there"),
        make_chunk("!"),
    ])

    client = GeminiClient(sdk_client=sdk, model="test-model")

    events = list(client.stream_response(
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.7,
    ))

    text_chunks = [
        event["text"] for event in events
        if event["type"] == "text"
    ]

    assert text_chunks == ["Hello", " there", "!"]


def test_forwards_temperature_to_gemini():
    sdk = Mock()
    sdk.models.generate_content_stream.return_value = iter([
        make_chunk("Response")
    ])

    client = GeminiClient(sdk_client=sdk, model="test-model")

    list(client.stream_response(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=1.2,
    ))

    kwargs = sdk.models.generate_content_stream.call_args.kwargs

    assert kwargs["config"]["temperature"] == 1.2


def test_emits_final_token_usage():
    sdk = Mock()
    usage = SimpleNamespace(
        prompt_token_count=4,
        candidates_token_count=8,
        total_token_count=12,
    )

    sdk.models.generate_content_stream.return_value = iter([
        make_chunk("Hello"),
        make_chunk(usage=usage),
    ])

    client = GeminiClient(sdk_client=sdk, model="test-model")

    events = list(client.stream_response(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
    ))

    usage_events = [
        event for event in events
        if event["type"] == "usage"
    ]

    assert usage_events == [{
        "type": "usage",
        "usage": {
            "input_tokens": 4,
            "output_tokens": 8,
            "total_tokens": 12,
        },
    }]


def test_handles_missing_usage_metadata():
    sdk = Mock()
    sdk.models.generate_content_stream.return_value = iter([
        make_chunk("Hello")
    ])

    client = GeminiClient(sdk_client=sdk, model="test-model")

    events = list(client.stream_response(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
    ))

    assert not any(event["type"] == "usage" for event in events)


def test_tool_calling_loop_executes_tool_and_yields_attribution():
    sdk = Mock()

    # Step 1: Model returns a function call
    func_call = SimpleNamespace(
        name="calculate",
        args={"expression": "32**5"},
    )
    chunk_1 = SimpleNamespace(
        text=None,
        function_calls=[func_call],
        usage_metadata=None,
    )

    # Step 2: Model returns final text after getting function response
    chunk_2 = SimpleNamespace(
        text="The calculated answer is 33554432.",
        function_calls=None,
        usage_metadata=None,
    )

    sdk.models.generate_content_stream.side_effect = [
        iter([chunk_1]),
        iter([chunk_2]),
    ]

    mock_registry = Mock()
    mock_registry.get_tools.return_value = [Mock()]
    mock_registry.execute.return_value = "33554432"

    client = GeminiClient(
        sdk_client=sdk,
        model="test-model",
        tool_registry=mock_registry,
    )

    events = list(client.stream_response(
        messages=[{"role": "user", "content": "What is 32^5?"}],
        temperature=0.7,
    ))

    mock_registry.execute.assert_called_once_with(
        "calculate",
        {"expression": "32**5"},
    )

    types_emitted = [e["type"] for e in events]
    assert "tool_start" in types_emitted
    assert "tool_complete" in types_emitted
    assert "text" in types_emitted

    # Verify that the text includes tool attribution
    full_text = "".join(e["text"] for e in events if e["type"] == "text")
    assert "🛠️ **Tools Used:**" in full_text
    assert "calculate(expression='32**5')" in full_text
    assert "33554432" in full_text
    assert "The calculated answer is 33554432." in full_text
