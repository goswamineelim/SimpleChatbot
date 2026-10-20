from types import SimpleNamespace

from services.token_service import normalize_usage


def test_normalizes_token_usage():
    metadata = SimpleNamespace(
        prompt_token_count=12,
        candidates_token_count=25,
        total_token_count=37,
    )

    result = normalize_usage(metadata)

    assert result == {
        "input_tokens": 12,
        "output_tokens": 25,
        "total_tokens": 37,
    }


def test_missing_metadata_returns_zero_counts():
    result = normalize_usage(None)

    assert result == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_missing_individual_counts_default_to_zero():
    metadata = SimpleNamespace(
        prompt_token_count=None,
        candidates_token_count=10,
        total_token_count=None,
    )

    result = normalize_usage(metadata)

    assert result == {
        "input_tokens": 0,
        "output_tokens": 10,
        "total_tokens": 10,
    }


def test_preserves_provider_total_when_it_differs_from_sum():
    metadata = SimpleNamespace(
        prompt_token_count=2,
        candidates_token_count=9,
        total_token_count=15,
    )

    result = normalize_usage(metadata)

    assert result["total_tokens"] == 15
