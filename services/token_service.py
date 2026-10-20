
def normalize_usage(metadata):
    """Convert Gemini token metadata into a consistent dictionary."""

    if metadata is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    input_tokens = metadata.prompt_token_count or 0
    output_tokens = metadata.candidates_token_count or 0

    total_tokens = metadata.total_token_count
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
