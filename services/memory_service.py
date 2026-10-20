def estimate_tokens(text: str) -> int:
    """Estimate token count using the standard ~4 chars per token heuristic."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of message dicts including turn overhead."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("content", "")) + 4
    return total


class MemoryService:
    """Summary-based short-term memory management within configurable token budgets."""

    def __init__(self, client=None):
        self.client = client

    def prepare_context(
        self,
        messages: list,
        current_summary: str = "",
        conv_token_limit: int = 1500,
        summary_budget: int = 200,
    ) -> tuple:
        """
        Prepare context payload for the LLM.
        Summarizes older messages if total tokens exceed conv_token_limit.
        Returns: (context_messages, updated_summary)
        """
        total_tokens = estimate_messages_tokens(messages)
        if current_summary:
            total_tokens += estimate_tokens(current_summary)

        # Under limit: send messages directly (with prepended summary if one exists)
        if total_tokens <= conv_token_limit or len(messages) <= 2:
            if current_summary:
                context_messages = [
                    {
                        "role": "user",
                        "content": f"[Earlier conversation summary: {current_summary}]",
                    },
                    {
                        "role": "assistant",
                        "content": "Understood. I will keep this context in mind.",
                    },
                    *messages,
                ]
                return context_messages, current_summary
            return list(messages), ""

        # Over limit: partition into older messages (to summarize) and recent messages (to preserve)
        recent_messages = []
        recent_tokens = 0
        split_index = len(messages)

        # Preserve recent messages within roughly half the token budget, at least keeping latest message
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens(messages[i].get("content", "")) + 4
            if recent_tokens + msg_tokens > (conv_token_limit // 2) and len(recent_messages) >= 2:
                split_index = i + 1
                break
            recent_messages.insert(0, messages[i])
            recent_tokens += msg_tokens
            split_index = i

        if split_index == 0:
            split_index = max(1, len(messages) - 2)
            recent_messages = messages[split_index:]

        older_messages = messages[:split_index]

        older_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in older_messages
        )
        prev_summary_prompt = (
            f"Previous summary:\n{current_summary}\n\n" if current_summary else ""
        )
        prompt = [
            {
                "role": "user",
                "content": (
                    f"Summarize the key information from this earlier conversation into a concise summary "
                    f"of approximately {summary_budget} tokens. Retain important facts, decisions, and context.\n\n"
                    f"{prev_summary_prompt}"
                    f"Conversation to condense:\n{older_text}"
                ),
            }
        ]

        summary_text = ""
        if self.client:
            for event in self.client.stream_response(prompt, temperature=0.3):
                if event.get("type") == "text":
                    summary_text += event.get("text", "")

        summary_text = summary_text.strip() or current_summary

        context_messages = [
            {
                "role": "user",
                "content": f"[Earlier conversation summary: {summary_text}]",
            },
            {
                "role": "assistant",
                "content": "Understood. I will keep this context in mind.",
            },
            *recent_messages,
        ]

        return context_messages, summary_text
