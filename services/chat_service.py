
class ChatService:
    def __init__(self, provider):
        self.provider = provider

    def stream_reply(self, messages, temperature):
        yield from self.provider.stream_response(
            messages=messages,
            temperature=temperature,
        )
