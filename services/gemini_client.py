
import os

from google import genai

from services.token_service import normalize_usage


class GeminiClient:
    def __init__(self, sdk_client=None, model=None):
        self.client = sdk_client or genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = model or os.getenv(
            "GEMINI_MODEL", "gemini-3.1-flash-lite"
        )

    def stream_response(self, messages, temperature):
        contents = []

        for message in messages:
            role = message["role"]
            if role == "assistant":
                role = "model"

            contents.append({
                "role": role,
                "parts": [{"text": message["content"]}],
            })

        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config={"temperature": temperature},
        )

        latest_usage = None

        for chunk in stream:
            if chunk.text:
                yield {"type": "text", "text": chunk.text}

            if chunk.usage_metadata is not None:
                latest_usage = chunk.usage_metadata

        if latest_usage is not None:
            yield {
                "type": "usage",
                "usage": normalize_usage(latest_usage),
            }
