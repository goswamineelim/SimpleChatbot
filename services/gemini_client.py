import os
from google import genai
from google.genai import types

from services.token_service import normalize_usage


def format_tool_attribution(executed_tools):
    """Formats a user-facing markdown callout detailing all tools invoked during generation."""
    if not executed_tools:
        return ""

    lines = ["> 🛠️ **Tools Used:**"]
    for t in executed_tools:
        name = t["tool"]
        args = t.get("args", {})
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        result_str = str(t.get("result", "")).strip()
        if len(result_str) > 120:
            result_str = result_str[:117] + "..."
        result_str = result_str.replace("\n", " ")
        icon = "🧮" if name == "calculate" else ("🌐" if name == "web_search" else "🔧")
        lines.append(f"> - {icon} `{name}({args_str})` → `{result_str}`")

    return "\n".join(lines) + "\n\n"


class GeminiClient:
    def __init__(self, sdk_client=None, model=None, tool_registry=None):
        self.client = sdk_client or genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = model or os.getenv(
            "GEMINI_MODEL", "gemini-3.1-flash-lite"
        )
        self.tool_registry = tool_registry

    def stream_response(self, messages, temperature, max_tool_hops=5):
        contents = []

        for message in messages:
            role = message["role"]
            if role == "assistant":
                role = "model"

            contents.append({
                "role": role,
                "parts": [{"text": message["content"]}],
            })

        tools = self.tool_registry.get_tools() if self.tool_registry else []
        config = {"temperature": temperature}
        if tools:
            config["tools"] = tools
            config["automatic_function_calling"] = {"disable": True}

        executed_tools = []
        attribution_yielded = False
        latest_usage = None

        for _ in range(max_tool_hops):
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            pending_function_calls = []
            model_turn_parts = []

            for chunk in stream:
                # Capture the original parts from Gemini to preserve thought_signature
                if getattr(chunk, "candidates", None):
                    for cand in chunk.candidates:
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            for p in content.parts:
                                model_turn_parts.append(p)
                                if getattr(p, "function_call", None):
                                    pending_function_calls.append(p.function_call)

                # Fallback to chunk.function_calls if not in parts
                if not pending_function_calls and getattr(chunk, "function_calls", None):
                    for call in chunk.function_calls:
                        pending_function_calls.append(call)

                if chunk.text:
                    if executed_tools and not attribution_yielded:
                        yield {
                            "type": "text",
                            "text": format_tool_attribution(executed_tools),
                        }
                        attribution_yielded = True

                    yield {"type": "text", "text": chunk.text}

                if getattr(chunk, "usage_metadata", None) is not None:
                    latest_usage = chunk.usage_metadata

            if pending_function_calls:
                tool_response_parts = []

                for call in pending_function_calls:
                    yield {
                        "type": "tool_start",
                        "tool": call.name,
                        "args": call.args,
                    }

                    result = (
                        self.tool_registry.execute(call.name, call.args)
                        if self.tool_registry
                        else f"Error: No tool registry available to execute '{call.name}'."
                    )

                    executed_tools.append({
                        "tool": call.name,
                        "args": call.args,
                        "result": result,
                    })

                    yield {
                        "type": "tool_complete",
                        "tool": call.name,
                        "args": call.args,
                        "result": result,
                    }

                    try:
                        resp_part = types.Part.from_function_response(
                            name=call.name, response={"result": result}
                        )
                    except Exception:
                        resp_part = {"function_response": {"name": call.name, "response": {"result": result}}}

                    tool_response_parts.append(resp_part)

                # Append model's turn with original parts preserving thought_signature
                if model_turn_parts:
                    contents.append({"role": "model", "parts": model_turn_parts})
                else:
                    synth_call_parts = [
                        types.Part.from_function_call(name=c.name, args=c.args)
                        for c in pending_function_calls
                    ]
                    contents.append({"role": "model", "parts": synth_call_parts})

                # Append user's turn containing all tool responses for this hop
                contents.append({"role": "user", "parts": tool_response_parts})
            else:
                break

        if executed_tools and not attribution_yielded:
            yield {
                "type": "text",
                "text": format_tool_attribution(executed_tools),
            }

        if latest_usage is not None:
            yield {
                "type": "usage",
                "usage": normalize_usage(latest_usage),
            }
