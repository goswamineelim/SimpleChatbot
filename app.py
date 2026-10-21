import streamlit as st
from dotenv import load_dotenv

from services.chat_service import ChatService
from services.gemini_client import GeminiClient
from services.memory_service import MemoryService
from services.long_term_memory_service import LongTermMemoryService
from state.session import get_chat_store
from ui.chat_view import render_messages
from ui.sidebar import render_sidebar

load_dotenv()

st.set_page_config(
    page_title="LLM Chat Playground",
    page_icon="💬",
    layout="wide",
)

st.title("LLM Chat Playground")

gemini_client = GeminiClient()
ltm_service = LongTermMemoryService(client=gemini_client)
chat_store = get_chat_store()
sidebar_settings = render_sidebar(chat_store, ltm_service=ltm_service)
temperature = sidebar_settings.temperature
conv_token_limit = sidebar_settings.conv_token_limit
summary_budget = sidebar_settings.summary_budget

# Render all previous messages from full storage history.
render_messages(chat_store.get_messages())

prompt = st.chat_input("Message the assistant...")

if prompt:
    # Save and display the user's message.
    chat_store.add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant's response.
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        latest_usage = None

        try:
            client = gemini_client
            service = ChatService(client)
            memory_service = MemoryService(client)

            # 1. Recall relevant long-term memories for this prompt
            recalled_memories = ltm_service.recall_memories(prompt, top_k=3)

            # 2. Prepare context within token limit, summary budget, and long-term memories
            context_messages, updated_summary = memory_service.prepare_context(
                messages=chat_store.get_messages(),
                current_summary=chat_store.get_running_summary(),
                conv_token_limit=conv_token_limit,
                summary_budget=summary_budget,
                long_term_memories=recalled_memories,
            )

            if updated_summary != chat_store.get_running_summary():
                chat_store.set_running_summary(updated_summary)

            for event in service.stream_reply(
                context_messages,
                temperature=temperature,
            ):
                if event["type"] == "text":
                    full_response += event["text"]
                    response_placeholder.markdown(
                        full_response + "▌"
                    )

                elif event["type"] == "usage":
                    latest_usage = event["usage"]

            response_placeholder.markdown(full_response)

            # Persist the completed assistant response.
            chat_store.add_message("assistant", full_response)

            if latest_usage is not None:
                chat_store.set_token_usage(latest_usage)

            # 3. Post-generation extraction of any long-term memories
            new_memories = ltm_service.extract_and_store_facts(prompt)
            if new_memories:
                st.toast(f"🧠 Remembered: {new_memories[0]}")

        except Exception as error:
            response_placeholder.empty()
            st.error(f"Unable to generate a response: {error}")

            # Remove the failed user turn.
            chat_store.pop_message()
