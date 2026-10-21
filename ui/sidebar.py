import streamlit as st

from config.settings import (
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    DEFAULT_TEMPERATURE,
    DEFAULT_CONV_TOKEN_LIMIT,
    MIN_CONV_TOKEN_LIMIT,
    MAX_CONV_TOKEN_LIMIT,
    DEFAULT_SUMMARY_BUDGET,
    MIN_SUMMARY_BUDGET,
    MAX_SUMMARY_BUDGET,
)
from state.session import get_chat_store


class SidebarSettings:
    """Encapsulates sidebar user configurations while allowing tuple unpacking."""

    def __init__(self, temperature, conv_token_limit, summary_budget):
        self.temperature = temperature
        self.conv_token_limit = conv_token_limit
        self.summary_budget = summary_budget

    def __iter__(self):
        return iter((self.temperature, self.conv_token_limit, self.summary_budget))

    def __getitem__(self, item):
        return getattr(self, item)


def render_sidebar(chat_store=None, ltm_service=None):
    if chat_store is None:
        chat_store = get_chat_store()

    with st.sidebar:
        st.header("Chats")

        chats = chat_store.get_all_chats()
        active_id = chat_store.get_active_chat_id()
        chat_dict = {c.id: c.title for c in chats}

        # Chat selection
        chat_ids = list(chat_dict.keys())
        current_index = chat_ids.index(active_id) if active_id in chat_ids else 0

        selected_id = st.selectbox(
            "Select chat",
            options=chat_ids,
            index=current_index,
            format_func=lambda cid: chat_dict.get(cid, "Chat"),
            label_visibility="collapsed",
        )

        if selected_id != active_id:
            chat_store.switch_chat(selected_id)
            st.rerun()

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("New chat", use_container_width=True):
                chat_store.create_chat()
                st.rerun()
        with col2:
            if len(chats) > 1:
                if st.button("🗑️", help="Delete active chat", use_container_width=True):
                    chat_store.delete_chat(active_id)
                    st.rerun()

        st.divider()
        st.header("Settings")

        temperature = st.slider(
            "Temperature",
            min_value=MIN_TEMPERATURE,
            max_value=MAX_TEMPERATURE,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            key="temperature",
        )

        st.subheader("Memory Limits")

        conv_token_limit = st.slider(
            "Conversation Token Limit",
            min_value=MIN_CONV_TOKEN_LIMIT,
            max_value=MAX_CONV_TOKEN_LIMIT,
            value=DEFAULT_CONV_TOKEN_LIMIT,
            step=100,
            key="conv_token_limit",
            help="Active tokens threshold before older history is summarized.",
        )

        summary_budget = st.slider(
            "Summary Token Budget",
            min_value=MIN_SUMMARY_BUDGET,
            max_value=MAX_SUMMARY_BUDGET,
            value=DEFAULT_SUMMARY_BUDGET,
            step=25,
            key="summary_budget",
            help="Target token budget for the condensed conversation summary.",
        )

        running_summary = chat_store.get_running_summary()
        if running_summary:
            with st.expander("Active Memory Summary"):
                st.caption(running_summary)

        if ltm_service is not None:
            memories = ltm_service.get_all_memories()
            with st.expander(f"🧠 Long-Term Memories ({len(memories)})"):
                if memories:
                    for mem in memories:
                        m_col1, m_col2 = st.columns([5, 1])
                        with m_col1:
                            st.markdown(f"• {mem['text']}")
                        with m_col2:
                            if st.button("🗑️", key=f"del_mem_{mem['id']}", help="Delete memory"):
                                ltm_service.delete_memory(mem["id"])
                                st.rerun()

                    if st.button("Clear all memories", use_container_width=True):
                        ltm_service.clear_all()
                        st.rerun()
                else:
                    st.caption("No long-term memories saved yet. Mention facts or preferences in conversation!")

        st.divider()
        st.subheader("Token usage")

        usage = chat_store.get_token_usage()

        st.metric("Input tokens", usage["input_tokens"])
        st.metric("Output tokens", usage["output_tokens"])
        st.metric("Total tokens", usage["total_tokens"])

    return SidebarSettings(temperature, conv_token_limit, summary_budget)
