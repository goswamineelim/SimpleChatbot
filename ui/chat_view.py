
import streamlit as st


def render_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
