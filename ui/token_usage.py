
import streamlit as st


def render_token_usage(usage):
    col1, col2, col3 = st.columns(3)

    col1.metric("Input tokens", usage.get("input_tokens", 0))
    col2.metric("Output tokens", usage.get("output_tokens", 0))
    col3.metric("Total tokens", usage.get("total_tokens", 0))
