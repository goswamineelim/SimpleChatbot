from state.sqlite_store import ChatSession, SqliteChatStore, ZERO_USAGE

DEFAULT_TEMPERATURE = 0.7

_store = None


def get_chat_store(db_path=None) -> SqliteChatStore:
    """Get or initialize the persistent SQLite chat store."""
    global _store
    if _store is None:
        _store = SqliteChatStore(db_path=db_path)
    return _store


def reset_chat_store(db_path=":memory:") -> SqliteChatStore:
    """Reset the global store, useful for tests or fresh sessions."""
    global _store
    _store = SqliteChatStore(db_path=db_path)
    return _store


def initialize_session(state=None, db_path="chats.db"):
    """Legacy helper maintained for backward compatibility."""
    if state is not None and isinstance(state, dict):
        if "messages" not in state:
            state["messages"] = []
        if "temperature" not in state:
            state["temperature"] = DEFAULT_TEMPERATURE
        if "token_usage" not in state:
            state["token_usage"] = ZERO_USAGE.copy()
    return get_chat_store(db_path=db_path)


def reset_chat(state=None):
    """Legacy helper maintained for backward compatibility."""
    if state is not None and isinstance(state, dict):
        state["messages"] = []
        state["token_usage"] = ZERO_USAGE.copy()
    else:
        get_chat_store().reset_active_chat()
