from state.session import (
    ChatSession,
    SqliteChatStore,
    get_chat_store,
    reset_chat_store,
    initialize_session,
    reset_chat,
)


def test_chat_session_dataclass():
    session = ChatSession(title="Test")
    assert session.messages == []
    assert session.token_usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_sqlite_chat_store_multi_chat_readiness():
    store = SqliteChatStore(db_path=":memory:")
    assert len(store.get_all_chats()) == 1

    chat1 = store.get_active_chat()
    store.add_message("user", "Chat 1 message")
    store.set_token_usage({"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})

    # Create a second chat
    chat2 = store.create_chat(title="Chat 2")
    assert store.get_active_chat().id == chat2.id
    assert store.get_messages() == []

    store.add_message("user", "Chat 2 message")
    assert len(store.get_messages()) == 1
    assert store.get_messages()[0]["content"] == "Chat 2 message"

    # Switch back to chat 1 and verify message isolation
    assert store.switch_chat(chat1.id) is True
    assert len(store.get_messages()) == 1
    assert store.get_messages()[0]["content"] == "Chat 1 message"
    assert store.get_token_usage()["total_tokens"] == 10

    # Pop message from active chat
    popped = store.pop_message()
    assert popped["content"] == "Chat 1 message"
    assert len(store.get_messages()) == 0


def test_chat_store_singleton():
    store1 = get_chat_store(db_path=":memory:")
    store2 = get_chat_store(db_path=":memory:")
    assert store1 is store2


def test_legacy_session_state_helpers():
    state = {}
    initialize_session(state, db_path=":memory:")
    assert state["messages"] == []
    assert state["token_usage"]["total_tokens"] == 0
    assert isinstance(state["temperature"], (int, float))

    state["messages"].append({"role": "user", "content": "Hi"})
    reset_chat(state)
    assert state["messages"] == []
