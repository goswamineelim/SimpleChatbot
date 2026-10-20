import pytest
from state.sqlite_store import SqliteChatStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "test_chats.db"
    return SqliteChatStore(db_path=str(db_file))


def test_initial_db_setup(store):
    chats = store.get_all_chats()
    assert len(chats) == 1
    assert chats[0].title == "Chat 1"
    assert store.get_messages() == []


def test_add_and_pop_message(store):
    store.add_message("user", "Hello SQLite")
    messages = store.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello SQLite"

    # Chat title auto-updates on first message
    active_chat = store.get_active_chat()
    assert active_chat.title == "Hello SQLite"

    popped = store.pop_message()
    assert popped == {"role": "user", "content": "Hello SQLite"}
    assert len(store.get_messages()) == 0


def test_multi_chat_creation_and_switching(store):
    chat1 = store.get_active_chat()
    store.add_message("user", "Message for chat 1")
    store.set_token_usage({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})

    # Create new chat
    chat2 = store.create_chat(title="Custom Chat 2")
    assert store.get_active_chat().id == chat2.id
    assert store.get_messages() == []

    store.add_message("user", "Message for chat 2")
    assert len(store.get_messages()) == 1

    # Switch back to chat 1 and check isolation
    assert store.switch_chat(chat1.id) is True
    assert len(store.get_messages()) == 1
    assert store.get_messages()[0]["content"] == "Message for chat 1"
    assert store.get_token_usage()["total_tokens"] == 30


def test_delete_chat(store):
    chat1 = store.get_active_chat()
    chat2 = store.create_chat(title="Chat 2")
    assert store.get_active_chat().id == chat2.id

    store.delete_chat(chat2.id)
    chats = store.get_all_chats()
    assert len(chats) == 1
    assert store.get_active_chat().id == chat1.id


def test_reset_active_chat(store):
    store.add_message("user", "To be cleared")
    store.set_token_usage({"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})

    store.reset_active_chat()
    assert store.get_messages() == []
    assert store.get_token_usage()["total_tokens"] == 0


def test_running_summary_persistence(store):
    assert store.get_running_summary() == ""
    store.set_running_summary("This is a running summary.")
    assert store.get_running_summary() == "This is a running summary."

    # Resets when chat is reset
    store.reset_active_chat()
    assert store.get_running_summary() == ""

