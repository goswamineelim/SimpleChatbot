
from streamlit.testing.v1 import AppTest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

def test_chat_interface_renders():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        "LLM Chat Playground" in title.value
        for title in app.title
    )
    assert len(app.chat_input) == 1
    assert any(
        slider.label == "Temperature"
        for slider in app.slider
    )

def test_new_chat_button_is_available():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        button.label == "New chat"
        for button in app.button
    )

def test_memory_sliders_are_available():
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert any(
        slider.label == "Conversation Token Limit"
        for slider in app.slider
    )
    assert any(
        slider.label == "Summary Token Budget"
        for slider in app.slider
    )

