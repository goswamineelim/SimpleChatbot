# LLM Chat Playground

A modular, multi-turn AI chat playground built with **Streamlit** and the **Google GenAI SDK (Gemini)**. The application features persistent multi-chat storage in **SQLite**, streaming responses, real-time token tracking, and an intelligent **summary-based short-term memory system** with configurable token budgets.

---

## Table of Contents
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How Short-Term Memory Works](#how-short-term-memory-works)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running the Test Suite](#running-the-test-suite)

---

## Key Features

- **Multi-Chat Management**: Create new chats, switch between existing conversations, and delete chats with cascading cleanup.
- **Persistent SQLite Storage**: Chat histories, metadata, running summaries, and token counts are stored in a local SQLite database (`chats.db`) that survives app reloads and server restarts.
- **Summary-Based Short-Term Memory**:
  - Automatically condenses older conversation turns when token thresholds are reached.
  - Maintains long-term context without exceeding model token limits or inflating API costs.
  - Keeps the full, verbatim transcript visible in the UI scrollback.
- **Configurable Dashboard Controls**:
  - **Temperature Slider** (model creativity/randomness).
  - **Conversation Token Limit Slider** (active token threshold before condensation).
  - **Summary Token Budget Slider** (target size for condensed background context).
- **Streaming LLM Responses**: Real-time response streaming with live token usage metrics (input, output, and total tokens).
- **Decoupled Clean Architecture**: Domain and state logic are decoupled from Streamlit's `st.session_state`, ensuring easy testing and backend swappability.
- **Comprehensive Test Coverage**: Tested using **Pytest** and Streamlit's **AppTest** following Test-Driven Development (TDD).

---

## System Architecture

The application is structured into decoupled layers following clean architectural principles:

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                   │
│   (app.py, ui/sidebar.py, ui/chat_view.py)             │
└───────────────▲────────────────────────▲───────────────┘
                │                        │
┌───────────────▼──────────────┐ ┌───────▼───────────────┐
│     Memory & Chat Services   │ │     State & Storage   │
│   - MemoryService            │ │   - SqliteChatStore   │
│   - ChatService              │ │   - ChatSession       │
│   - GeminiClient             │ │   - chats.db (SQLite) │
│   - TokenService             │ └───────────────────────┘
└───────────────▲──────────────┘
                │
┌───────────────▼──────────────┐
│     Google GenAI API         │
│   (gemini-3.8-flash)         │
└──────────────────────────────┘
```

### 1. Presentation Layer (`app.py`, `ui/`)
- Orchestrates page rendering and user interactions.
- Renders the full chat history directly from the storage layer.
- Collects user prompts and triggers streaming responses.

### 2. Service Layer (`services/`)
- **`gemini_client.py`**: Wraps the official Google GenAI SDK to handle streaming content generation and token usage normalization.
- **`chat_service.py`**: Intermediary provider interface for streaming assistant responses.
- **`memory_service.py`**: Evaluates conversation token usage and condenses older message turns into a running summary when the conversation crosses configured limits.
- **`token_service.py`**: Normalizes model usage metadata into a unified format.

### 3. State & Storage Layer (`state/`)
- **`sqlite_store.py`**: Manages persistent SQLite interactions (`chats.db`):
  - `chats` table: Stores chat IDs, dynamic titles, running summaries, and token counts.
  - `messages` table: Stores historical messages linked by `chat_id` with foreign key cascades.
  - `app_state` table: Tracks the active chat ID across browser reloads without depending on `st.session_state`.
- **`session.py`**: Provides singleton accessors and backward-compatibility wrappers.

---

## How Short-Term Memory Works

Traditional chat apps often use a simple **sliding window** that drops older messages once a limit is hit, causing the model to completely forget earlier instructions and context. 

This playground uses a **Summary-Based Memory Buffer**:

```
[Full Conversation in SQLite]
  ├── Turn 1: User details requirements
  ├── Turn 2: Assistant outlines plan       ──► Condensed by LLM into a
  ├── Turn 3: User specifies constraint          Running Summary
  │                                              (Budget: ~200 tokens)
  ▼
[Payload Sent to Gemini API]
  1. [Earlier conversation summary: <condensed_summary>]
  2. Assistant: "Understood. I will keep this context in mind."
  3. Turn 4: Recent user prompt
  4. Turn 5: Recent assistant reply
  5. Turn 6: Active user prompt
```

1. **Storage (Long-Term)**: All messages remain saved in SQLite and are fully rendered in the UI.
2. **Context Threshold**: When total prompt tokens exceed the configured **Conversation Token Limit** (e.g. 1,500 tokens), older messages are automatically summarized within the **Summary Token Budget** (e.g. 200 tokens).
3. **Payload Construction**: The prompt sent to Gemini includes the background summary followed by the latest unsummarized messages.

---

## Project Structure

```text
GenAI/
├── app.py                      # Main application entry point & orchestration
├── chats.db                    # Persistent SQLite database (auto-created)
├── config/
│   └── settings.py             # Global constants, model names, and default limits
├── services/
│   ├── chat_service.py         # Response streaming orchestration
│   ├── gemini_client.py        # Google GenAI SDK integration
│   ├── memory_service.py       # Context window & summary-based memory management
│   └── token_service.py        # Token usage normalization utilities
├── state/
│   ├── session.py              # Store accessors & compatibility interface
│   └── sqlite_store.py         # SQLite database schema, CRUD, and migrations
├── ui/
│   ├── chat_view.py            # Chat message renderer
│   ├── sidebar.py              # Sidebar controls (chats, settings, token metrics)
│   └── token_usage.py          # Token metric components
├── tests/
│   ├── test_chat_service.py    # ChatService unit tests
│   ├── test_gemini_client.py   # GeminiClient unit tests
│   ├── test_memory_service.py  # MemoryService & token estimation tests
│   ├── test_session.py         # Session store integration tests
│   ├── test_sqlite_store.py    # SQLite CRUD & multi-chat isolation tests
│   ├── test_token_service.py   # Token normalization tests
│   └── test_ui.py              # Streamlit AppTest interface tests
├── .env                        # Local environment secrets (API keys)
├── .gitignore                  # Git ignore rules (.env, *.db, caches)
└── README.md                   # Project documentation
```

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.10+**
- A **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone and Navigate to the Repository
```bash
cd GenAI
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root with the following variables:

```ini
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Additional configuration constants can be tuned in [config/settings.py](config/settings.py):
- `DEFAULT_TEMPERATURE`: Default sampling temperature (`0.7`).
- `DEFAULT_CONV_TOKEN_LIMIT`: Default trigger threshold for summarization (`1500`).
- `DEFAULT_SUMMARY_BUDGET`: Default target size for summaries (`200`).

---

## Running the Application

To launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Using the Playground:
1. **Send Messages**: Enter prompts in the chat input at the bottom. Responses will stream in real time.
2. **Switch Conversations**: Use the dropdown in the sidebar to jump between different conversations.
3. **Start New Chats**: Click **New chat** to start a blank session.
4. **Tune Parameters**:
   - Adjust the **Temperature** slider to change model randomness.
   - Adjust the **Conversation Token Limit** to control when history gets summarized.
   - Adjust the **Summary Token Budget** to control summary conciseness.
5. **View Active Summary**: Expand the **Active Memory Summary** section in the sidebar to inspect the background context currently supplied to the LLM.

---

## Running the Test Suite

All unit, service, and UI tests are automated via **pytest**:

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v
```

All 29 tests verify:
- Message forwarding and temperature handling.
- Multi-chat isolation and cascading deletion in SQLite.
- Token count estimation and threshold-triggered summarization.
- Streamlit UI rendering and widget availability via `AppTest`.
