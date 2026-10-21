from unittest.mock import Mock
import pytest

from services.long_term_memory_service import LongTermMemoryService


@pytest.fixture
def ltm_service(tmp_path):
    mock_client = Mock()
    # Use tmp_path so Chroma creates isolated test databases for each test
    service = LongTermMemoryService(
        client=mock_client,
        persist_directory=str(tmp_path / "test_chroma"),
        collection_name="test_memories",
    )
    return service


def test_add_and_recall_memory(ltm_service):
    # Add a memory
    fact_id = ltm_service._add_memory("User prefers Python over JavaScript")
    assert fact_id is not None

    # Recall using semantic similarity
    recalled = ltm_service.recall_memories("What programming language do I like?", top_k=2)
    assert len(recalled) == 1
    assert "User prefers Python over JavaScript" in recalled[0]


def test_duplicate_memory_detection(ltm_service):
    ltm_service._add_memory("User lives in Berlin, Germany")
    
    # Check duplicate detection for the same or almost identical statement
    is_dup = ltm_service._is_duplicate("User lives in Berlin, Germany", threshold=0.35)
    assert is_dup is True

    # Completely different fact should not be flagged as duplicate
    is_diff = ltm_service._is_duplicate("User is allergic to peanuts", threshold=0.35)
    assert is_diff is False


def test_extract_and_store_facts(ltm_service):
    # Mock LLM response for extraction
    ltm_service.client.stream_response.return_value = [
        {"type": "text", "text": '["User works at an AI startup in Seattle", "User has a cat named Luna"]'}
    ]

    new_facts = ltm_service.extract_and_store_facts("I work at an AI startup in Seattle and have a cat named Luna")
    assert len(new_facts) == 2
    assert "User works at an AI startup in Seattle" in new_facts
    assert "User has a cat named Luna" in new_facts

    # Verify they were actually persisted
    all_memories = ltm_service.get_all_memories()
    assert len(all_memories) == 2


def test_chitchat_filter_skips_llm_call(ltm_service):
    new_facts = ltm_service.extract_and_store_facts("hello")
    assert new_facts == []
    ltm_service.client.stream_response.assert_not_called()


def test_delete_and_clear_all(ltm_service):
    id1 = ltm_service._add_memory("Fact 1")
    id2 = ltm_service._add_memory("Fact 2")

    assert len(ltm_service.get_all_memories()) == 2

    # Delete one
    deleted = ltm_service.delete_memory(id1)
    assert deleted is True
    memories = ltm_service.get_all_memories()
    assert len(memories) == 1
    assert memories[0]["id"] == id2

    # Clear all
    ltm_service.clear_all()
    assert len(ltm_service.get_all_memories()) == 0
