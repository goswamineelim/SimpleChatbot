import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import chromadb

DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_db"

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant.
Analyze the user's message and determine if it reveals any persistent, long-term personal facts, preferences, constraints, background info, or operating instructions about the user that would be useful to remember in future conversations.

Examples of memorable facts:
- "I'm a Python developer with 5 years experience" -> "User is a Python developer with 5 years of experience"
- "I live in Berlin and I'm moving to Tokyo in May" -> ["User lives in Berlin", "User is moving to Tokyo in May"]
- "Don't use semicolons in your code examples" -> "User prefers code examples without semicolons"
- "My cat is named Mochi" -> "User has a cat named Mochi"

Examples of things NOT to remember (transient chatter, questions, greetings, generic requests):
- "Can you write an essay about climate change?" -> []
- "Hello, how are you?" -> []
- "Thanks, that was helpful!" -> []
- "What is the capital of France?" -> []

OUTPUT INSTRUCTIONS:
Return ONLY a valid JSON array of strings containing the atomic facts. If there are no long-term facts to store, return an empty array [].
Do NOT wrap in markdown code fences. Output raw JSON only.
"""


class LongTermMemoryService:
    """Manages long-term semantic memory storage, retrieval, and extraction using ChromaDB."""

    def __init__(
        self,
        client: Optional[Any] = None,
        persist_directory: Optional[str] = None,
        collection_name: str = "user_memories",
    ):
        self.client = client
        self.persist_directory = str(persist_directory or DEFAULT_CHROMA_DIR)
        
        # Initialize persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Long-term user memories and facts"}
        )

    def recall_memories(
        self,
        query: str,
        top_k: int = 5,
        max_distance: float = 3.0,
    ) -> List[str]:
        """
        Retrieve semantically relevant memories for the given query using vector similarity.
        """
        if not query or self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
            )

            memories = []
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, dist in zip(documents, distances):
                if dist <= max_distance:
                    memories.append(doc)

            return memories
        except Exception:
            return []

    def extract_and_store_facts(
        self,
        user_message: str,
        duplicate_distance_threshold: float = 0.35,
    ) -> List[str]:
        """
        Extract facts from user message and store non-duplicate facts into ChromaDB.
        Returns the list of newly stored facts.
        """
        if not self.client or not user_message or len(user_message.strip()) < 3:
            return []

        # Heuristic pre-filter to avoid unnecessary LLM calls for common chitchat
        cleaned = user_message.strip().lower()
        if cleaned in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "good morning", "good evening"}:
            return []

        extracted_facts = self._extract_facts_with_llm(user_message)
        saved_facts = []

        for fact in extracted_facts:
            fact = fact.strip()
            if not fact:
                continue

            if not self._is_duplicate(fact, threshold=duplicate_distance_threshold):
                self._add_memory(fact)
                saved_facts.append(fact)

        return saved_facts

    def _extract_facts_with_llm(self, user_message: str) -> List[str]:
        """Use Gemini to extract atomic facts as a JSON array."""
        prompt = [
            {
                "role": "user",
                "content": f"{EXTRACTION_SYSTEM_PROMPT}\n\nUser message to evaluate:\n\"{user_message}\"",
            }
        ]

        response_text = ""
        try:
            for event in self.client.stream_response(prompt, temperature=0.1):
                if event.get("type") == "text":
                    response_text += event.get("text", "")

            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Strip markdown code blocks if the model returned them
                lines = response_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
            elif isinstance(parsed, str) and parsed:
                return [parsed]
            return []
        except Exception:
            return []

    def _is_duplicate(self, fact: str, threshold: float = 0.35) -> bool:
        """Check if an existing memory is nearly identical in meaning."""
        if self.collection.count() == 0:
            return False

        try:
            results = self.collection.query(
                query_texts=[fact],
                n_results=1,
            )
            distances = results.get("distances", [[]])[0]
            if distances and distances[0] <= threshold:
                return True
        except Exception:
            pass
        return False

    def _add_memory(self, fact: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a single memory fact."""
        fact_id = str(uuid.uuid4())[:8]
        meta = metadata or {}
        meta.setdefault("created_at", datetime.utcnow().isoformat())
        self.collection.add(
            documents=[fact],
            ids=[fact_id],
            metadatas=[meta],
        )
        return fact_id

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all stored memories for the UI manager."""
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.get()
            memories = []
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", []) or [{}] * len(ids)

            for mem_id, doc, meta in zip(ids, documents, metadatas):
                memories.append({
                    "id": mem_id,
                    "text": doc,
                    "created_at": (meta or {}).get("created_at", ""),
                })
            return memories
        except Exception:
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its ID."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def clear_all(self) -> bool:
        """Clear all stored memories."""
        try:
            all_ids = self.collection.get().get("ids", [])
            if all_ids:
                self.collection.delete(ids=all_ids)
            return True
        except Exception:
            return False
