"""Minimal API tests.

Required by the assignment:
  * GET /health              -> 200
  * POST /query (bad input)  -> 422

A third, mocked "happy path" test is included too, since a real end-to-end
test would need a live cloud LLM call. It monkeypatches get_answer() so no
network call or API key is needed to run the tests.
"""

from fastapi.testclient import TestClient

import main as main_module

client = TestClient(main_module.app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_with_missing_question_returns_422():
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_with_blank_question_returns_422():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_happy_path_returns_grounded_answer(monkeypatch):
    """Mocks the RAG pipeline so this test never calls the real cloud LLM."""

    def fake_get_answer(question, chat_history):
        return (
            "The warranty lasts for two years.",
            [{"filename": "manual.pdf", "page": 4}],
        )

    monkeypatch.setattr(main_module, "get_answer", fake_get_answer)

    response = client.post("/query", json={"question": "How long is the warranty?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The warranty lasts for two years."
    assert body["sources"] == [{"filename": "manual.pdf", "page": 4}]
    assert "session_id" in body
    assert "model" in body
