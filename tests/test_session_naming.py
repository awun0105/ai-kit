"""Session auto-naming: the title request must be one a reasoning model accepts."""

from app.schemas.chat import SessionTitle
from app.services.database import database_service
from app.services.llm import LLMRegistry
from app.services.session_naming import _persist_session_name


async def test_title_request_sends_no_sampling_knobs_to_a_reasoning_model(monkeypatch):
    # gpt-5.x reasoning models reject temperature with a 400, which left every session on its placeholder.
    seen: dict = {}

    class TitleModel:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages, config=None):
            return SessionTitle(title="Leave policy")

    def fake_get(name, **kwargs):
        seen.update(kwargs, name=name)
        return TitleModel()

    saved = []

    async def update_session_name(session_id, name):
        saved.append((session_id, name))

    monkeypatch.setattr(LLMRegistry, "get", fake_get)
    monkeypatch.setattr(database_service, "update_session_name", update_session_name)

    await _persist_session_name("s1", "How many leave days do I get?")

    assert saved == [("s1", "Leave policy")]
    assert "temperature" not in seen and "max_tokens" not in seen
    assert seen["reasoning"] == {"effort": "low"}
