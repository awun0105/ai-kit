"""The kit's agent as a harness driver, and the in-app eval, against the sample notes app."""

from langchain_core.messages import AIMessage

from app.harness_kit_driver import KitDriver
from app.services.guardrails import Guardrails
from app.services.harness import (
    AdapterClient,
    Harness,
    TaskState,
    adapter_handlers,
)
from app.services.harness.evaluation import (
    InAppCase,
    harness_runner,
    run_in_app_eval,
)
from examples.harness_notes import NotesAdapter
from tests.fakes import ScriptedModel
from tests.test_harness_core import _peers


def _write(note_id: str, title: str) -> AIMessage:
    args = {"id": note_id, "title": title, "body": "from the kit agent"}
    return AIMessage(content="", tool_calls=[{"name": "write_note", "args": args, "id": f"call-{note_id}"}])


async def _harness(tmp_path, model: ScriptedModel) -> tuple[Harness, NotesAdapter]:
    notes = NotesAdapter(tmp_path / "data")
    harness_side, app_side = await _peers({}, adapter_handlers(notes))
    notes.peer = app_side  # type: ignore[attr-defined]  # keep the app's end of the pipe alive

    async def emit(event, data):
        pass

    harness = Harness(AdapterClient(harness_side), lambda: KitDriver(model, Guardrails()), emit, review="auto")
    return harness, notes


async def test_kit_driver_edits_through_the_harness(tmp_path):
    model = ScriptedModel(replies=[_write("a", "Alpha"), AIMessage(content="Added note a.")], seen=[], bound=[])
    harness, notes = await _harness(tmp_path, model)
    task_id = await harness.start_task("Add a note called Alpha")

    assert await harness.wait(task_id) == TaskState.accepted
    assert notes.context().state == {"notes": {"a": "Alpha"}}
    assert {"task_context", "validate_draft", "write_note", "publish_note"} <= set(model.bound[0])
    await harness.close()


async def test_repair_turn_keeps_the_conversation(tmp_path):
    replies = [_write("a", ""), AIMessage(content="Done."), _write("a", "Alpha"), AIMessage(content="Fixed.")]
    model = ScriptedModel(replies=replies, seen=[], bound=[])
    harness, notes = await _harness(tmp_path, model)
    task_id = await harness.start_task("Add a note")

    assert await harness.wait(task_id) == TaskState.accepted
    repair_view = [m.content for m in model.seen[-2]]
    assert any("Validation failed" in str(content) for content in repair_view)
    assert any("Add a note" in str(content) for content in repair_view)  # the first turn is still there
    await harness.close()


async def test_in_app_eval_pass_rate(tmp_path):
    replies = [_write("a", "Alpha"), AIMessage(content="ok"), _write("b", "Beta"), AIMessage(content="ok")]
    model = ScriptedModel(replies=replies, seen=[], bound=[])
    harness, _ = await _harness(tmp_path, model)
    case = InAppCase(
        id="add-alpha",
        goal="Add a note Alpha",
        expected_tools=["write_note"],
        expect={"state": {"notes": {"a": "Alpha"}}},
    )

    report = await run_in_app_eval([case], harness_runner(harness), runs=2)
    await harness.close()

    # Run 1 adds a; run 2 adds b, and a is still there, so both contain a. Then check a trajectory miss.
    assert report.items[0].passed == [True, True]
    miss = InAppCase(id="miss", goal="x", expected_tools=["delete_note"])
    assert (await run_in_app_eval([miss], _replay(report.items[0].runs[0]))).pass_rate == 0.0


def _replay(run):
    async def run_case(case):
        return run

    return run_case
