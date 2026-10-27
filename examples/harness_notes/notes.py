"""A notes app with a full harness adapter: drafts, validation, revisions, undo, working directories, effects.

Data lives in one folder: ``notes.json`` (current revision and notes),
``revisions/<n>.json`` (a snapshot per revision, for drafts and undo) and
``outbox/`` (where ``publish_note``, the dangerous operation, writes).
"""

import json
import re
import uuid
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from app.services.harness import (
    APP_ERROR,
    PROTOCOL_VERSION,
    AppContext,
    AppDescription,
    CommitResult,
    DraftDiff,
    Effect,
    Operation,
    RpcError,
    ValidationIssue,
    ValidationReport,
)

TITLE_MAX = 80
NOTE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _params(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties)}


STR = {"type": "string"}
OPERATIONS = [
    Operation(name="list_notes", description="List the notes (id and title).", level="read"),
    Operation(name="read_note", description="Read one note.", params=_params({"id": STR}), level="read"),
    Operation(
        name="write_note",
        description="Create or replace a note. Ids are lowercase letters, digits, '-' or '_'.",
        params=_params({"id": STR, "title": STR, "body": STR}),
        level="write",
    ),
    Operation(name="delete_note", description="Delete a note.", params=_params({"id": STR}), level="write"),
    Operation(
        name="publish_note",
        description="Publish a note to the outbox (leaves the app; needs the user's approval).",
        params=_params({"id": STR}),
        level="dangerous",
    ),
]

Notes = dict[str, dict[str, str]]


def _refuse(message: str) -> RpcError:
    return RpcError(APP_ERROR, message)


class NotesAdapter:
    """The harness adapter of the notes app."""

    def __init__(self, data_dir: Path):
        """Use (and create if needed) the data folder ``data_dir``."""
        self.data_dir = Path(data_dir)
        (self.data_dir / "revisions").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "outbox").mkdir(exist_ok=True)
        self._drafts: dict[str, dict[str, Any]] = {}
        if not self._store_path.exists():
            self._save(0, {})

    # Storage ------------------------------------------------------------------------------------

    @property
    def _store_path(self) -> Path:
        return self.data_dir / "notes.json"

    def _load(self) -> tuple[int, Notes]:
        store = json.loads(self._store_path.read_text(encoding="utf-8"))
        return int(store["revision"]), store["notes"]

    def _snapshot(self, revision: int) -> Notes:
        path = self.data_dir / "revisions" / f"{revision}.json"
        if not path.exists():
            raise _refuse(f"unknown revision {revision}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, revision: int, notes: Notes) -> None:
        text = json.dumps(notes, ensure_ascii=False, indent=2)
        (self.data_dir / "revisions" / f"{revision}.json").write_text(text, encoding="utf-8")
        store = json.dumps({"revision": revision, "notes": notes}, ensure_ascii=False, indent=2)
        self._store_path.write_text(store, encoding="utf-8")

    def _draft(self, draft_id: Optional[str]) -> dict[str, Any]:
        if not draft_id or draft_id not in self._drafts:
            raise _refuse("unknown draft")
        return self._drafts[draft_id]

    # Adapter ------------------------------------------------------------------------------------

    def describe(self) -> AppDescription:
        """The operations and features of the app (``app.describe``)."""
        return AppDescription(
            protocol_version=PROTOCOL_VERSION, operations=OPERATIONS, supports=["undo", "materialize", "effects"]
        )

    def context(self) -> AppContext:
        """The current revision and a bounded view of the state (``app.context``)."""
        revision, notes = self._load()
        return AppContext(revision=str(revision), state={"notes": {i: n["title"] for i, n in notes.items()}})

    def open_draft(self, base_revision: str) -> str:
        """Open a draft from ``base_revision``; returns its id (``draft.open``)."""
        draft_id = uuid.uuid4().hex[:12]
        self._drafts[draft_id] = {"base": base_revision, "notes": self._snapshot(int(base_revision))}
        return draft_id

    def call(self, draft_id: Optional[str], op: str, args: dict) -> Any:
        """Run one operation, on the draft when ``draft_id`` is given (``op.call``)."""
        notes = self._draft(draft_id)["notes"] if draft_id else self._load()[1]
        if op == "list_notes":
            return [{"id": note_id, "title": note["title"]} for note_id, note in sorted(notes.items())]
        note_id = str(args.get("id", ""))
        if op == "read_note":
            if note_id not in notes:
                raise _refuse(f"no note {note_id}")
            return {"id": note_id, **notes[note_id]}
        if op in ("write_note", "delete_note") and not draft_id:
            raise _refuse("writes need a draft")
        if op == "write_note":
            if not NOTE_ID.match(note_id):
                raise _refuse("ids are lowercase letters, digits, '-' or '_'")
            notes[note_id] = {"title": str(args.get("title", "")), "body": str(args.get("body", ""))}
            return {"ok": True}
        if op == "delete_note":
            if notes.pop(note_id, None) is None:
                raise _refuse(f"no note {note_id}")
            return {"ok": True}
        if op == "publish_note":
            raise _refuse("publish_note runs with effect.run after a commit")
        raise _refuse(f"unknown operation {op}")

    def validate(self, draft_id: str) -> ValidationReport:
        """Check the draft (``draft.validate``)."""
        issues = []
        for note_id, note in sorted(self._draft(draft_id)["notes"].items()):
            title = note["title"].strip()
            if not title:
                issues.append(
                    ValidationIssue(code="title_empty", message=f"note {note_id} has no title", path=note_id)
                )
            elif len(title) > TITLE_MAX:
                issues.append(
                    ValidationIssue(
                        code="title_too_long",
                        message=f"note {note_id}: title over {TITLE_MAX} characters",
                        path=note_id,
                    )
                )
        return ValidationReport(ok=not issues, issues=issues)

    def diff(self, draft_id: str) -> DraftDiff:
        """What the draft changes compared with its base (``draft.diff``)."""
        draft = self._draft(draft_id)
        before, after = self._snapshot(int(draft["base"])), draft["notes"]
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(i for i in set(after) & set(before) if after[i] != before[i])
        summary = f"{len(added)} added, {len(changed)} changed, {len(removed)} removed"
        return DraftDiff(summary=summary, detail={"added": added, "changed": changed, "removed": removed})

    def commit(self, draft_id: str, base_revision: str) -> CommitResult:
        """Make the draft the next revision, or report a conflict (``draft.commit``)."""
        draft = self._draft(draft_id)
        current, _ = self._load()
        if str(current) != base_revision or draft["base"] != base_revision:
            return CommitResult(conflict=True)
        self._save(current + 1, draft["notes"])
        del self._drafts[draft_id]
        return CommitResult(revision=str(current + 1))

    def discard(self, draft_id: str) -> None:
        """Drop the draft (``draft.discard``)."""
        self._drafts.pop(draft_id, None)

    def undo(self, revision: str) -> str:
        """Open a draft that reverses ``revision``, the latest one (``revision.undo``)."""
        current, _ = self._load()
        if str(current) != revision or current == 0:
            raise _refuse("only the latest revision can be undone")
        draft_id = uuid.uuid4().hex[:12]
        self._drafts[draft_id] = {"base": revision, "notes": self._snapshot(current - 1)}
        return draft_id

    def materialize(self, draft_id: str, path: str) -> None:
        """Write the draft as files into ``path`` (``draft.materialize``)."""
        draft = self._draft(draft_id)
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.md"):  # the folder mirrors the draft: a deleted note loses its file
            stale.unlink()
        for note_id, note in draft["notes"].items():
            (folder / f"{note_id}.md").write_text(f"# {note['title']}\n\n{note['body']}\n", encoding="utf-8")
        draft["workdir"] = str(folder)

    def sync(self, draft_id: str) -> ValidationReport:
        """Read the working directory back into the draft (``draft.sync``)."""
        draft = self._draft(draft_id)
        if "workdir" not in draft:
            raise _refuse("the draft has no working directory")
        notes: Notes = {}
        issues = []
        for file in sorted(Path(draft["workdir"]).glob("*.md")):
            text = file.read_text(encoding="utf-8")
            first, _, rest = text.partition("\n")
            if not NOTE_ID.match(file.stem) or not first.startswith("# "):
                issues.append(
                    ValidationIssue(
                        code="bad_file", message=f"{file.name} must start with '# <title>'", path=file.name
                    )
                )
                continue
            notes[file.stem] = {"title": first[2:].strip(), "body": rest.strip("\n")}
        if issues:
            return ValidationReport(ok=False, issues=issues)
        draft["notes"] = notes
        return ValidationReport(ok=True)

    def run_effect(self, revision: str, effect: Effect) -> Any:
        """Run an approved dangerous operation after its revision was committed (``effect.run``)."""
        if effect.op != "publish_note":
            raise _refuse(f"unknown effect {effect.op}")
        notes = self._snapshot(int(revision))
        note_id = str(effect.args.get("id", ""))
        if note_id not in notes:
            raise _refuse(f"no note {note_id} in revision {revision}")
        note = notes[note_id]
        (self.data_dir / "outbox" / f"{note_id}.md").write_text(f"# {note['title']}\n\n{note['body']}\n", "utf-8")
        return {"published": note_id}
