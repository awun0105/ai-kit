"""Set up a fresh checkout (or a project generated with copier): env file, secrets, features.

Standard library only, so it runs before `uv sync`:

    python3 scripts/bootstrap.py --features rag,answer-cache
    python3 scripts/bootstrap.py --name "HR Assistant" --slug hr-assistant --features rag,guard-llm-rail

It writes `.env.<env>` from `.env.example` with a random JWT secret, turns the chosen
features on, optionally renames the project, and prints the next commands.
"""

import argparse
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# feature -> (settings to write, uv extra or None)
FEATURES: dict[str, tuple[dict[str, str], str | None]] = {
    "rag": ({"RAG_ENABLED": "true"}, None),
    "answer-cache": ({"RAG_ENABLED": "true", "ANSWER_CACHE_ENABLED": "true"}, None),
    "guard-llm-rail": ({"GUARD_LLM_RAIL_ENABLED": "true"}, None),
    "rerank": ({"RAG_RERANKER": "flashrank"}, "rerank"),
    "mcp": ({"MCP_CONFIG_FILE": "mcp.json"}, None),
    "a2a": ({"A2A_ENABLED": "true", "A2A_SERVER_TOKEN": "<generated>"}, "a2a"),
    "deep-research": ({"DEEP_RESEARCH_ENABLED": "true", "GUARD_APPROVAL_TOOLS": "deep_research"}, "deep"),
}


def _set(text: str, key: str, value: str) -> str:
    """Replace ``KEY=...`` (commented out or not) or append it."""
    line = f"{key}={value}"
    pattern = re.compile(rf"^#?\s*{re.escape(key)}=.*$", re.M)
    return pattern.sub(line, text, count=1) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"


def write_env(env: str, name: str | None, features: list[str], force: bool) -> Path:
    """Create ``.env.<env>`` with secrets and the chosen features."""
    target = ROOT / f".env.{env}"
    if target.exists() and not force:
        print(f"{target.name} exists; leaving it alone (use --force to overwrite)")
        return target
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    text = _set(text, "APP_ENV", env)
    text = _set(text, "JWT_SECRET_KEY", f'"{secrets.token_urlsafe(48)}"')
    if env == "development":
        text = _set(text, "POSTGRES_HOST", "localhost")  # docker compose overrides this to "db" for the app container
    if name:
        text = _set(text, "PROJECT_NAME", f'"{name}"')
    for feature in features:
        for key, value in FEATURES[feature][0].items():
            text = _set(text, key, secrets.token_urlsafe(32) if value == "<generated>" else value)
    target.write_text(text, encoding="utf-8")
    target.chmod(0o600)
    print(f"wrote {target.name} (mode 600) with a random JWT secret; features: {', '.join(features) or 'none'}")
    return target


def rename_project(slug: str, description: str | None) -> None:
    """Rename the Python project in pyproject.toml (run `uv lock` afterwards)."""
    path = ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'^name = ".*"$', f'name = "{slug}"', text, count=1, flags=re.M)
    if description:
        text = re.sub(r'^description = ".*"$', f'description = "{description}"', text, count=1, flags=re.M)
    path.write_text(text, encoding="utf-8")
    print(f'renamed the project to "{slug}" in pyproject.toml')


def main() -> int:
    """Parse arguments and set the project up."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="development", choices=["development", "staging", "production", "test"])
    parser.add_argument("--name", help="Display name (PROJECT_NAME)")
    parser.add_argument("--slug", help="Python project name for pyproject.toml")
    parser.add_argument("--description")
    parser.add_argument("--features", default="", help=f"Comma list from: {', '.join(FEATURES)}")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing env file")
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = sorted(set(features) - set(FEATURES))
    if unknown:
        print(f"unknown features: {', '.join(unknown)}; choose from {', '.join(FEATURES)}", file=sys.stderr)
        return 2

    write_env(args.env, args.name, features, args.force)
    if args.slug:
        rename_project(args.slug, args.description)

    extras = sorted({extra for f in features if (extra := FEATURES[f][1])})
    sync = "uv sync" + "".join(f" --extra {e}" for e in extras)
    steps = [
        sync + ("  # then: uv lock (the project was renamed)" if args.slug else ""),
        f"edit .env.{args.env}: set OPENAI_API_KEY (and EVALUATION_API_KEY for evals)",
        "make test                      # offline test suite",
        "make docker-up                 # API + Postgres/pgvector on http://localhost:8000/docs",
    ]
    if "rag" in features or "answer-cache" in features:
        steps.append("make ingest CORPUS=path/to/docs   # load your documents into the knowledge base")
        steps.append("make eval-rag-offline         # retrieval regression check on the sample golden set")
    if "mcp" in features:
        steps.append('create mcp.json: {"mcpServers": {...}} (see docs/tools.md)')
    print("\nnext steps:\n  " + "\n  ".join(steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
