"""Shared test setup.

Settings are read from the environment at import time, so the variables below
must be set before anything under ``app`` is imported. No test talks to a real
LLM provider or database: LLM calls are scripted and Postgres is replaced by an
in-memory checkpointer.
"""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret-0123456789abcdef0123456789abcdef")
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
os.environ.setdefault("SESSION_NAMING_ENABLED", "false")
