"""Application configuration management.

This module handles environment-specific configuration loading, parsing, and management
for the application. It includes environment detection, .env file loading, and
configuration value parsing.
"""

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv


WEAK_JWT_SECRET_KEYS = {
    "",
    "changeme",
    "change_me",
    "change_me_to_a_random_32_plus_char_string",
    "supersecretkeythatshouldbechangedforproduction",
    "your-jwt-secret-key",
}


# Define environment types
class Environment(str, Enum):
    """Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production, and test.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# Determine environment
def get_environment() -> Environment:
    """Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


# Load appropriate .env file based on environment
def load_env_file():
    """Load environment-specific .env file."""
    env = get_environment()
    print(f"Loading environment: {env}")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define env files in priority order
    env_files = [
        os.path.join(base_dir, f".env.{env.value}.local"),
        os.path.join(base_dir, f".env.{env.value}"),
        os.path.join(base_dir, ".env.local"),
        os.path.join(base_dir, ".env"),
    ]

    # Load the first env file that exists
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(dotenv_path=env_file)
            print(f"Loaded environment from {env_file}")
            return env_file

    # Fall back to default if no env file found
    return None


ENV_FILE = load_env_file()


# Parse list values from environment variables
def parse_list_from_env(env_key, default=None):
    """Parse a comma-separated list from an environment variable."""
    value = os.getenv(env_key)
    if not value:
        return default or []

    # Remove quotes if they exist
    value = value.strip("\"'")
    # Handle single value case
    if "," not in value:
        return [value]
    # Split comma-separated values
    return [item.strip() for item in value.split(",") if item.strip()]


# Parse dict of lists from environment variables with prefix
def parse_dict_of_lists_from_env(prefix, default_dict=None):
    """Parse dictionary of lists from environment variables with a common prefix."""
    result = default_dict or {}

    # Look for all env vars with the given prefix
    for key, value in os.environ.items():
        if key.startswith(prefix):
            endpoint = key[len(prefix) :].lower()  # Extract endpoint name
            # Parse the values for this endpoint
            if value:
                value = value.strip("\"'")
                if "," in value:
                    result[endpoint] = [item.strip() for item in value.split(",") if item.strip()]
                else:
                    result[endpoint] = [value]

    return result


class Settings:
    """Application settings without using pydantic."""

    def __init__(self):
        """Initialize application settings from environment variables.

        Loads and sets all configuration values from environment variables,
        with appropriate defaults for each setting. Also applies
        environment-specific overrides based on the current environment.
        """
        # Set the environment
        self.ENVIRONMENT = get_environment()

        # Application Settings
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "FastAPI LangGraph Template")
        self.VERSION = os.getenv("VERSION", "1.0.0")
        self.DESCRIPTION = os.getenv(
            "DESCRIPTION", "A production-ready FastAPI template with LangGraph and Langfuse integration"
        )
        self.API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
        self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "yes")

        # CORS Settings
        self.ALLOWED_ORIGINS = parse_list_from_env("ALLOWED_ORIGINS", ["*"])

        # Langfuse Configuration
        self.LANGFUSE_TRACING_ENABLED = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in (
            "true",
            "1",
            "t",
            "yes",
        )
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # LangGraph Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-5.6-luna")
        self.SESSION_NAMING_ENABLED = os.getenv("SESSION_NAMING_ENABLED", "true").lower() == "true"
        self.DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.2"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
        self.MAX_LLM_CALL_RETRIES = int(os.getenv("MAX_LLM_CALL_RETRIES", "3"))
        self.LLM_TOTAL_TIMEOUT = int(os.getenv("LLM_TOTAL_TIMEOUT", "60"))
        # Reliability (see docs/reliability.md). MAX_LLM_CALL_RETRIES is the attempt count per model.
        self.LLM_ATTEMPT_TIMEOUT = float(os.getenv("LLM_ATTEMPT_TIMEOUT", "30"))
        self.LLM_RETRY_BASE_S = float(os.getenv("LLM_RETRY_BASE_S", "0.5"))
        self.LLM_RETRY_CAP_S = float(os.getenv("LLM_RETRY_CAP_S", "8"))
        self.LLM_RETRY_BUDGET_RATIO = float(os.getenv("LLM_RETRY_BUDGET_RATIO", "0.1"))
        self.LLM_MAX_RETRY_AFTER_S = float(os.getenv("LLM_MAX_RETRY_AFTER_S", "10"))
        self.LLM_QUOTA_COOLDOWN_S = float(os.getenv("LLM_QUOTA_COOLDOWN_S", "300"))
        self.LLM_BREAKER_WINDOW = int(os.getenv("LLM_BREAKER_WINDOW", "20"))
        self.LLM_BREAKER_MIN_CALLS = int(os.getenv("LLM_BREAKER_MIN_CALLS", "10"))
        self.LLM_BREAKER_FAILURE_RATE = float(os.getenv("LLM_BREAKER_FAILURE_RATE", "0.5"))
        self.LLM_BREAKER_SLOW_CALL_S = float(os.getenv("LLM_BREAKER_SLOW_CALL_S", "20"))
        self.LLM_BREAKER_OPEN_S = float(os.getenv("LLM_BREAKER_OPEN_S", "30"))
        self.LLM_BREAKER_HALF_OPEN_PROBES = int(os.getenv("LLM_BREAKER_HALF_OPEN_PROBES", "2"))

        # Long term memory Configuration
        self.LONG_TERM_MEMORY_MODEL = os.getenv("LONG_TERM_MEMORY_MODEL", "gpt-5-nano")
        self.LONG_TERM_MEMORY_EMBEDDER_MODEL = os.getenv("LONG_TERM_MEMORY_EMBEDDER_MODEL", "text-embedding-3-small")
        self.LONG_TERM_MEMORY_COLLECTION_NAME = os.getenv("LONG_TERM_MEMORY_COLLECTION_NAME", "longterm_memory")
        # RAG Configuration (the knowledge_search tool is registered only when enabled)
        self.RAG_ENABLED = os.getenv("RAG_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        self.RAG_VECTOR_STORE = os.getenv("RAG_VECTOR_STORE", "pgvector")  # "pgvector" or "memory"
        self.RAG_TABLE = os.getenv("RAG_TABLE", "rag_chunks")
        self.RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "openai")  # "openai" or "hashing"
        self.RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.RAG_EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1536"))
        self.RAG_EMBEDDING_BASE_URL = os.getenv("RAG_EMBEDDING_BASE_URL", "")
        self.RAG_EMBEDDING_API_KEY = os.getenv("RAG_EMBEDDING_API_KEY", self.OPENAI_API_KEY)
        self.RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
        self.RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
        self.RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
        self.RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))
        self.RAG_LEXICAL_TOKENIZER = os.getenv("RAG_LEXICAL_TOKENIZER", "auto")  # "auto", "vi" or "simple"
        # Empty disables reranking. Otherwise a rerankers model name, e.g. "flashrank" or "BAAI/bge-reranker-v2-m3"
        self.RAG_RERANKER = os.getenv("RAG_RERANKER", "")
        self.RAG_RERANKER_TYPE = os.getenv(
            "RAG_RERANKER_TYPE", ""
        )  # e.g. "cross-encoder"; empty lets rerankers infer it
        self.RAG_RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))

        # MCP (see docs/tools.md). A JSON file in the standard {"mcpServers": {...}} format.
        self.MCP_CONFIG_FILE = os.getenv("MCP_CONFIG_FILE", "")
        self.MCP_TOOL_ALLOWLIST = parse_list_from_env("MCP_TOOL_ALLOWLIST")  # empty exposes every discovered tool
        self.MCP_SERVER_TOKEN = os.getenv("MCP_SERVER_TOKEN", "")  # bearer token for the kit's own HTTP MCP server

        # A2A (see docs/tools.md); needs `uv sync --extra a2a`
        self.A2A_ENABLED = os.getenv("A2A_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        self.A2A_SERVER_TOKEN = os.getenv("A2A_SERVER_TOKEN", "")
        self.A2A_PUBLIC_URL = os.getenv("A2A_PUBLIC_URL", "http://localhost:8000/a2a")
        self.A2A_AGENT_DESCRIPTION = os.getenv(
            "A2A_AGENT_DESCRIPTION", "Answers questions using the organisation's knowledge base and tools."
        )
        # Remote agents the model may delegate to, as name=url pairs; tokens in A2A_REMOTE_TOKEN_<NAME>
        self.A2A_REMOTE_AGENTS = parse_list_from_env("A2A_REMOTE_AGENTS")

        # Deep research team (see docs/agents.md); needs `uv sync --extra deep`
        self.DEEP_RESEARCH_ENABLED = os.getenv("DEEP_RESEARCH_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        self.DEEP_RESEARCH_MAX_MODEL_CALLS = int(os.getenv("DEEP_RESEARCH_MAX_MODEL_CALLS", "40"))
        self.DEEP_RESEARCH_MAX_TOOL_CALLS = int(os.getenv("DEEP_RESEARCH_MAX_TOOL_CALLS", "80"))

        # Cost guard (see docs/deployment.md): monthly USD budget per user; 0 disables it.
        self.COST_MONTHLY_BUDGET_USD = float(os.getenv("COST_MONTHLY_BUDGET_USD", "0"))
        # {"model name": [input USD per 1M tokens, output USD per 1M tokens]}, from your provider's price list
        self.COST_PRICES_JSON = os.getenv("COST_PRICES_JSON", "")
        self.COST_DEFAULT_INPUT_PER_1M = float(os.getenv("COST_DEFAULT_INPUT_PER_1M", "0"))
        self.COST_DEFAULT_OUTPUT_PER_1M = float(os.getenv("COST_DEFAULT_OUTPUT_PER_1M", "0"))

        # Answer cache (see docs/reliability.md); needs RAG_ENABLED because answers are keyed to their sources
        self.ANSWER_CACHE_ENABLED = os.getenv("ANSWER_CACHE_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        self.ANSWER_CACHE_STORE = os.getenv("ANSWER_CACHE_STORE", "pgvector")  # "pgvector" or "memory"
        self.ANSWER_CACHE_TABLE = os.getenv("ANSWER_CACHE_TABLE", "rag_answer_cache")
        self.ANSWER_CACHE_THRESHOLD = float(os.getenv("ANSWER_CACHE_THRESHOLD", "0.92"))
        self.ANSWER_CACHE_TTL_S = float(os.getenv("ANSWER_CACHE_TTL_S", "3600"))
        self.ANSWER_CACHE_MAX_STALE_S = float(os.getenv("ANSWER_CACHE_MAX_STALE_S", "86400"))
        self.ANSWER_CACHE_VERSION = os.getenv("ANSWER_CACHE_VERSION", "1")  # bump to discard every cached answer

        # Guardrails Configuration (deterministic rails are on by default; see docs/guardrails.md)
        self.GUARD_ENABLED = os.getenv("GUARD_ENABLED", "true").lower() in ("true", "1", "t", "yes")
        self.GUARD_BLOCK_INJECTION = os.getenv("GUARD_BLOCK_INJECTION", "true").lower() in ("true", "1", "t", "yes")
        self.GUARD_INPUT_PII_ACTION = os.getenv("GUARD_INPUT_PII_ACTION", "redact")  # "redact", "block" or "allow"
        self.GUARD_OUTPUT_REDACTION = os.getenv("GUARD_OUTPUT_REDACTION", "true").lower() in ("true", "1", "t", "yes")
        # Values or "@domain" suffixes never redacted, e.g. a public support address
        self.GUARD_PII_ALLOWLIST = parse_list_from_env("GUARD_PII_ALLOWLIST")
        # Application secrets that must never reach the model or the user, e.g. an internal hostname
        self.GUARD_DENYLIST = parse_list_from_env("GUARD_DENYLIST")
        # Tools that pause for human approval before running
        self.GUARD_APPROVAL_TOOLS = parse_list_from_env("GUARD_APPROVAL_TOOLS")
        # When set, tool arguments may only contain https URLs on these hosts (and their subdomains)
        self.GUARD_EGRESS_ALLOWLIST = parse_list_from_env("GUARD_EGRESS_ALLOWLIST")
        self.GUARD_SCREEN_TOOL_OUTPUT = os.getenv("GUARD_SCREEN_TOOL_OUTPUT", "true").lower() in (
            "true",
            "1",
            "t",
            "yes",
        )
        self.GUARD_LLM_RAIL_ENABLED = os.getenv("GUARD_LLM_RAIL_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        self.GUARD_LLM_RAIL_POLICY = os.getenv(
            "GUARD_LLM_RAIL_POLICY", "A general-purpose assistant. Any lawful topic is in scope."
        )
        self.GUARD_LLM_RAIL_MODEL = os.getenv("GUARD_LLM_RAIL_MODEL", "")  # empty uses the default model
        self.GUARD_LLM_RAIL_FAIL_MODE = os.getenv("GUARD_LLM_RAIL_FAIL_MODE", "open")  # "open" or "closed"
        self.GUARD_REFUSAL_MESSAGE = os.getenv("GUARD_REFUSAL_MESSAGE", "Sorry, I can't help with that request.")

        # JWT Configuration
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_DAYS", "30"))
        self.validate_jwt_secret_key()

        # Logging Configuration
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"

        # Profiling Configuration (DEBUG only)
        self.PROFILING_DIR = Path(os.getenv("PROFILING_DIR", "/tmp/fastapi_profiles"))
        self.PROFILING_THRESHOLD_SECONDS = float(os.getenv("PROFILING_THRESHOLD_SECONDS", "2.0"))

        # Postgres Configuration
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "food_order_db")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
        self.POSTGRES_MAX_OVERFLOW = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
        self.CHECKPOINT_TABLES = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]

        # Valkey/Redis Cache Configuration (optional — if host is set, caching is enabled)
        self.VALKEY_HOST = os.getenv("VALKEY_HOST", "")
        self.VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))
        self.VALKEY_DB = int(os.getenv("VALKEY_DB", "0"))
        self.VALKEY_PASSWORD = os.getenv("VALKEY_PASSWORD", "")
        self.VALKEY_MAX_CONNECTIONS = int(os.getenv("VALKEY_MAX_CONNECTIONS", "20"))
        self.CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))

        # Rate Limiting Configuration
        self.RATE_LIMIT_DEFAULT = parse_list_from_env("RATE_LIMIT_DEFAULT", ["200 per day", "50 per hour"])

        # Rate limit endpoints defaults
        default_endpoints = {
            "chat": ["30 per minute"],
            "chat_stream": ["20 per minute"],
            "messages": ["50 per minute"],
            "register": ["10 per hour"],
            "login": ["20 per minute"],
            "root": ["10 per minute"],
            "health": ["20 per minute"],
        }

        # Update rate limit endpoints from environment variables
        self.RATE_LIMIT_ENDPOINTS = default_endpoints.copy()
        for endpoint in default_endpoints:
            env_key = f"RATE_LIMIT_{endpoint.upper()}"
            value = parse_list_from_env(env_key)
            if value:
                self.RATE_LIMIT_ENDPOINTS[endpoint] = value

        # Evaluation Configuration
        self.EVALUATION_LLM = os.getenv("EVALUATION_LLM", "gpt-5")
        self.EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", "https://api.openai.com/v1")
        self.EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY", self.OPENAI_API_KEY)
        self.EVALUATION_SLEEP_TIME = int(os.getenv("EVALUATION_SLEEP_TIME", "10"))

        # Apply environment-specific settings
        self.apply_environment_settings()

    def apply_environment_settings(self):
        """Apply environment-specific settings based on the current environment."""
        env_settings = {
            Environment.DEVELOPMENT: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "200 per hour"],
            },
            Environment.STAGING: {
                "DEBUG": False,
                "LOG_LEVEL": "INFO",
                "RATE_LIMIT_DEFAULT": ["500 per day", "100 per hour"],
            },
            Environment.PRODUCTION: {
                "DEBUG": False,
                "LOG_LEVEL": "WARNING",
                "RATE_LIMIT_DEFAULT": ["200 per day", "50 per hour"],
            },
            Environment.TEST: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "1000 per hour"],  # Relaxed for testing
            },
        }

        # Get settings for current environment
        current_env_settings = env_settings.get(self.ENVIRONMENT, {})

        # Apply settings if not explicitly set in environment variables
        for key, value in current_env_settings.items():
            env_var_name = key.upper()
            # Only override if environment variable wasn't explicitly set
            if env_var_name not in os.environ:
                setattr(self, key, value)

    def validate_jwt_secret_key(self):
        """Reject empty, short, or placeholder JWT secrets outside test runs."""
        if self.ENVIRONMENT == Environment.TEST:
            return

        secret = self.JWT_SECRET_KEY.strip()
        if len(secret) < 32:
            raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters long")

        if secret.lower() in WEAK_JWT_SECRET_KEYS:
            raise RuntimeError("JWT_SECRET_KEY must be a strong random value, not a placeholder or default")


# Create settings instance
settings = Settings()
