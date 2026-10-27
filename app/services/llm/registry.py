"""LLM model registry with pre-initialized instances."""

import json
from typing import (
    Any,
    Dict,
    List,
)

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from app.core.config import settings
from app.core.logging import logger

_API_KEY = SecretStr(settings.OPENAI_API_KEY)

# Every model here is a reasoning model, and the API rejects the classic sampling
# knobs (`top_p`, `presence_penalty`, `frequency_penalty`) with a 400 once
# `reasoning` is set. Tune quality with `reasoning.effort` instead.


def _is_openai_model(model: str) -> bool:
    """Return whether `model` resolves to the OpenAI provider.

    Mirrors `init_chat_model`'s own inference: a bare `gpt-*` name (no
    `provider:` prefix) is OpenAI, same as an explicit `openai:` prefix.
    """
    return model.startswith("openai:") or (":" not in model and model.startswith("gpt-"))


def _build_llm(model: str, **kwargs: Any) -> BaseChatModel:
    """Construct a chat model via `init_chat_model`.

    `model` accepts either a bare name that `init_chat_model` infers a
    provider for (e.g. "gpt-5.6-luna" -> OpenAI, keeping existing `gpt-*`
    registry entries working unchanged) or an explicit "provider:model"
    string for any other provider (e.g. "anthropic:claude-opus-4-6",
    "ollama:llama3"). Non-OpenAI providers need their LangChain integration
    package installed (see the `anthropic` / `ollama` extras in
    pyproject.toml) and read their API key from the provider's own env var
    (e.g. `ANTHROPIC_API_KEY`). An OpenAI-compatible third party (an
    `openai:` model with a custom `base_url`) must pass its own `api_key`.
    """
    if _is_openai_model(model):
        # Without its own key, ChatOpenAI would send OPENAI_API_KEY to the custom endpoint.
        if kwargs.keys() & {"base_url", "openai_api_base"} and not kwargs.keys() & {"api_key", "openai_api_key"}:
            raise ValueError(f"'{model}' sets a custom base_url but no api_key; pass that endpoint's own api_key")
        kwargs.setdefault("api_key", _API_KEY)
        # ResilientInvoker is the only retry layer; SDK retries would multiply attempts
        # (3 app retries x 3 SDK attempts x N models is how a retry storm starts).
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("timeout", settings.LLM_ATTEMPT_TIMEOUT)
    return init_chat_model(model, **kwargs)


def parse_model_configs(raw: str) -> List[Dict[str, Any]]:
    """Parse ``LLM_MODELS_JSON``: a non-empty JSON list of ``{"name", "model", "kwargs"?, "supports_tools"?}``.

    Example, a local model through Ollama or llama.cpp's OpenAI-compatible server:
    ``[{"name": "local", "model": "ollama:qwen3:8b"}]`` or
    ``[{"name": "local", "model": "openai:qwen3", "kwargs": {"base_url": "http://127.0.0.1:8080/v1", "api_key": "none"}}]``.
    """
    configs = json.loads(raw)
    if not isinstance(configs, list) or not configs:
        raise ValueError("LLM_MODELS_JSON must be a non-empty JSON list")
    names: set[str] = set()
    for config in configs:
        if not isinstance(config, dict) or not config.get("name") or not config.get("model"):
            raise ValueError("every LLM_MODELS_JSON entry needs a name and a model")
        if config["name"] in names:
            raise ValueError(f"LLM_MODELS_JSON lists {config['name']!r} twice")
        names.add(config["name"])
        if not isinstance(config.get("kwargs", {}), dict):
            raise ValueError(f"kwargs of {config['name']!r} must be an object")
    return configs


class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances.

    This class maintains a list of LLM configurations and provides
    methods to retrieve them by name with optional argument overrides.
    """

    # Ordered by preference: index 0 is the default and the head of the circular
    # fallback chain, so it degrades newest -> cheapest.
    #
    # `model` is passed straight to `init_chat_model`, so an entry for another
    # provider only needs a "provider:model" string here, e.g.
    # {"name": "claude-opus", "model": "anthropic:claude-opus-4-6", "kwargs": {...}}.
    LLM_CONFIGS: List[Dict[str, Any]] = [
        {
            "name": "gpt-5.6-luna",
            "model": "gpt-5.6-luna",
            "kwargs": {
                "max_completion_tokens": settings.MAX_TOKENS,
                "reasoning": {"effort": "medium"},
            },
        },
        {
            "name": "gpt-5.4",
            "model": "gpt-5.4",
            "kwargs": {
                "max_completion_tokens": settings.MAX_TOKENS,
                "reasoning": {"effort": "medium"},
            },
        },
        {
            "name": "gpt-5.4-mini",
            "model": "gpt-5.4-mini",
            "kwargs": {
                "max_completion_tokens": settings.MAX_TOKENS,
                "reasoning": {"effort": "low"},
            },
        },
        {
            "name": "gpt-5.4-nano",
            "model": "gpt-5.4-nano",
            "kwargs": {
                "max_completion_tokens": settings.MAX_TOKENS,
                "reasoning": {"effort": "low"},
            },
        },
    ]

    if settings.LLM_MODELS_JSON.strip():
        LLM_CONFIGS = parse_model_configs(settings.LLM_MODELS_JSON)

    LLMS: List[Dict[str, Any]] = [
        {"name": config["name"], "llm": _build_llm(config["model"], **config.get("kwargs", {}))}
        for config in LLM_CONFIGS
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name with optional argument overrides.

        When kwargs are provided a fresh model instance is returned with
        those overrides applied, leaving the shared registry entry untouched.

        Args:
            model_name: Name of the model to retrieve.
            **kwargs: Optional arguments to override default model configuration.

        Returns:
            BaseChatModel instance.

        Raises:
            ValueError: If model_name is not found in LLM_CONFIGS.
        """
        config = next((c for c in cls.LLM_CONFIGS if c["name"] == model_name), None)

        if not config:
            available = ", ".join(c["name"] for c in cls.LLM_CONFIGS)
            raise ValueError(f"model '{model_name}' not found in registry. available models: {available}")

        if kwargs:
            # Take the model id from the config rather than reusing the registry
            # name, so a name that ever diverges from its model can't send an
            # unknown id to the API.
            logger.debug(
                "creating_llm_with_custom_args",
                model_name=model_name,
                model=config["model"],
                custom_args=list(kwargs.keys()),
            )
            override_kwargs: Dict[str, Any] = dict(kwargs)
            if _is_openai_model(config["model"]):
                # ponytail: carries the token limit but not per-entry `reasoning`;
                # add that here if a caller ever needs to override a reasoning model.
                override_kwargs.setdefault("max_completion_tokens", settings.MAX_TOKENS)
            return _build_llm(config["model"], **override_kwargs)

        logger.debug("using_default_llm_instance", model_name=model_name)
        return next(entry["llm"] for entry in cls.LLMS if entry["name"] == model_name)

    @classmethod
    def supports_tools(cls, model_name: str) -> bool:
        """Whether a registry entry can do tool calling (``"supports_tools": False`` opts out).

        The fallback ladder skips such models for tool-bound agent calls instead
        of sending them a request they cannot serve.
        """
        config = next((c for c in cls.LLM_CONFIGS if c["name"] == model_name), {})
        return bool(config.get("supports_tools", True))

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Return all registered model names in order.

        Returns:
            List of model name strings.
        """
        return [config["name"] for config in cls.LLM_CONFIGS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:
        """Return the model entry at a specific index, wrapping to 0 if out of range.

        Args:
            index: Index into LLMS.

        Returns:
            Model entry dict.
        """
        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]
        return cls.LLMS[0]
