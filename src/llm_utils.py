from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional, Sequence

from openai import AzureOpenAI, OpenAI

AZURE_OPENAI_PROVIDER = "azure_openai"
OPENAI_PROVIDER = "openai"
DEEPSEEK_PROVIDER = "deepseek"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
ZAI_PROVIDER = "zai"
ZAI_DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"
MODEL_MAX_RETRIES = 5
MODEL_RETRY_DELAY_SECONDS = 10


def normalize_llm_provider(provider: Optional[str]) -> str:
    value = (provider or os.environ.get("LLM_PROVIDER") or AZURE_OPENAI_PROVIDER).strip().lower()
    aliases = {
        "azure": AZURE_OPENAI_PROVIDER,
        "azure_openai": AZURE_OPENAI_PROVIDER,
        "aoai": AZURE_OPENAI_PROVIDER,
        "openai": OPENAI_PROVIDER,
        "oai": OPENAI_PROVIDER,
        "deepseek": DEEPSEEK_PROVIDER,
        "ds": DEEPSEEK_PROVIDER,
        "zai": ZAI_PROVIDER,
        "z.ai": ZAI_PROVIDER,
        "glm": ZAI_PROVIDER,
        "zhipu": ZAI_PROVIDER,
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Expected one of: {AZURE_OPENAI_PROVIDER}, {OPENAI_PROVIDER}, "
            f"{DEEPSEEK_PROVIDER}, {ZAI_PROVIDER}."
        )
    return normalized


def resolve_api_version(api_version: Optional[str] = None) -> str:
    return api_version or os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)


def resolve_api_key(provider: Optional[str] = None, api_key: Optional[str] = None) -> Optional[str]:
    normalized = normalize_llm_provider(provider)
    if api_key:
        return api_key
    if normalized == AZURE_OPENAI_PROVIDER:
        return os.environ.get("AZURE_OPENAI_API_KEY")
    if normalized == DEEPSEEK_PROVIDER:
        return os.environ.get("DEEPSEEK_API_KEY")
    if normalized == ZAI_PROVIDER:
        return os.environ.get("ZAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def resolve_base_url(provider: Optional[str] = None, base_url: Optional[str] = None) -> Optional[str]:
    normalized = normalize_llm_provider(provider)
    if base_url:
        return base_url
    if normalized == AZURE_OPENAI_PROVIDER:
        return os.environ.get("AZURE_OPENAI_ENDPOINT")
    if normalized == DEEPSEEK_PROVIDER:
        return os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL)
    if normalized == ZAI_PROVIDER:
        return os.environ.get("ZAI_BASE_URL", ZAI_DEFAULT_BASE_URL)
    return os.environ.get("OPENAI_BASE_URL")


def create_chat_client(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    base_url: Optional[str] = None,
    azure_ad_token_provider: Any = None,
) -> Any:
    normalized = normalize_llm_provider(provider)
    if normalized == AZURE_OPENAI_PROVIDER:
        endpoint = resolve_base_url(normalized, azure_endpoint)
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT for Azure OpenAI client creation.")
        kwargs: Dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": resolve_api_version(api_version),
            "max_retries": 0,
        }
        if azure_ad_token_provider is not None:
            kwargs["azure_ad_token_provider"] = azure_ad_token_provider
        else:
            key = resolve_api_key(normalized, api_key)
            if not key:
                raise ValueError("Missing AZURE_OPENAI_API_KEY for Azure OpenAI client creation.")
            kwargs["api_key"] = key
        return AzureOpenAI(**kwargs)

    key = resolve_api_key(normalized, api_key)
    if not key:
        key_names = {
            DEEPSEEK_PROVIDER: "DEEPSEEK_API_KEY",
            ZAI_PROVIDER: "ZAI_API_KEY",
        }
        key_name = key_names.get(normalized, "OPENAI_API_KEY")
        raise ValueError(f"Missing {key_name} for {normalized} client creation.")
    kwargs = {"api_key": key, "max_retries": 0}
    resolved_base_url = resolve_base_url(normalized, base_url)
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def with_retry(
    func: Any,
    *args: Any,
    retries: Optional[int] = None,
    delay_seconds: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    resolved_retries = retries or int(os.environ.get("MODEL_MAX_RETRIES", MODEL_MAX_RETRIES))
    resolved_delay = delay_seconds or int(
        os.environ.get("MODEL_RETRY_DELAY_SECONDS", MODEL_RETRY_DELAY_SECONDS)
    )
    max_delay = int(os.environ.get("MODEL_RETRY_MAX_DELAY_SECONDS", "120"))
    last_error: Exception | None = None
    for attempt in range(1, resolved_retries + 1):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == resolved_retries:
                break
            backoff = min(resolved_delay * (2 ** (attempt - 1)), max_delay)
            time.sleep(backoff * random.uniform(0.8, 1.2))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry wrapper exhausted without capturing an exception.")


def model_uses_max_completion_tokens(model: str) -> bool:
    name = (model or "").strip().lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def is_deepseek_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("deepseek-")


def deepseek_thinking_enabled() -> bool:
    value = os.environ.get("DEEPSEEK_THINKING", "enabled").strip().lower()
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError("DEEPSEEK_THINKING must be enabled or disabled.")


def deepseek_reasoning_effort() -> str:
    value = os.environ.get("DEEPSEEK_REASONING_EFFORT", "max").strip().lower()
    aliases = {"xhigh": "max", "high": "high", "max": "max"}
    if value not in aliases:
        raise ValueError("DEEPSEEK_REASONING_EFFORT must be high or max.")
    return aliases[value]


def is_glm_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("glm-")


def zai_thinking_enabled() -> bool:
    value = os.environ.get("ZAI_THINKING", "enabled").strip().lower()
    if value in {"1", "true", "yes", "on", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError("ZAI_THINKING must be enabled or disabled.")


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }
    deepseek_thinking = is_deepseek_model(model) and deepseek_thinking_enabled()
    zai_thinking = is_glm_model(model) and zai_thinking_enabled()
    if max_tokens is not None:
        if model_uses_max_completion_tokens(model):
            # Reasoning models (gpt-5 / o1 / o3 / o4) consume hidden reasoning
            # tokens out of max_completion_tokens before emitting output. The
            # callsites pick budgets tuned for gpt-4o, so bump them up with a
            # multiplier + floor to keep reasoning-heavy short calls alive.
            kwargs["max_completion_tokens"] = max(max_tokens * 4, 2000)
        elif deepseek_thinking:
            # DeepSeek reasoning and final output share this budget. The
            # original limits were tuned for non-reasoning chat models.
            minimum = int(os.environ.get("DEEPSEEK_MIN_MAX_TOKENS", "4096"))
            kwargs["max_tokens"] = max(max_tokens, minimum)
        elif zai_thinking:
            minimum = int(os.environ.get("ZAI_MIN_MAX_TOKENS", "4096"))
            kwargs["max_tokens"] = max(max_tokens, minimum)
        else:
            kwargs["max_tokens"] = max_tokens
    if temperature is not None and not model_uses_max_completion_tokens(model) and not deepseek_thinking:
        kwargs["temperature"] = temperature
    if deepseek_thinking:
        # extra_body works with older OpenAI SDKs too and is merged into the
        # top-level JSON request sent to DeepSeek.
        deepseek_extra = dict(extra.pop("extra_body", {}) or {})
        deepseek_extra["thinking"] = {"type": "enabled"}
        deepseek_extra["reasoning_effort"] = deepseek_reasoning_effort()
        kwargs["extra_body"] = deepseek_extra
    elif zai_thinking:
        zai_extra = dict(extra.pop("extra_body", {}) or {})
        zai_extra["thinking"] = {"type": "enabled"}
        kwargs["extra_body"] = zai_extra
    kwargs.update(extra)
    return kwargs


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> Any:
    kwargs = build_chat_completion_kwargs(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
    )
    return with_retry(client.chat.completions.create, **kwargs)


def chat_completion_text(
    client: Any,
    *,
    model: str,
    messages: Sequence[Dict[str, Any]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> str:
    response = chat_completion(
        client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **extra,
    )
    content = response.choices[0].message.content if response.choices else ""
    return content or ""


def build_mem0_llm_config(
    *,
    provider: Optional[str],
    model: str,
    api_key: Optional[str],
    api_version: Optional[str],
    azure_endpoint: Optional[str],
    openai_base_url: Optional[str],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    normalized = normalize_llm_provider(provider)
    config: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if normalized == AZURE_OPENAI_PROVIDER:
        endpoint = resolve_base_url(normalized, azure_endpoint)
        if not endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT for mem0 Azure OpenAI config.")
        azure_kwargs: Dict[str, Any] = {
            "azure_deployment": model,
            "azure_endpoint": endpoint,
            "api_version": resolve_api_version(api_version),
        }
        key = resolve_api_key(normalized, api_key)
        if key:
            azure_kwargs["api_key"] = key
        config["azure_kwargs"] = azure_kwargs
        return {"provider": AZURE_OPENAI_PROVIDER, "config": config}

    key = resolve_api_key(normalized, api_key)
    if key:
        config["api_key"] = key
    resolved_openai_base_url = resolve_base_url(normalized, openai_base_url)
    if resolved_openai_base_url:
        config["openai_base_url"] = resolved_openai_base_url
    return {"provider": OPENAI_PROVIDER, "config": config}
