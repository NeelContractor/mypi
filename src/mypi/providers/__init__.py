from __future__ import annotations

from collections.abc import Callable

from ..types import Provider
from .anthropic_provider import AnthropicProvider
from .openai_compat import OpenAICompatProvider

DEFAULT_MAX_TOKENS = 8192

# OpenAI-compatible shims: any vendor exposing that API shape plugs in here
PROVIDERS: dict[str, Callable[[int], Provider]] = {
    "anthropic": AnthropicProvider,
    "anthropic-openai": lambda mt: OpenAICompatProvider(
        "anthropic-openai",
        "https://api.anthropic.com/v1/",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-5-5",
        mt,
    ),
    "groq": lambda mt: OpenAICompatProvider(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "openai/gpt-oss-20b",
        mt,
    ),
}


def get_provider(name: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Provider:
    create = PROVIDERS.get(name)
    if create is None:
        raise ValueError(f"unknown provider '{name}', choose from: {', '.join(PROVIDERS)}")
    return create(max_tokens)
