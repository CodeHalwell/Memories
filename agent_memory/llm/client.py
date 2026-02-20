"""LiteLLM wrapper with retry and structured logging."""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from agent_memory.config import MEMORY_CONFIG

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


async def llm_complete(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 3,
) -> str:
    """Send a completion request via litellm and return the text response."""
    model = model or MEMORY_CONFIG["llm_model"]
    temperature = temperature if temperature is not None else MEMORY_CONFIG["llm_temperature"]

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                raise
            logger.warning("LLM call failed (attempt %d/%d), retrying...", attempt + 1, max_retries)

    return ""  # unreachable but satisfies type checker


async def llm_complete_json(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Send a completion request and parse the response as JSON.

    The prompt should instruct the LLM to respond with valid JSON only.
    """
    text = await llm_complete(prompt, system=system, model=model, temperature=temperature)

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    return json.loads(cleaned)
