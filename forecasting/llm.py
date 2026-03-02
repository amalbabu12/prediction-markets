"""
LLM backend interface for the forecasting pipeline.

Provides an abstract base so any model can be swapped in,
plus a HuggingFaceBackend that works with:
  - HF Inference API (hosted, any public model)
  - Local TGI / vLLM server (OpenAI-compatible endpoint)
  - Any model accessible via huggingface_hub AsyncInferenceClient

Usage:
    # HF Inference API
    backend = HuggingFaceBackend(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key="hf_...",
    )

    # Local TGI server
    backend = HuggingFaceBackend(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        base_url="http://localhost:8080",
    )

    text = await backend.generate("What is 2+2?", system_prompt="Be brief.")
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Abstract interface ────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """
    Minimal async interface for any LLM.

    Implementations only need to define `generate`. Everything else in the
    forecasting pipeline works through this single method.
    """

    @abstractmethod
    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
    ) -> str:
        """
        Generate a response for the given prompt.

        Args:
            user_prompt: The user turn content.
            system_prompt: Optional system/instruction prefix.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Raw string response from the model.
        """
        ...


# ── OpenAI-compatible implementation (Gemini, Groq, Together, etc.) ──────────

class OpenAICompatibleBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible API using the openai package.

    Works with Gemini (generativelanguage.googleapis.com/v1beta/openai/),
    Groq (api.groq.com/openai/v1), Together, and any other OpenAI-compatible
    endpoint. Properly handles 429s, rate-limit errors, and RESOURCE_EXHAUSTED
    responses from Google.

    Args:
        model: Model name (e.g. "gemini-2.0-flash", "llama-3.3-70b-versatile")
        api_key: API key for the service.
        base_url: Base URL of the OpenAI-compatible endpoint.
        temperature: Sampling temperature (0.0 = deterministic).
        rpm_limit: Max requests per minute (0 = no limit).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        rpm_limit: int = 0,
        **extra_kwargs: Any,
    ) -> None:
        from openai import AsyncOpenAI
        import asyncio as _asyncio

        self.model = model
        self._temperature = temperature
        self._extra_kwargs = extra_kwargs
        self._rpm_lock = _asyncio.Lock() if rpm_limit > 0 else None
        self._rpm_interval = (60.0 / rpm_limit) if rpm_limit > 0 else 0.0
        self._last_call_time: float = 0.0
        # max_retries=0: disable openai's built-in retry loop so our token-bucket
        # retry logic has full control and doesn't double the request count.
        self._client = AsyncOpenAI(api_key=api_key or "none", base_url=base_url, max_retries=0)
        logger.info(
            "OpenAICompatibleBackend initialised — model=%s  endpoint=%s",
            model, base_url or "OpenAI",
        )

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        _retries: int = 6,
    ) -> str:
        import asyncio as _asyncio
        import time as _time
        import re as _re

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(_retries):
            # Token bucket — enforced on every attempt including retries
            if self._rpm_lock is not None:
                async with self._rpm_lock:
                    elapsed = _time.monotonic() - self._last_call_time
                    wait = self._rpm_interval - elapsed
                    if wait > 0:
                        await _asyncio.sleep(wait)
                    self._last_call_time = _time.monotonic()

            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=self._temperature,
                    **self._extra_kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                err = str(exc)
                is_rate_limit = (
                    "429" in err
                    or "rate_limit" in err.lower()
                    or "resource_exhausted" in err.lower()
                    or "quota" in err.lower()
                    or "too many requests" in err.lower()
                )
                if is_rate_limit:
                    m = _re.search(r"try again in (\d+(?:\.\d+)?)s", err, _re.IGNORECASE)
                    wait = float(m.group(1)) + 2 if m else min(60, 5 * (2 ** attempt))
                    logger.warning("Rate limited, retrying in %.0fs (attempt %d/%d)", wait, attempt + 1, _retries)
                    await _asyncio.sleep(wait)
                    continue
                logger.error("generate() failed (attempt %d/%d): %s", attempt + 1, _retries, exc)
                if attempt < _retries - 1:
                    await _asyncio.sleep(2)
                    continue
        logger.error("generate() exhausted %d retries", _retries)
        return ""


# ── HuggingFace implementation ────────────────────────────────────────────────

class HuggingFaceBackend(LLMBackend):
    """
    LLM backend backed by any Hugging Face model via AsyncInferenceClient.

    Supports chat_completion (all modern instruction models) and falls back
    to text_generation for base/non-chat models.

    Args:
        model: HF model ID (e.g. "mistralai/Mistral-7B-Instruct-v0.3") or
               a full URL for a local TGI/vLLM server.
        api_key: HF API token (hf_...). Not needed for local servers.
        base_url: Override the endpoint (e.g. "http://localhost:8080").
                  When set, `model` is passed as the model name to the server.
        temperature: Sampling temperature (0.0 = deterministic).
        extra_kwargs: Any additional kwargs forwarded to chat_completion.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        rpm_limit: int = 0,
        **extra_kwargs: Any,
    ) -> None:
        from huggingface_hub import AsyncInferenceClient
        import asyncio as _asyncio

        self.model = model
        self._temperature = temperature
        self._extra_kwargs = extra_kwargs
        # Shared token-bucket rate limiter across all coroutines
        self._rpm_lock = _asyncio.Lock() if rpm_limit > 0 else None
        self._rpm_interval = (60.0 / rpm_limit) if rpm_limit > 0 else 0.0
        self._last_call_time: float = 0.0
        # AsyncInferenceClient doesn't accept both model + base_url.
        # For OpenAI-compatible endpoints (Groq, Together, etc.) pass base_url
        # only and supply model per-call; for HF Inference API pass model only.
        if base_url:
            self._client = AsyncInferenceClient(token=api_key, base_url=base_url)
        else:
            self._client = AsyncInferenceClient(model=model, token=api_key)
        logger.info(
            "HuggingFaceBackend initialised — model=%s  endpoint=%s",
            model,
            base_url or "HF Inference API",
        )

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        _retries: int = 6,
    ) -> str:
        import asyncio as _asyncio
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        import time as _time
        import re as _re

        for attempt in range(_retries):
            # Token bucket: enforce inter-call spacing before every attempt
            # (including retries) so the next group never fires immediately
            # after a long retry delay.
            if self._rpm_lock is not None:
                async with self._rpm_lock:
                    elapsed = _time.monotonic() - self._last_call_time
                    wait = self._rpm_interval - elapsed
                    if wait > 0:
                        await _asyncio.sleep(wait)
                    self._last_call_time = _time.monotonic()

            try:
                response = await self._client.chat_completion(
                    messages=messages,
                    model=self.model,
                    max_tokens=max_new_tokens,
                    temperature=self._temperature,
                    **self._extra_kwargs,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                err = str(exc)
                # Rate limit — back off and retry
                if "429" in err or "rate_limit" in err.lower():
                    # Try to parse "Please try again in Xs" from Groq's message
                    m = _re.search(r"try again in (\d+(?:\.\d+)?)s", err)
                    wait = float(m.group(1)) + 2 if m else min(65, 5 * (2 ** attempt))
                    logger.warning("Rate limited, retrying in %.0fs (attempt %d/%d)", wait, attempt + 1, _retries)
                    await _asyncio.sleep(wait)
                    continue
                # Non-chat model — fall back to text_generation
                logger.warning("chat_completion failed (%s), trying text_generation: %s", self.model, exc)
                flat = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
                try:
                    response = await self._client.text_generation(
                        flat, max_new_tokens=max_new_tokens, temperature=self._temperature,
                    )
                    return response if isinstance(response, str) else str(response)
                except Exception as exc2:
                    logger.warning("text_generation also failed: %s", exc2)
                    return ""
        logger.error("generate() failed after %d retries", _retries)
        return ""


# ── JSON extraction utility ───────────────────────────────────────────────────

def extract_json(text: str) -> Any:
    """
    Best-effort extraction of a JSON value from raw model output.

    Tries, in order:
      1. Direct json.loads on the stripped response
      2. Content inside a ```json ... ``` or ``` ... ``` code fence
      3. First {...} or [...] block found anywhere in the text

    Returns the parsed Python object, or None if nothing valid is found.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fence
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced brace/bracket block
    for pattern in (r"(\{[\s\S]*\})", r"(\[[\s\S]*\])"):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    logger.debug("extract_json: no valid JSON found in: %r", text[:200])
    return None
