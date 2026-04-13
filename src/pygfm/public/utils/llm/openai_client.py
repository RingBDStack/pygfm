"""OpenAI Chat / Embeddings helpers (official openai Python SDK 1.x).

Requires: ``pip install openai>=1.0``; ``openai`` is imported lazily when you call these APIs.
Auth: uses ``OPENAI_API_KEY`` when ``api_key`` is not passed.
"""
from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from openai import OpenAI

ChatMessage = Mapping[str, str]


def _import_openai():
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Calling OpenAI API requires the 'openai' package. "
            "Install with: pip install 'openai>=1.0'"
        ) from e
    return OpenAI


def _resolve_client(
    *,
    api_key: Optional[str],
    base_url: Optional[str],
    client: Optional["OpenAI"],
    timeout: Optional[float] = None,
) -> "OpenAI":
    OpenAI = _import_openai()
    if client is not None:
        return client
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "Missing API key: set environment variable OPENAI_API_KEY or pass api_key=..."
        )
    kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> "OpenAI":
    """Build a reusable ``openai.OpenAI`` client (same rules as :func:`openai_chat_completion`)."""
    return _resolve_client(
        api_key=api_key, base_url=base_url, client=None, timeout=timeout
    )


def openai_chat_completion(
    messages: Sequence[ChatMessage],
    *,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["OpenAI"] = None,
    timeout: Optional[float] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    return_full_response: bool = False,
    **extra: Any,
) -> Union[str, Any]:
    """Chat Completions. Returns assistant text by default; ``return_full_response=True`` returns the full SDK object."""
    oc = _resolve_client(
        api_key=api_key, base_url=base_url, client=client, timeout=timeout
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in messages],
        "temperature": temperature,
        **extra,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p

    resp = oc.chat.completions.create(**payload)
    if return_full_response:
        return resp
    content = resp.choices[0].message.content
    return content if content is not None else ""


def openai_chat(
    prompt_or_messages: Union[str, List[dict[str, str]]],
    *,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    client: Optional["OpenAI"] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    **extra: Any,
) -> str:
    """Single- or multi-turn chat: ``str`` is one user message; ``list`` is OpenAI ``messages``.

    If the first argument is ``str``, optional ``system_prompt`` is inserted before the user message.
    """
    if isinstance(prompt_or_messages, str):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_or_messages})
    else:
        # Caller supplies full messages; do not inject system_prompt
        messages = [dict(m) for m in prompt_or_messages]

    return openai_chat_completion(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        client=client,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        **extra,
    )  # type: ignore[return-value]


class OpenAIChatClient:
    """Chat helper that reuses one client and default generation kwargs."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional["OpenAI"] = None,
        timeout: Optional[float] = None,
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
        **default_extra: Any,
    ) -> None:
        self.model = model
        self._client = _resolve_client(
            api_key=api_key,
            base_url=base_url,
            client=client,
            timeout=timeout,
        )
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._default_extra = default_extra

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_full_response: bool = False,
        **extra: Any,
    ) -> Union[str, Any]:
        merged = {**self._default_extra, **extra}
        return openai_chat_completion(
            messages,
            model=self.model,
            client=self._client,
            temperature=(
                self._default_temperature
                if temperature is None
                else temperature
            ),
            max_tokens=(
                self._default_max_tokens if max_tokens is None else max_tokens
            ),
            return_full_response=return_full_response,
            **merged,
        )

    def chat(
        self,
        user_prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages: List[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        result = self.complete(messages, **kwargs)
        assert isinstance(result, str)
        return result


def create_openai_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["OpenAI"] = None,
    timeout: Optional[float] = None,
) -> "OpenAI":
    """Return a reusable ``OpenAI`` instance (same kwargs contract as ``openai_chat_completion``)."""
    return _resolve_client(
        api_key=api_key, base_url=base_url, client=client, timeout=timeout
    )


def openai_embeddings(
    input_text: Union[str, Sequence[str]],
    *,
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional["OpenAI"] = None,
    timeout: Optional[float] = None,
    return_full_response: bool = False,
    **extra: Any,
) -> Union[List[float], List[List[float]], Any]:
    """Embeddings API: one string → ``List[float]``; multiple strings → ``List[List[float]]``."""
    oc = _resolve_client(
        api_key=api_key, base_url=base_url, client=client, timeout=timeout
    )
    if isinstance(input_text, str):
        inputs: List[str] = [input_text]
        single = True
    else:
        inputs = list(input_text)
        single = len(inputs) == 1

    resp = oc.embeddings.create(model=model, input=inputs, **extra)
    if return_full_response:
        return resp
    vecs = [d.embedding for d in resp.data]
    if single and isinstance(input_text, str):
        return vecs[0]
    return vecs


__all__ = [
    "ChatMessage",
    "OpenAIChatClient",
    "create_openai_client",
    "get_openai_client",
    "openai_chat",
    "openai_chat_completion",
    "openai_embeddings",
]
