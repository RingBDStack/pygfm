"""Compatibility shim; implementation lives in :mod:`pygfm.public.utils.llm.openai_client`."""

from .openai_client import get_openai_client, openai_chat

__all__ = ["get_openai_client", "openai_chat"]
