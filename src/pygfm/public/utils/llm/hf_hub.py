"""Hugging Face Hub download + AutoModel load (shared across baselines)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def download_hf_ckpt_to_local(
    hf_name: str,
    local_dir: str,
    *,
    hf_token: Optional[str] = None,
) -> None:
    """If ``local_dir/config.json`` is missing, ``snapshot_download`` into ``local_dir``.

    Uses ``HF_ACCESS_TOKEN`` when ``hf_token`` is not passed (same as legacy GraphText).
    """
    from huggingface_hub import snapshot_download

    root = Path(local_dir)
    if (root / "config.json").is_file():
        return
    token = hf_token if hf_token is not None else os.environ.get("HF_ACCESS_TOKEN")
    root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", hf_name, local_dir)
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    snapshot_download(
        repo_id=hf_name,
        local_dir=str(root),
        token=token if token else None,
    )


def load_hf_auto_from_local_dir(local_dir: str) -> Tuple[Any, Any, Any]:
    """Load ``AutoModel`` / ``AutoTokenizer`` / ``AutoConfig`` from an existing local dir."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    bert = AutoModel.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model_cfg = AutoConfig.from_pretrained(local_dir)
    return bert, tokenizer, model_cfg


def load_hf_auto_model_and_tokenizer(
    hf_name: str,
    local_dir: str,
    *,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """``download_hf_ckpt_to_local`` then ``load_hf_auto_from_local_dir``."""
    download_hf_ckpt_to_local(hf_name, local_dir, hf_token=hf_token)
    return load_hf_auto_from_local_dir(local_dir)
