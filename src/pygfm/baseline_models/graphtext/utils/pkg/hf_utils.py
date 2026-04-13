"""HF download/load: implementation in ``pygfm.public.utils.llm.hf_hub``; decorators and graphtext logger here."""
from __future__ import annotations

import os

from pygfm.public.utils.llm.hf_hub import (
    download_hf_ckpt_to_local as _download_hf_ckpt_to_local,
    load_hf_auto_from_local_dir,
)
from utils.basics import logger, init_path, time_logger
from utils.pkg.distributed import master_process_only


@time_logger()
@master_process_only
def download_hf_ckpt_to_local(hf_name, local_dir):
    local_dir = init_path(local_dir)
    if not os.path.exists(f"{local_dir}config.json"):
        logger.critical(f"Downloading {hf_name} ckpt to {local_dir}")
        os.environ["CURL_CA_BUNDLE"] = ""
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    _download_hf_ckpt_to_local(
        hf_name,
        local_dir,
        hf_token=hf_token if hf_token else None,
    )


def load_hf_auto_model_and_tokenizer(hf_name, local_dir):
    local_dir = init_path(local_dir)
    download_hf_ckpt_to_local(hf_name, local_dir)
    return load_hf_auto_from_local_dir(local_dir)
