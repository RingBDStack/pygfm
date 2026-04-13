# FlashAttn: set GRAPHGPT_NO_FLASH_ATTN=1 to skip if missing (see scripts/tune_script/graphgpt_stage1_1gpu.sh)
import os

if os.environ.get("GRAPHGPT_NO_FLASH_ATTN", "").lower() not in ("1", "true", "yes"):
    try:
        from .llama_flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )

        replace_llama_attn_with_flash_attn()
    except ModuleNotFoundError as e:
        # flash-attn is optional; fall back silently unless explicitly required.
        if getattr(e, "name", "") not in ("flash_attn", "flash-attn"):
            raise

from .train_graph import train

if __name__ == "__main__":
    train()
