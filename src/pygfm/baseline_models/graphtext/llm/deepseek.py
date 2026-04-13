"""DeepSeek Chat API (OpenAI-compatible ``/v1/chat/completions``)."""
import json
import os
import urllib.error
import urllib.request
from time import sleep

from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils.basics import logger

from .llm import LLM


class DeepSeek(LLM):
    """HTTP client for DeepSeek, same role as the GPT wrapper in GraphText ICL."""

    def __init__(
        self,
        model="deepseek-chat",
        api_url="https://api.deepseek.com/v1/chat/completions",
        api_key=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=200,
        sleep_time=0,
        system_message=None,
        request_timeout=120,
        **kwargs,
    ):
        key = (api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        assert key, (
            "Set DEEPSEEK_API_KEY in the environment or pass llm.api_key=... (Hydra override)."
        )
        self.api_key = key
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time
        self.system_message = system_message
        self.request_timeout = request_timeout
        logger.critical(f"Using DeepSeek API, model={model}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False):
        # ICL class selection: very low temperature + short output, aligned with gpt.py
        temp = 0.0 if choice_only else self.temperature
        max_t = 1 if choice_only else max(int(self.max_tokens), int(max_new_tokens))

        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_t,
            "stream": False,
        }
        if self.top_p is not None:
            body["top_p"] = self.top_p

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace") if e.fp else ""
            logger.error(f"DeepSeek HTTP {e.code}: {err}")
            raise
        sleep(self.sleep_time)
        return result["choices"][0]["message"]["content"]
