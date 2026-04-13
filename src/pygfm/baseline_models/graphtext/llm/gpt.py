import os
from time import sleep

from tenacity import retry, stop_after_attempt, wait_random_exponential

from pygfm.public.utils.llm import openai_chat_completion
from utils.basics import logger
from .llm import LLM


class GPT(LLM):
    def __init__(
        self,
        openai_name="gpt-3.5-turbo",
        temperature=0,
        top_p=1,
        max_tokens=200,
        sleep_time=0,
        api_key=None,
        **kwargs,
    ):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert resolved_key, "Please provide openai_api_key in config or set OPENAI_API_KEY env var."
        self._api_key = resolved_key
        self.model = openai_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.sleep_time = sleep_time
        logger.critical(f"Using OPENAI {openai_name.upper()}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False):
        text = openai_chat_completion(
            [{"role": "user", "content": prompt}],
            model=self.model,
            api_key=self._api_key,
            temperature=0.0 if choice_only else self.temperature,
            top_p=self.top_p,
            max_tokens=1 if choice_only else self.max_tokens,
        )
        sleep(self.sleep_time)
        return text
