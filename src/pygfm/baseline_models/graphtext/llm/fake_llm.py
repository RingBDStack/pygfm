"""Debug fake LLM; no langchain dependency."""


class CpuFakeDebugraph_text:
    _i = 0

    def __init__(self, **kwargs):
        pass

    def generate_text(self, prompt, max_new_tokens=1, choice_only=False):
        if choice_only:
            # rotate letters for debug
            ch = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self._i % 26]
            self._i += 1
            return ch
        return "<answer>C</answer>"[: max(max_new_tokens, 20)]
