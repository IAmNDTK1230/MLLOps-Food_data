import json

from src.aspect_conditioned.prompts import input_payload
from src.proposed_v1.runtime import LocalQwen
from src.proposed_v2.runner import TrackedRuntime


class AspectRuntime(TrackedRuntime):
    """Reuse exact generation/parser/retry implementation with a scoped transport adapter."""

    def predict_aspect(self, instruction, text, aspect):
        payload = input_payload(text, aspect)
        original = self.session

        class AspectTransport:
            def post(self, url, **kwargs):
                # LocalQwen creates a text-only user message. Replace only that
                # message with the explicitly whitelisted new input contract.
                kwargs["json"]["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
                return original.post(url, **kwargs)

        self.session = AspectTransport()
        try:
            return LocalQwen.predict(self, instruction, text)
        finally:
            self.session = original
