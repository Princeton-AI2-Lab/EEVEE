"""
Generator agent: answers questions using a candidate prompt.
"""
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from jinja2 import BaseLoader, Environment, StrictUndefined
from utils.llm import timed_llm_call

_jinja_env = Environment(loader=BaseLoader(), undefined=StrictUndefined, keep_trailing_newline=True)
_template_path = Path(__file__).parent.parent / "prompts" / "generator.jinja"
with open(_template_path, "r", encoding="utf-8") as _f:
    _gen_template = _jinja_env.from_string(_f.read())


class Generator:
    """Generates answers using a candidate prompt."""

    def __init__(self, api_client, api_provider: str, model: str,
                 api_params: Optional[Dict[str, Any]] = None,
                 max_retry: int = 3):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.api_params = api_params
        self.max_retry = max_retry

    def generate(
        self,
        question: str,
        prompt_text: str,
        context: str = "",
        call_id: str = "gen",
        log_dir: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        prompt = _gen_template.render(
            prompt_text=prompt_text,
            question=question,
            context=context,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retry + 1):
            try:
                response, call_info = timed_llm_call(
                    self.api_client,
                    self.api_provider,
                    self.model,
                    prompt,
                    role="generator",
                    call_id=f"{call_id}_retry{attempt}" if attempt > 0 else call_id,
                    log_dir=log_dir,
                    api_params=self.api_params,
                )
                if response is not None and str(response).strip():
                    return response, call_info
                raise ValueError("Generator returned empty response.")
            except Exception as e:
                last_error = e
                if attempt < self.max_retry:
                    print(f"Warning: Generator output failed (attempt {attempt + 1}/{self.max_retry + 1}), retrying: {e}")
                else:
                    raise last_error
        raise last_error or ValueError("Generator generate failed.")

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        call_id: str = "gen",
        log_dir: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        prompt_for_log = "\n\n".join(
            f"{message.get('role', 'unknown').upper()}:\n{message.get('content', '')}"
            for message in messages
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retry + 1):
            try:
                response, call_info = timed_llm_call(
                    self.api_client,
                    self.api_provider,
                    self.model,
                    prompt_for_log,
                    role="generator",
                    call_id=f"{call_id}_retry{attempt}" if attempt > 0 else call_id,
                    log_dir=log_dir,
                    api_params=self.api_params,
                    messages=messages,
                )
                if response is not None and str(response).strip():
                    return response, call_info
                raise ValueError("Generator returned empty response.")
            except Exception as e:
                last_error = e
                if attempt < self.max_retry:
                    print(f"Warning: Generator output failed (attempt {attempt + 1}/{self.max_retry + 1}), retrying: {e}")
                else:
                    raise last_error
        raise last_error or ValueError("Generator generate_messages failed.")
