from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import BaseLoader, Environment, StrictUndefined

from utils.llm import timed_llm_call
from utils import normalize_prompt_output


_jinja_env = Environment(loader=BaseLoader(), undefined=StrictUndefined, keep_trailing_newline=True)


class TemplateTextAgent:
    def __init__(
        self,
        *,
        api_client: Any,
        api_provider: str,
        model: str,
        template_path: Path,
        role: str,
        api_params: Optional[Dict[str, Any]] = None,
        max_output_length: int = 10000,
        max_retry: int = 3,
        output_mode: str = "prompt",
    ):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.template_path = Path(template_path)
        self.role = role
        self.api_params = dict(api_params or {})
        self.max_output_length = int(max_output_length)
        self.max_retry = int(max_retry)
        self.output_mode = str(output_mode)
        self._template = _jinja_env.from_string(self.template_path.read_text(encoding="utf-8"))

    def run(
        self,
        *,
        call_id: str,
        log_dir: Optional[str] = None,
        **template_vars: Any,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retry + 1):
            try:
                rendered = self._template.render(**template_vars)
                api_params = dict(self.api_params)
                response, _ = timed_llm_call(
                    self.api_client,
                    self.api_provider,
                    self.model,
                    rendered,
                    role=self.role,
                    call_id=f"{call_id}_retry{attempt}" if attempt > 0 else call_id,
                    log_dir=log_dir,
                    api_params=api_params,
                )
                text = normalize_prompt_output(response) if self.output_mode == "prompt" else str(response or "").strip()
                if not text:
                    raise ValueError(f"{self.role} output is empty.")
                return text[: self.max_output_length]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retry:
                    raise
        raise last_error or ValueError(f"{self.role} failed without a concrete exception.")
