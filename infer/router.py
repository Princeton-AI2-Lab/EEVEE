from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from jinja2 import BaseLoader, Environment, StrictUndefined

from utils.llm import timed_llm_call


_jinja_env = Environment(loader=BaseLoader(), undefined=StrictUndefined, keep_trailing_newline=True)
_template = _jinja_env.from_string(
    (Path(__file__).parent.parent / "prompts" / "router.jinja").read_text(encoding="utf-8")
)


@dataclass(frozen=True)
class RouterDecision:
    slot_id: int
    reason: str


class Router:
    def __init__(
        self,
        *,
        api_client: Any,
        api_provider: str,
        model: str,
        api_params: Optional[Dict[str, Any]] = None,
        max_retry: int = 3,
    ):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.api_params = dict(api_params or {})
        self.max_retry = int(max_retry)

    @staticmethod
    def _fallback_decision(
        *,
        display_slot_labels: Sequence[str],
        error_tag: str,
    ) -> RouterDecision:
        if not display_slot_labels:
            return RouterDecision(slot_id=-1, reason=f"{error_tag}:no_labels")
        slot_id = random.randrange(len(display_slot_labels))
        return RouterDecision(slot_id=slot_id, reason=f"{error_tag}:random_fallback")

    def route(
        self,
        *,
        router_prompt: str,
        display_slot_labels: Sequence[str],
        prompt_set: Sequence[str],
        visible_text: str,
        call_id: str,
        log_dir: Optional[str] = None,
    ) -> RouterDecision:
        rendered = _template.render(
            router_prompt=router_prompt,
            slot_entries=[
                {
                    "slot_label": str(slot_label),
                    "prompt_text": str(prompt_set[idx]) if idx < len(prompt_set) else "",
                }
                for idx, slot_label in enumerate(display_slot_labels)
            ],
            visible_text=str(visible_text),
        )
        last_error: Optional[Exception] = None
        last_raw_response = ""
        for attempt in range(self.max_retry + 1):
            try:
                response, _ = timed_llm_call(
                    self.api_client,
                    self.api_provider,
                    self.model,
                    rendered,
                    role="router",
                    call_id=f"{call_id}_retry{attempt}" if attempt > 0 else call_id,
                    log_dir=log_dir,
                    api_params=self.api_params,
                )
                last_raw_response = str(response or "")
                slot_id, reason = self._parse_response(
                    last_raw_response,
                    display_slot_labels=display_slot_labels,
                )
                return RouterDecision(slot_id=slot_id, reason=reason)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retry:
                    break
        if last_error is not None:
            return self._fallback_decision(
                display_slot_labels=display_slot_labels,
                error_tag=f"invalid_after_retry:{type(last_error).__name__}",
            )
        return self._fallback_decision(
            display_slot_labels=display_slot_labels,
            error_tag="invalid_after_retry:empty",
        )

    @staticmethod
    def _parse_response(
        raw_response: str,
        *,
        display_slot_labels: Sequence[str],
    ) -> tuple[int, str]:
        stripped = str(raw_response or "").strip()
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("router_response_not_json_object")
        slot = str(payload.get("label", payload.get("slot", ""))).strip()
        reason = str(payload.get("reason", "")).strip()
        if slot in display_slot_labels:
            return display_slot_labels.index(slot), reason
        match = re.fullmatch(r"slot[_\s-]?([0-9]+)", slot, flags=re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(display_slot_labels):
                return idx, reason
        raise ValueError(f"router_response_invalid_slot:{slot}")
