from __future__ import annotations

from dataclasses import dataclass

from .template_text_agent import TemplateTextAgent


@dataclass(frozen=True)
class PromptAgents:
    mutation: TemplateTextAgent
    reflection: TemplateTextAgent
