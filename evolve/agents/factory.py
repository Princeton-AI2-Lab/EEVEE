from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

from .prompt import PromptAgents
from .router import RouterAgents
from .template_text_agent import TemplateTextAgent

ClientFactory = Callable[..., Tuple[Any, str, str, dict[str, Any]]]


def build_evolve_agents(
    *,
    client_factory: ClientFactory,
    prompts_dir: Path,
    max_prompt_length: int,
    max_retry: int,
) -> tuple[PromptAgents, RouterAgents]:
    prompt_mutation_client, prompt_mutation_provider, prompt_mutation_model, prompt_mutation_params = client_factory(
        "prompt_mutation",
        "generator",
    )
    prompt_reflection_client, prompt_reflection_provider, prompt_reflection_model, prompt_reflection_params = client_factory(
        "prompt_reflection",
        "generator",
    )
    router_mutation_client, router_mutation_provider, router_mutation_model, router_mutation_params = client_factory(
        "router_mutation",
        "generator",
    )
    router_reflection_client, router_reflection_provider, router_reflection_model, router_reflection_params = client_factory(
        "router_reflection",
        "router_mutation",
        "generator",
    )
    router_analysis_client, router_analysis_provider, router_analysis_model, router_analysis_params = client_factory(
        "router_analysis",
        "router_reflection",
        "generator",
    )

    prompt_agents = PromptAgents(
        mutation=TemplateTextAgent(
            api_client=prompt_mutation_client,
            api_provider=prompt_mutation_provider,
            model=prompt_mutation_model,
            template_path=prompts_dir / "prompt_mutation.jinja",
            role="prompt_mutation",
            api_params=prompt_mutation_params,
            max_output_length=max_prompt_length,
            max_retry=max_retry,
            output_mode="prompt",
        ),
        reflection=TemplateTextAgent(
            api_client=prompt_reflection_client,
            api_provider=prompt_reflection_provider,
            model=prompt_reflection_model,
            template_path=prompts_dir / "prompt_reflection.jinja",
            role="prompt_reflection",
            api_params=prompt_reflection_params,
            max_output_length=max_prompt_length,
            max_retry=max_retry,
            output_mode="prompt",
        ),
    )
    router_agents = RouterAgents(
        mutation=TemplateTextAgent(
            api_client=router_mutation_client,
            api_provider=router_mutation_provider,
            model=router_mutation_model,
            template_path=prompts_dir / "router_mutation.jinja",
            role="router_mutation",
            api_params=router_mutation_params,
            max_output_length=max_prompt_length,
            max_retry=max_retry,
            output_mode="prompt",
        ),
        analysis=TemplateTextAgent(
            api_client=router_analysis_client,
            api_provider=router_analysis_provider,
            model=router_analysis_model,
            template_path=prompts_dir / "router_analysis.jinja",
            role="router_analysis",
            api_params=router_analysis_params,
            max_output_length=2000,
            max_retry=max_retry,
            output_mode="text",
        ),
        reflection=TemplateTextAgent(
            api_client=router_reflection_client,
            api_provider=router_reflection_provider,
            model=router_reflection_model,
            template_path=prompts_dir / "router_reflection.jinja",
            role="router_reflection",
            api_params=router_reflection_params,
            max_output_length=max_prompt_length,
            max_retry=max_retry,
            output_mode="prompt",
        ),
    )
    return prompt_agents, router_agents
