"""
LLM judge utility for EEVEE.

This module provides LLM-based answer evaluation as an alternative to rule-based evaluation.
It calls an LLM API to determine if a predicted answer is equivalent to the ground truth.

Usage:
    judge = LLMJudge(api_provider="openrouter", model="deepseek/deepseek-chat-v3.1")
    is_correct = judge.judge(predicted="42", ground_truth="42", question="What is 6*7?")
"""

from __future__ import annotations

import json
import os
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import openai
from jinja2 import BaseLoader, Environment, StrictUndefined

from utils import safe_filename_fragment


# Initialize Jinja2 environment
_env = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    keep_trailing_newline=True
)

# Load prompt template
_template_path = Path(__file__).parent / "judge_prompt.jinja"
_prompt_template = None


def _get_prompt_template():
    """Lazy load the prompt template."""
    global _prompt_template
    if _prompt_template is None:
        with open(_template_path, "r", encoding="utf-8") as f:
            _prompt_template = _env.from_string(f.read())
    return _prompt_template


def _get_provider_base(api_provider: str) -> Tuple[str, str]:
    """Get base URL and API key for the given provider."""
    if api_provider == "sambanova":
        base_url = "https://api.sambanova.ai/v1"
        api_key = os.getenv('SAMBANOVA_API_KEY', '')
        if not api_key:
            raise ValueError("SambaNova api key not found in environment variables")
        return base_url, api_key
    elif api_provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError("OpenAI api key not found in environment variables")
        return base_url, api_key
    elif api_provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not api_key:
            raise ValueError("Openrouter api key not found in environment variables")
        return base_url, api_key
    elif api_provider == "deepseek":
        base_url = "https://api.deepseek.com"
        api_key = os.getenv('DEEPSEEK_API_KEY', '')
        if not api_key:
            raise ValueError("DeepSeek api key not found in environment variables")
        return base_url, api_key
    else:
        raise ValueError(f"Invalid api_provider: {api_provider}. Must be one of: sambanova, openai, openrouter, deepseek, sglang.")


def _resolve_provider(api_provider: str, model: str) -> Tuple[str, str, str]:
    """
    Resolve (url, api_key, client_model) for the judge.
    - sglang: model is the port; url = http://localhost:{port}/v1, client_model = "default"
    - others: url = base_url, api_key from env, client_model = model
    """
    if api_provider == "sglang":
        url = f"http://localhost:{model}/v1"
        return url, "EMPTY", "default"
    base_url, api_key = _get_provider_base(api_provider)
    return base_url, api_key, model


class LLMJudge:
    """
    LLM-based answer judge for evaluating if predicted answers match ground truth.
    
    This uses an LLM to determine semantic equivalence between answers, which is
    more flexible than rule-based matching for complex or open-ended answers.
    """
    
    def __init__(
        self,
        api_provider: str,
        model: str,
        api_params: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        sleep_seconds: float = 5.0,
        client=None,
        client_model: Optional[str] = None,
    ):
        """
        Initialize the LLM Judge.

        Args:
            api_provider: API provider (e.g., "openrouter", "openai", "sglang")
            model: Model name or port (for sglang) — only used when client is None
            api_params: Optional API parameters (temperature, max_tokens, etc.)
            max_retries: Maximum number of retries for failed API calls
            sleep_seconds: Base sleep time between retries
            client: Pre-built openai.OpenAI client (if provided, skips internal creation)
            client_model: Resolved model name to pass to the API (used with external client)
        """
        self.api_provider = api_provider
        self.api_params = api_params or {}
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

        if client is not None:
            self.client = client
            self.client_model = client_model or model
        else:
            url, api_key, self.client_model = _resolve_provider(api_provider, model)
            self.client = openai.OpenAI(api_key=api_key, base_url=url, timeout=300.0)

        self.model = self.client_model
        print(f"[LLM Judge] Initialized with {api_provider}:{self.client_model}")
    
    def judge(
        self,
        predicted: str,
        ground_truth: str,
        question: Optional[str] = None,
        context: Optional[str] = None,
        log_dir: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> bool:
        """
        Judge if the predicted answer is equivalent to the ground truth.
        
        Args:
            predicted: The predicted answer to evaluate
            ground_truth: The ground truth answer
            question: Optional question text for context
            context: Optional additional context
            log_dir: Optional directory for logging
            call_id: Optional identifier for this call
            
        Returns:
            True if the predicted answer is considered correct, False otherwise
        """
        # Render prompt
        template = _get_prompt_template()
        prompt = template.render(
            predicted=predicted,
            ground_truth=ground_truth,
            question=question or "",
            context=context or "",
        )
        
        # Make API call with retries
        attempt = 0
        start_time = time.time()
        
        while True:
            try:
                attempt += 1
                
                # Build API parameters from api_params_config (same as other models)
                # Start with user-provided api_params from config
                api_call_params = {
                    "model": self.client_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},  # Always use JSON mode for judge
                }
                
                # Apply params from api_params_config
                if self.api_params:
                    for k, v in self.api_params.items():
                        if k not in ["model", "messages", "response_format"]:
                            # Handle max_tokens key conversion for OpenAI
                            if k == "max_tokens" and self.api_provider == "openai":
                                api_call_params["max_completion_tokens"] = v
                            else:
                                api_call_params[k] = v
                call_start = time.time()
                response = self.client.chat.completions.create(**api_call_params)
                call_end = time.time()
                
                if not response or not response.choices:
                    raise Exception("Empty response from API")
                
                raw_response_content = response.choices[0].message.content
                if raw_response_content is None:
                    raise Exception("API returned None content")
                response_content = raw_response_content
                
                # Parse JSON response
                is_correct = False
                reasoning = ""
                
                # Try to extract JSON from response (may be wrapped in markdown code block)
                json_str = response_content.strip()
                
                # Remove markdown code block if present
                if json_str.startswith("```"):
                    # Find the end of first line (may have ```json or just ```)
                    first_newline = json_str.find("\n")
                    if first_newline != -1:
                        json_str = json_str[first_newline + 1:]
                    # Remove trailing ```
                    if json_str.endswith("```"):
                        json_str = json_str[:-3].strip()
                
                try:
                    result = json.loads(json_str)
                    is_correct_raw = result.get("is_correct", False)
                    reasoning = result.get("reasoning", "")
                    
                    # Handle various response formats
                    if isinstance(is_correct_raw, bool):
                        is_correct = is_correct_raw
                    elif isinstance(is_correct_raw, str):
                        is_correct = is_correct_raw.lower() in ["true", "yes", "correct", "1"]
                    elif isinstance(is_correct_raw, int):
                        is_correct = bool(is_correct_raw)
                    else:
                        # Fallback: look for specific pattern
                        is_correct = '"is_correct": true' in response_content.lower()
                        
                except json.JSONDecodeError:
                    # Fallback: look for specific JSON pattern in raw text
                    response_lower = response_content.lower()
                    # Must match the exact pattern, not just substring
                    if '"is_correct": true' in response_lower or '"is_correct":true' in response_lower:
                        is_correct = True
                    elif '"is_correct": false' in response_lower or '"is_correct":false' in response_lower:
                        is_correct = False
                    else:
                        # Default to False if can't parse
                        is_correct = False
                
                # Log successful call
                total_time = time.time() - start_time
                if log_dir:
                    self._log_judge_call(
                        log_dir=log_dir,
                        call_id=call_id,
                        prompt=prompt,
                        response=response_content,
                        raw_response=raw_response_content,
                        predicted=predicted,
                        ground_truth=ground_truth,
                        is_correct=is_correct,
                        reasoning=reasoning,
                        call_time=call_end - call_start,
                        total_time=total_time,
                        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                        completion_tokens=response.usage.completion_tokens if response.usage else None,
                    )
                
                return is_correct
                    
            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(k in error_msg for k in [
                    "timeout", "rate limit", "429", "500", "502", "503",
                    "empty response", "none content", "connection"
                ])
                
                if is_retryable and attempt < self.max_retries:
                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = self.sleep_seconds * jitter * attempt
                    print(f"[LLM Judge] Error: {e}, retrying in {sleep_time:.1f}s ({attempt}/{self.max_retries})...")
                    time.sleep(sleep_time)
                    continue
                
                # Log error and return False as fallback
                print(f"[LLM Judge] Failed after {attempt} attempts: {e}")
                total_time = time.time() - start_time
                if log_dir:
                    self._log_judge_call(
                        log_dir=log_dir,
                        call_id=call_id,
                        prompt=prompt,
                        response=None,
                        predicted=predicted,
                        ground_truth=ground_truth,
                        is_correct=False,
                        reasoning=None,
                        call_time=None,
                        total_time=total_time,
                        prompt_tokens=None,
                        completion_tokens=None,
                        error=str(e),
                    )
                
                # Default to False on error
                return False
    
    def _log_judge_call(
        self,
        log_dir: str,
        call_id: Optional[str],
        prompt: str,
        response: Optional[str],
        predicted: str,
        ground_truth: str,
        is_correct: bool,
        reasoning: Optional[str],
        call_time: Optional[float],
        total_time: float,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        raw_response: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Log judge call details to file."""
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_call_id = safe_filename_fragment(str(call_id or "unknown"), fallback="unknown")
        filename = f"judge_{safe_call_id}_{timestamp}.json"
        filepath = os.path.join(log_dir, filename)
        
        call_info = {
            "role": "judge",
            "call_id": call_id,
            "model": f"{self.api_provider}:{self.model}",
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "predicted": predicted,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "reasoning": reasoning,
            "prompt": prompt,
            "response": response,
            "raw_response": raw_response if raw_response is not None else response,
            "prompt_length": len(prompt),
            "response_length": len(response) if response else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "call_time": call_time,
            "total_time": total_time,
            "error": error,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(call_info, f, indent=2, ensure_ascii=False)


def create_evaluator(
    data_processor,
    judge_mode: str = "rule",
    llm_judge: Optional[LLMJudge] = None,
):
    """
    Create a unified evaluator function that wraps both rule-based and LLM-based evaluation.
    
    Args:
        data_processor: The data processor instance with answer_is_correct method
        judge_mode: "rule" for rule-based, "llm" for LLM-based evaluation
        llm_judge: LLMJudge instance (required if judge_mode is "llm")
        
    Returns:
        A callable that takes (predicted, ground_truth, task_dict=None, question=None, context=None, log_dir=None, call_id=None)
        and returns a boolean indicating if the answer is correct.
    """
    task_runner = getattr(data_processor, "run_sample", None)

    def _attach_task_runner(evaluate_fn):
        if callable(task_runner):
            setattr(evaluate_fn, "run_sample", task_runner)
        return evaluate_fn

    if judge_mode == "llm":
        if llm_judge is None:
            raise ValueError("llm_judge must be provided when judge_mode is 'llm'")

        def evaluate_fn(
            predicted: str,
            ground_truth: str,
            task_dict: Optional[Dict[str, Any]] = None,
            question: Optional[str] = None,
            context: Optional[str] = None,
            log_dir: Optional[str] = None,
            call_id: Optional[str] = None,
        ) -> bool:
            # Extract question from task_dict if not provided
            if question is None and task_dict:
                question = task_dict.get("question", "")
            if context is None and task_dict:
                context = task_dict.get("context", "")

            parse_fn = getattr(data_processor, "parse_prediction_for_evaluation", None)
            if callable(parse_fn):
                is_valid, parsed_answer, _ = parse_fn(predicted, task_dict=task_dict)
                if not is_valid:
                    return False
                predicted = parsed_answer or ""
            
            return llm_judge.judge(
                predicted=predicted,
                ground_truth=ground_truth,
                question=question,
                context=context,
                log_dir=log_dir,
                call_id=call_id,
            )
        
        return _attach_task_runner(evaluate_fn)
    
    else:  # rule mode (default)
        def evaluate_fn(
            predicted: str,
            ground_truth: str,
            task_dict: Optional[Dict[str, Any]] = None,
            question: Optional[str] = None,
            context: Optional[str] = None,
            log_dir: Optional[str] = None,
            call_id: Optional[str] = None,
        ) -> bool:
            parse_fn = getattr(data_processor, "parse_prediction_for_evaluation", None)
            if callable(parse_fn):
                is_valid, parsed_answer, _ = parse_fn(predicted, task_dict=task_dict)
                if not is_valid:
                    return False
                predicted = parsed_answer or ""

            return data_processor.answer_is_correct(predicted, ground_truth, task_dict=task_dict)
        
        return _attach_task_runner(evaluate_fn)
