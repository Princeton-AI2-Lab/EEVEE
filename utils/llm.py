"""
==============================================================================
llm.py
==============================================================================

This file contains the LLM class for the project.

"""
import time
import random
from datetime import datetime
import os
from utils.llm_logger import log_llm_call, log_problematic_request
from utils.llm_errors import (
    completion_tokens_key,
    handle_llm_error,
)

import yaml
import importlib
from pathlib import Path


def timed_llm_call(client, api_provider, model, prompt, role, call_id, log_dir=None,
                   sleep_seconds=15, retries_upper_bound=10, attempt=1,
                   api_params=None, messages=None):
    """
    Make a timed LLM call with error handling and retry logic.
    
    ERROR HANDLING STRATEGY:
    - All errors (timeouts, rate limits, server errors, empty responses) are retried up to retries_upper_bound times
    - Empty responses are logged to problematic_requests/ for SambaNova support analysis
    - After all retries exhausted, the exception is raised to let the caller decide how to handle it
    
    Args:
        client: API client
        model: Model name to use
        prompt: Text prompt to send
        role: Role for logging, such as generator, mutation, analysis, or reflection.
        call_id: Unique identifier for this call (format: {train|test}_{role}_{details})
        log_dir: Directory for detailed logging
        sleep_seconds: Base sleep time between retries
        retries_upper_bound: Maximum number of retries for timeouts/rate limits/empty responses
        attempt: Current attempt number (for recursive calls)
        api_params: Optional dict of additional API parameters (e.g., {"temperature": 0.7, "max_tokens": 4096})
    
    Returns:
        tuple: (response_text, call_info_dict)
        
    Raises:
        Exception: If all retries are exhausted, the original exception is raised
    """

    start_time = time.time()
    prompt_time = time.time()
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    
    print(f"[{role.upper()}] Starting call {call_id}...")
    
    # Check if we're using API key mixer for dynamic key rotation on retries
    using_key_mixer = False
    adaptive_completion_limit = None
    
    while True:
        try:
            # Get client
            active_client = client

            # Build base API parameters. Optional generation controls should come
            # only from api_params_config or caller-provided api_params.
            final_api_params = {
                "model": model,
                "messages": messages,
            }

            # Merge user-provided api_params.
            if api_params:
                for k, v in api_params.items():
                    # Handle max_tokens key conversion for OpenAI
                    if k == "max_tokens" and api_provider == "openai":
                        final_api_params["max_completion_tokens"] = v
                    elif k not in ["model", "messages"]:  # Don't override required params
                        final_api_params[k] = v

            if adaptive_completion_limit is not None:
                final_api_params[completion_tokens_key(api_provider)] = adaptive_completion_limit
            
            # Keep final_api_params for error logging (copy to avoid mutation)
            log_api_params = final_api_params.copy()

            # Some OpenAI client versions do not accept extra_body, so retry without it.
            call_start = time.time()
            try:
                response = active_client.chat.completions.create(**final_api_params)
            except TypeError as te:
                if "extra_body" in str(te) and "extra_body" in final_api_params:
                    final_api_params_no_extra = {k: v for k, v in final_api_params.items() if k != "extra_body"}
                    print(f"[{role.upper()}] Call {call_id} client does not support extra_body; retrying without it: {te}")
                    response = active_client.chat.completions.create(**final_api_params_no_extra)
                else:
                    raise
            call_end = time.time()
            
            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("Empty response from API")
            
            response_time = time.time()
            total_time = response_time - start_time
            raw_response_content = response.choices[0].message.content
            
            if raw_response_content is None:
                raise Exception("API returned None content")

            response_content = raw_response_content
            
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "raw_response": raw_response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": response.usage.prompt_tokens,
                "response_num_tokens": response.usage.completion_tokens,
            }
            
            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            return response_content, call_info
            
        except Exception as e:
            plan = handle_llm_error(
                e,
                role=role,
                call_id=call_id,
                model=model,
                prompt=prompt,
                api_params=log_api_params,
                api_provider=api_provider,
                sleep_seconds=sleep_seconds,
                log_problematic_request=log_problematic_request,
                log_dir=log_dir,
                using_key_mixer=using_key_mixer,
                client=client,
            )
            if attempt < retries_upper_bound and plan:
                attempt += 1
                if plan.get("completion_limit") is not None:
                    adaptive_completion_limit = plan["completion_limit"]
                error_type = plan["message"]
                base_sleep = plan["sleep"]
                if error_type == "unknown error":
                    print(f"[{role.upper()}] Call {call_id} actual exception: {type(e).__name__}: {e}")
                sleep_time = base_sleep * random.uniform(0.5, 1.5) if base_sleep else 0
                if sleep_time:
                    print(f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                          f"({attempt}/{retries_upper_bound})...")
                    time.sleep(sleep_time)
                else:
                    print(f"[{role.upper()}] Call {call_id} {error_type}, retrying ({attempt}/{retries_upper_bound})...")
                continue
            
            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }
            
            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")
            
            if log_dir:
                log_llm_call(log_dir, call_info)
            
            raise e
        
