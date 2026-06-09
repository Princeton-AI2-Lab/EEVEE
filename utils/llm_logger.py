"""
EEVEE Logger
===========
Logging functions for LLM calls and experiment events.
"""
import os
import json
from datetime import datetime

from utils.files import safe_filename_fragment


def log_llm_call(log_dir, call_info):
    """Log detailed information about each LLM call."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_role = safe_filename_fragment(str(call_info.get("role", "unknown")), fallback="unknown")
    safe_call_id = safe_filename_fragment(str(call_info.get("call_id", "unknown")), fallback="unknown")
    filename = f"{safe_role}_{safe_call_id}_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)

    call_info["timestamp"] = timestamp
    call_info["datetime"] = datetime.now().isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(call_info, f, indent=2, ensure_ascii=False)

    print(f"[LOG] {call_info['role']} call logged to {filename}")


def log_problematic_request(call_id, prompt, model, api_params, exception, log_dir,
                            using_key_mixer, key_mixer):
    """Log problematic requests that cause empty responses."""
    if not log_dir:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    problem_log_dir = os.path.join(log_dir, "problematic_requests")
    os.makedirs(problem_log_dir, exist_ok=True)

    current_api_key = None
    if using_key_mixer and key_mixer:
        stats = key_mixer.get_usage_stats()
        if stats:
            current_api_key = max(stats.keys(), key=lambda k: stats[k])
            current_api_key = f"{current_api_key[:8]}...{current_api_key[-8:]}"

    problem_info = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "call_id": call_id,
        "model": model,
        "api_params": api_params,
        "prompt": prompt,
        "prompt_length": len(prompt),
        "api_key_used": current_api_key,
        "exception_info": {
            "type": type(exception).__name__,
            "message": str(exception),
            "repr": repr(exception),
        },
    }

    if hasattr(exception, "response"):
        response_details = {"has_response_object": True}
        try:
            if hasattr(exception.response, "status_code"):
                response_details["status_code"] = exception.response.status_code
            if hasattr(exception.response, "text"):
                response_details["text"] = exception.response.text
        except Exception as e:
            response_details["extraction_error"] = str(e)
        problem_info["response_details"] = response_details

    safe_call_id = safe_filename_fragment(str(call_id or "unknown"), fallback="unknown")
    filename = f"empty_response_{safe_call_id}_{timestamp}.json"
    filepath = os.path.join(problem_log_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(problem_info, f, indent=2, ensure_ascii=False)

    print(f"[PROBLEM LOG] Saved problematic request to: problematic_requests/{filename}")
