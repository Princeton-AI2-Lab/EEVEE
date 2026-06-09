import json
import re

try:
    import openai
except ImportError:
    class _MissingOpenAIModule:
        class AuthenticationError(Exception):
            pass

        class RateLimitError(Exception):
            pass

        class InternalServerError(Exception):
            pass

        class BadRequestError(Exception):
            pass

    openai = _MissingOpenAIModule()


CONTEXT_RETRY_BUFFER_TOKENS = 256


def completion_tokens_key(api_provider):
    return "max_completion_tokens" if api_provider == "openai" else "max_tokens"


def classify_llm_error(exc):
    lower_msg = str(exc).lower()
    is_json_error = (
        isinstance(exc, json.JSONDecodeError)
        or "jsondecodeerror" in type(exc).__name__.lower()
        or "jsondecodeerror" in lower_msg
    )
    is_empty_response = (
        "empty response" in lower_msg
        or "api returned none content" in lower_msg
        or is_json_error
    )

    status_code = _response_status_code(exc)
    is_server_error = (
        (status_code is not None and status_code >= 500)
        or any(k in lower_msg for k in [
            "500 internal server error",
            "internal server error",
            "502 bad gateway",
            "503 service unavailable",
        ])
        or (hasattr(openai, 'InternalServerError') and isinstance(exc, openai.InternalServerError))
    )
    is_rate_limit = (
        any(k in lower_msg for k in ["rate limit", "429", "rate_limit_exceeded"])
        or (hasattr(openai, 'RateLimitError') and isinstance(exc, openai.RateLimitError))
    )
    is_auth_error = (
        "401" in lower_msg
        or "user not found" in lower_msg
        or (hasattr(openai, 'AuthenticationError') and isinstance(exc, openai.AuthenticationError))
    )

    return {
        "lower_msg": lower_msg,
        "is_timeout": any(k in lower_msg for k in ["timeout", "timed out", "connection"]),
        "is_rate_limit": is_rate_limit,
        "is_empty_response": is_empty_response,
        "is_json_error": is_json_error,
        "is_auth_error": is_auth_error,
        "is_server_error": is_server_error,
        "is_context_length_error": _is_context_length_error(exc, lower_msg, status_code),
        "status_code": status_code,
    }


def context_retry_action(exc, api_params, api_provider):
    token_info = _extract_context_length_tokens(str(exc))
    if not token_info:
        return None, None

    retry_limit = (
        token_info["max_context_tokens"]
        - token_info["input_tokens"]
        - CONTEXT_RETRY_BUFFER_TOKENS
    )
    current_limit = _current_completion_limit(api_params, api_provider, token_info)
    if retry_limit <= 0 or (current_limit is not None and retry_limit >= current_limit):
        return token_info, None

    return token_info, {
        "limit": retry_limit,
        "message": (
            f"reducing {completion_tokens_key(api_provider)} "
            f"{current_limit or 'unknown'} -> {retry_limit}"
        ),
    }


def retry_plan(error_info, exc, api_params, api_provider, sleep_seconds, role, call_id):
    if error_info["is_rate_limit"]:
        return {"message": "rate limited", "sleep": sleep_seconds * 2}
    if error_info["is_context_length_error"]:
        token_info, action = context_retry_action(exc, api_params, api_provider)
        if action:
            return {
                "message": action["message"],
                "sleep": 0,
                "completion_limit": action["limit"],
            }
        if token_info:
            print_context_retry_skip(role, call_id, token_info)
        return None
    if error_info["is_server_error"]:
        return {"message": "server error (500+)", "sleep": sleep_seconds * 1.5}
    if error_info["is_empty_response"]:
        return {"message": "returned empty response", "sleep": sleep_seconds}
    if error_info["is_auth_error"]:
        return {"message": "authentication error", "sleep": sleep_seconds * 1.5}
    if error_info["is_timeout"]:
        return {"message": "timed out", "sleep": sleep_seconds}
    return {"message": "unknown error", "sleep": sleep_seconds}


def handle_llm_error(
    exc,
    *,
    role,
    call_id,
    model,
    prompt,
    api_params,
    api_provider,
    sleep_seconds,
    log_problematic_request,
    log_dir,
    using_key_mixer,
    client,
):
    error_info = classify_llm_error(exc)
    print_error_summary(role, call_id, exc, error_info)

    if error_info["is_empty_response"]:
        print_empty_response_debug(call_id, exc, model, prompt)
        log_problematic_request(
            call_id,
            prompt,
            model,
            api_params,
            exc,
            log_dir,
            using_key_mixer,
            client if using_key_mixer else None,
        )

    return retry_plan(error_info, exc, api_params, api_provider, sleep_seconds, role, call_id)


def print_error_summary(role, call_id, exc, error_info):
    prefix = f"[{role.upper()}]"
    if error_info["is_json_error"]:
        print(f"{prefix} JSON parsing error for {call_id}: {exc}")
        print(f"{prefix} Treating JSON parsing error as empty/bad response from API")
    if error_info["is_server_error"]:
        if error_info["status_code"]:
            print(f"{prefix} Server error detected: HTTP {error_info['status_code']}")
        else:
            print(f"{prefix} Server error detected in message: {str(exc)[:100]}...")
    if error_info["is_context_length_error"]:
        print(f"{prefix} Context length exceeded for {call_id}: {exc}")


def print_context_retry_skip(role, call_id, token_info):
    print(
        f"[{role.upper()}] Cannot reduce output length for {call_id}: "
        f"context={token_info['max_context_tokens']}, "
        f"input={token_info['input_tokens']}"
    )


def print_empty_response_debug(call_id, exc, model, prompt):
    print(f"\nDEBUG: Empty response detected for {call_id}")
    print(f"Exception type: {type(exc).__name__}")
    print(f"Exception message: {str(exc)}")
    print(f"Model: {model}")
    print(f"Prompt length: {len(prompt)}")
    print("Prompt preview (first 500 chars):")
    print(f"    {prompt[:500]}...")
    print(f"Full exception details: {repr(exc)}")

    if hasattr(exc, 'response'):
        print(f"Raw response object: {exc.response}")
        if hasattr(exc.response, 'text'):
            print(f"Raw response text: {exc.response.text}")
        if hasattr(exc.response, 'content'):
            print(f"Raw response content: {exc.response.content}")
    print("-" * 60)


def _response_status_code(exc):
    if not hasattr(exc, 'response'):
        return None
    try:
        return getattr(exc.response, 'status_code', None)
    except Exception:
        return None


def _is_context_length_error(exc, lower_msg, status_code):
    is_bad_request = (
        status_code == 400
        or "error code: 400" in lower_msg
        or "badrequesterror" in lower_msg
        or (hasattr(openai, 'BadRequestError') and isinstance(exc, openai.BadRequestError))
    )
    return is_bad_request and (
        "context length" in lower_msg
        or "context_length" in lower_msg
        or ("longer than" in lower_msg and "tokens" in lower_msg)
    )


def _extract_context_length_tokens(error_message):
    max_context = _parse_int(
        re.search(r"maximum context length (?:is|of)\s+([\d,]+)\s+tokens", error_message, re.IGNORECASE)
    )
    input_tokens = _parse_int(
        re.search(r"([\d,]+)\s+tokens?\s+from (?:the )?input messages", error_message, re.IGNORECASE)
    )
    if input_tokens is None:
        input_tokens = _parse_int(
            re.search(r"\(([\d,]+)\s+in (?:the )?messages?,\s*[\d,]+\s+in (?:the )?completion\)", error_message, re.IGNORECASE)
        )

    completion_tokens = _parse_int(
        re.search(r"([\d,]+)\s+tokens?\s+for (?:the )?completion", error_message, re.IGNORECASE)
    )
    if completion_tokens is None:
        completion_tokens = _parse_int(
            re.search(r"\([\d,]+\s+in (?:the )?messages?,\s*([\d,]+)\s+in (?:the )?completion\)", error_message, re.IGNORECASE)
        )

    if max_context is None or input_tokens is None:
        return None
    return {
        "max_context_tokens": max_context,
        "input_tokens": input_tokens,
        "completion_tokens": completion_tokens,
    }


def _current_completion_limit(api_params, api_provider, token_info):
    value = api_params.get(completion_tokens_key(api_provider))
    if value is None and token_info:
        value = token_info.get("completion_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_int(match):
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None
