import json
import re
from typing import Optional, Tuple


def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from an XML tag in text."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(response: str) -> str:
    """Extract final answer from model response."""
    if not isinstance(response, str):
        try:
            response = json.dumps(response, ensure_ascii=False)
        except TypeError:
            response = str(response)

    answer = extract_xml_tag(response, "final_answer")
    if answer:
        return answer

    answer = extract_boxed_content(response)
    if answer:
        return answer

    return "No final answer found"


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format."""
    pattern = r"\\boxed\{"
    match = re.search(pattern, text)
    if not match:
        return None
    start = match.end() - 1
    brace_count = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1 : i]
        i += 1
    return None


def extract_reasoning_and_answer(response: str) -> Tuple[str, str]:
    """
    Extract reasoning and answer from a model response.
    Returns (reasoning, answer).
    """
    answer = extract_answer(response)
    reasoning = extract_xml_tag(response, "reasoning")
    if reasoning is None:
        idx = response.find("<final_answer>")
        if idx > 0:
            reasoning = response[:idx].strip()
        else:
            reasoning = response
    return reasoning, answer


def extract_prompt_text(response: str) -> str:
    """Extract prompt text from a model response."""
    if not isinstance(response, str):
        try:
            response = json.dumps(response, ensure_ascii=False)
        except TypeError:
            response = str(response)

    match = re.search(r"```(?:\w+\n)?(.*?)```", response, re.DOTALL)
    if match:
        response = match.group(1)

    return response.strip()


def normalize_prompt_output(response: str) -> str:
    """Normalize a prompt-generation response into prompt text."""
    prompt = extract_prompt_text(response)
    if not prompt:
        raise ValueError("Prompt output is empty.")
    return prompt
