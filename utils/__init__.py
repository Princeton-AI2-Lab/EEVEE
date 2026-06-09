"""
EEVEE utilities package.
"""

from .api import (
    create_client,
    get_api_params_for_model,
    load_api_params_config,
    parse_model_spec,
)
from .files import make_unique_filename_fragment, safe_filename_fragment
from .parsing import (
    extract_answer,
    extract_boxed_content,
    extract_prompt_text,
    extract_reasoning_and_answer,
    extract_xml_tag,
    normalize_prompt_output,
)
from .pipeline_helpers import (
    build_correctness_cache,
    load_jsonl,
    make_qid,
    split_train_val_examples,
)
from .run_logger import RunLogger
from .sampling import sample_questions_proportional

__all__ = [
    "create_client",
    "get_api_params_for_model",
    "load_api_params_config",
    "parse_model_spec",
    "make_unique_filename_fragment",
    "safe_filename_fragment",
    "extract_answer",
    "extract_boxed_content",
    "extract_prompt_text",
    "extract_reasoning_and_answer",
    "extract_xml_tag",
    "normalize_prompt_output",
    "build_correctness_cache",
    "load_jsonl",
    "make_qid",
    "split_train_val_examples",
    "RunLogger",
    "sample_questions_proportional",
]
