import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import openai
    _OPENAI_IMPORT_ERROR = None
except ImportError as exc:
    _OPENAI_IMPORT_ERROR = exc

    class _MissingOpenAIModule:
        class OpenAI:
            pass

    openai = _MissingOpenAIModule()

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency; env vars may already be provided by the shell.
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()


def _get_provider_base(api_provider: str):
    """Resolve base_url and api_key for a given api_provider."""
    providers = {
        "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    }
    if api_provider not in providers:
        raise ValueError(
            f"Invalid api_provider: {api_provider}. "
            f"Must be one of: {', '.join(list(providers.keys()) + ['sglang'])}."
        )
    base_url, env_key = providers[api_provider]
    api_key = os.getenv(env_key, "")
    if not api_key:
        raise ValueError(f"{api_provider} api key not found in environment variables")
    return base_url, api_key


def _resolve_provider(api_provider: str, model: str):
    """Resolve (url, api_key, client_model) for one agent."""
    if api_provider == "sglang":
        url = f"http://localhost:{model}/v1"
        return url, "EMPTY", "default"
    base_url, api_key = _get_provider_base(api_provider)
    return base_url, api_key, model


def parse_model_spec(s: str) -> Tuple[str, str]:
    """Parse 'api_provider:model' string into (api_provider, model)."""
    if ":" not in s:
        raise ValueError(f"Invalid model spec: {s!r}. Expected format api_provider:model.")
    api_provider, model = s.strip().split(":", 1)
    api_provider = api_provider.strip()
    model = model.strip()
    if not api_provider or not model:
        raise ValueError(f"Invalid model spec: {s!r}. Both api_provider and model must be non-empty.")
    return api_provider, model


def load_api_params_config(config_path: str = None) -> Dict[str, Any]:
    """Load API parameters configuration from JSON file."""
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "api_params_config.json")
    if not os.path.exists(config_path):
        print(f"Warning: API params config file not found at {config_path}, using empty config")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load API params config: {e}, using empty config")
        return {}


def get_api_params_for_model(api_provider: str, model: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get API parameters for a specific model from configuration."""
    if config is None:
        config = load_api_params_config()
    model_key = f"{api_provider}:{model}"
    if model_key in config:
        return config[model_key].copy()
    if "default" in config:
        return config["default"].copy()
    return {}


def create_client(api_provider: str, model: str) -> Tuple[openai.OpenAI, str]:
    """Create an OpenAI client for the given provider/model. Returns (client, client_model)."""
    if _OPENAI_IMPORT_ERROR is not None:
        raise ImportError(
            "The 'openai' package is required to create API clients."
        ) from _OPENAI_IMPORT_ERROR
    url, api_key, client_model = _resolve_provider(api_provider, model)
    client = openai.OpenAI(api_key=api_key, base_url=url, timeout=300.0)
    return client, client_model
