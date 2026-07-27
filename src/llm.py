from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from .config import get_llm_settings, missing_key_message


def get_llm(config: RunnableConfig, json_mode: bool = True) -> ChatOpenAI:
    """Create a ChatOpenAI-compatible client from runtime configuration."""
    configurable = config.get("configurable", {}) if config else {}
    settings = get_llm_settings(configurable)
    api_key = settings.pop("api_key", None)
    if not api_key:
        raise RuntimeError(missing_key_message("DEEPSEEK_API_KEY"))

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        api_key=api_key,
        model_kwargs=model_kwargs,
        **settings,
    )
