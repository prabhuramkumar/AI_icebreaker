"""Module for interfacing with OpenAI (gpt-3.5-turbo) and OpenAI embeddings."""

import logging
from typing import Callable, Optional, Any, Dict

import openai
import config

logger = logging.getLogger(__name__)


def _ensure_api_key() -> bool:
    if not getattr(config, "OPENAI_API_KEY", None):
        logger.error("OPENAI_API_KEY not set in config.")
        return False
    openai.api_key = config.OPENAI_API_KEY
    return True


def create_openai_embedding() -> Optional[Callable[[str], Optional[list]]]:
    """Create a simple embedding function using OpenAI Embeddings API.

    Returns:
        A callable that accepts a single string and returns a list of floats (the embedding),
        or None if creation failed.
    """
    if not _ensure_api_key():
        return None

    model_id = getattr(config, "EMBEDDING_MODEL_ID", "text-embedding-3-small")

    def embed(text: str) -> Optional[list]:
        try:
            resp = openai.Embedding.create(model=model_id, input=text)
            return resp["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    logger.info(f"OpenAI embedding function created (model={model_id})")
    return embed


def create_openai_llm(
    temperature: float = None,
    max_new_tokens: int = None,
) -> Optional[Callable[[str, Optional[str]], str]]:
    """Create a simple LLM generator backed by OpenAI Chat API (gpt-3.5-turbo).

    The returned callable has the signature: generate(prompt: str, system_prompt: Optional[str] = None) -> str

    Args:
        temperature: sampling temperature (falls back to config.TEMPERATURE).
        max_new_tokens: maximum tokens to generate (falls back to config.MAX_NEW_TOKENS).

    Returns:
        A generate() callable or None if creation failed.
    """
    if not _ensure_api_key():
        return None

    model_id = getattr(config, "LLM_MODEL_ID", "gpt-3.5-turbo")
    temperature = config.TEMPERATURE if temperature is None else temperature
    max_tokens = config.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens

    def generate(prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = openai.ChatCompletion.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return ""

    logger.info(f"OpenAI LLM generator created (model={model_id})")
    return generate


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model id in runtime config."""
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")