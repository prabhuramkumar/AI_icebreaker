"""Module for interfacing with OpenAI (gpt-3.5-turbo) and OpenAI embeddings."""

import logging
from typing import Callable, Optional, Any, Dict

import openai
from openai import OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import config

logger = logging.getLogger(__name__)


def _ensure_api_key() -> bool:
    if not getattr(config, "OPENAI_API_KEY", None):
        logger.error("OPENAI_API_KEY not set in config.")
        return False
    openaiKey = config.OPENAI_API_KEY
    logger.info(f"-------Imported gggg OpenAI Key------:{openaiKey}")
    return True


def create_openai_embedding() -> OpenAIEmbedding:
    """Create a simple embedding function using OpenAI Embeddings API.

    Returns:
        A callable that accepts a single string and returns a list of floats (the embedding),
        or None if creation failed.
    """
    if not _ensure_api_key():
        return None
    model_id = getattr(config, "EMBEDDING_MODEL_ID", "text-embedding-3-small")

    try:
        openai_embedding = OpenAIEmbedding(
            model=model_id,
            embed_batch_size=10,
            api_key=config.OPENAI_API_KEY,
        )
        logger.info(f"Created OpenAI Embedding model: {model_id}")
        return openai_embedding

    except Exception as e:
        logger.error(f"Error in openai_embedding: {e}")
        return None


def create_openai_llm(
    temperature: float = None,
    max_new_tokens: int = None,
) -> OpenAI:
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

    try:
        openai_llm = OpenAI(
            model=model_id,              # or "gpt-4.1-mini" / "gpt-3.5-turbo"
            temperature=temperature,      # controls randomness
            max_tokens=max_tokens     # OpenAI uses `max_tokens` instead of `max_new_tokens`
        )
        logger.info("Created OpenAI LLM model: gpt-4.1")
        return openai_llm

    except Exception as e:
        logger.error(f"Error in openai_llm: {e}")
        return None


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model id in runtime config."""
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")