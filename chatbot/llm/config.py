from functools import lru_cache

from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from chatbot.settings import S


@lru_cache
def get_model(model_name: str | None = None):
    """Initialize LLM model"""

    name = model_name or S.LLM_MODEL

    if name in S.GOOGLE_LLM_MODELS:
        return ChatGoogleGenerativeAI(
            model=name,
            location="global",
            vertexai=True,
            temperature=S.TEMPERATURE,
            max_tokens=S.MAX_TOKENS,
            max_retries=1,
            thinking_budget=S.THINKING_BUDGET,
            timeout=S.TIMEOUT_LLM,
            include_thoughts=False,
        )

    if name in S.AWS_LLM_MODELS:
        return ChatBedrockConverse(
            model=name,
            temperature=S.TEMPERATURE,
            max_tokens=S.MAX_TOKENS,
        )

    if name in S.OLLAMA_LLM_MODELS:
        return ChatOllama(
            model=name,
            temperature=S.TEMPERATURE,
        )

    raise ValueError(f"LLM model '{name}' non available.")
