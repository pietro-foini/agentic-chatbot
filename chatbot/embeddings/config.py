from functools import lru_cache

from langchain_aws import BedrockEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from chatbot.settings import S


@lru_cache
def get_embeddings(model_name: str | None = None) -> Embeddings:
    """Initialize Embeddings model"""

    name = model_name or S.EMBEDDINGS_MODEL

    if name in S.GOOGLE_EMBEDDINGS_MODELS:
        return GoogleGenerativeAIEmbeddings(
            model=name,
            output_dimensionality=S.EMBEDDINGS_SIZE,
            vertexai=True,
        )

    if name in S.AWS_EMBEDDINGS_MODELS:
        return BedrockEmbeddings(
            model_id=name,
        )

    if name in S.OLLAMA_EMBEDDINGS_MODELS:
        return OllamaEmbeddings(
            model=name,
        )

    raise ValueError(f"Embeddings model '{name}' non available.")
