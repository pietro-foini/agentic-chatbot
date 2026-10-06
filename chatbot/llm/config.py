from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from chatbot.settings import S

CATEGORIZED_MODELS = {
    "Proprietary": {
        model_name: ChatGoogleGenerativeAI(
            model=model_name,
            project=S.PROJECT_ID,
            location=region,
            vertexai=True,
            temperature=S.TEMPERATURE,
            max_tokens=S.MAX_TOKENS,
            max_retries=1,
            thinking_budget=S.THINKING_BUDGET,
            timeout=S.TIMEOUT_LLM,
            include_thoughts=False,
        )
        for model_name, region in S.GOOGLE_MODELS.items()
    },
    "Open-Source": {
        model_name: ChatOllama(
            model=model_name,
            temperature=S.TEMPERATURE,
        )
        for model_name in S.OLLAMA_MODELS
    },
}

FLAT_REGISTRY = {
    model_name: model for category in CATEGORIZED_MODELS.values() for model_name, model in category.items()
}


def get_model(model_name: str):
    """Retrieve the model instance by ID with fallback"""

    return FLAT_REGISTRY.get(model_name, FLAT_REGISTRY[S.MODEL_NAME])
