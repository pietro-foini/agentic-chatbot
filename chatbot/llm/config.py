from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from chatbot.settings import S

MODELS = {
    **{
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
    **{
        model_name: ChatOllama(
            model=model_name,
            temperature=S.TEMPERATURE,
        )
        for model_name in S.OLLAMA_MODELS
    },
}
