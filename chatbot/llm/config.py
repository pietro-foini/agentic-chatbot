from langchain_ollama import ChatOllama

from chatbot.settings import S

MODELS = {
    **{
        model_name: ChatOllama(
            model=model_name,
            temperature=S.TEMPERATURE,
        )
        for model_name in S.OLLAMA_MODELS
    },
}
