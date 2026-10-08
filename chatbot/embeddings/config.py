from langchain_ollama import OllamaEmbeddings

from chatbot.settings import S

embeddings = OllamaEmbeddings(
    model=S.EMBEDDINGS_NAME,
)
