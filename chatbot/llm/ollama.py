from langchain_ollama import ChatOllama

from chatbot.settings import S

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=S.TEMPERATURE,
)
