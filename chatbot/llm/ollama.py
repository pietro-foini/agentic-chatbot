from langchain_ollama import ChatOllama

from chatbot.settings import S

llm = ChatOllama(
    model="llama3.1",
    temperature=S.TEMPERATURE,
)
