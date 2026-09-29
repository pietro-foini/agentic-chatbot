from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot.settings import S

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    project=S.PROJECT_ID,
    location=S.REGION,
    vertexai=True,
    temperature=S.TEMPERATURE,
    max_tokens=S.MAX_TOKENS,
    max_retries=1,
    thinking_budget=S.THINKING_BUDGET,
    timeout=S.TIMEOUT_LLM,
    include_thoughts=False,
)
