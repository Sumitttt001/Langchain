from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

messages = [
    SystemMessage("you are expert in building rag,agent based applications"),
    HumanMessage("build a rag application using langchain and google genai"),
]

result = llm.invoke(messages)

print (result.content)