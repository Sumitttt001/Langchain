from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

PROJECT_ID = "langchain-f4498"
SESSION_ID = "chat_session_1"
COLLECTION_NAME = "chat_history"

print ("Initializing Firestore client...")
client = firestore.Client(project=PROJECT_ID)

print ("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    client=client,
    session_id=SESSION_ID,
    collection_name=COLLECTION_NAME
)
print ("chat_history initialized.")
print ("current chat history:", chat_history.messages)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

print ("start chatting with the AI. Type 'exit' to end the conversation.")

while True:
    query = input("user: ")
    if query.lower() == "exit":
        break

    chat_history.add_user_message(query)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")