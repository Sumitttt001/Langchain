import os

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage,SystemMessage

# Define the present directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(current_dir, "chroma_db_with_metadata")

# Define embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the users query
query = "where is the Dracula castle located?"

# Retrieve relevant documents from the vector store
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2}
)
relevant_docs = retriever.invoke(query)

# Display the relevant documents with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
    print(f"Metadata: {doc.metadata}\n")