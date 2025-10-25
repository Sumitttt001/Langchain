import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Define the Persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
enbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing vector store with the existing function
db = Chroma( persist_directory=persistent_directory, embedding_function=enbeddings)

# Define the user question
query = "where does gandalf meet frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3 , "score_threshold": 0.5},  
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"source: {doc.metadata.get('source', 'Unknown')}\n")