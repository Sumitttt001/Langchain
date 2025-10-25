import os

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Define the directory containing the text file and the persist directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(curr_dir, "documents")
db_directory = os.path.join(curr_dir, "db")
persistent_directory = os.path.join(curr_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist, initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The directory {books_dir} does not exist. Please check the path.")

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document
            doc.metadata = {"source" : book_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)      
    # Display information about the split documents
    print("\n--- Document chunks information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
else:
    print("Vector store already exists. No need to initialize.")
