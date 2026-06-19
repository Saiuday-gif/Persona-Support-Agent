import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_database():
    """
    Reads text documents from the 'data' directory, splits them into manageable chunks,
    generates embeddings, and stores them in a local Chroma vector database.
    """
    print("1. Loading documents from 'data' folder...")
    # Initialize the loader to read all .txt files present in the 'data' folder
    loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Check if the folder is empty
    if not documents:
        print("Error: No documents found in the 'data' folder. Please add some text files.")
        return None

    print(f"Loaded {len(documents)} document(s).")

    print("2. Splitting documents into smaller chunks...")
    # Split the document text into chunks of 500 characters
    # A 50-character overlap ensures context isn't lost between consecutive chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("3. Initializing Local Embedding Model...")
    # Load a free, open-source HuggingFace model to convert text chunks into numerical vectors (embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("4. Storing chunks into ChromaDB...")
    # Create the vector database and persist (save) it locally in the 'chroma_db' directory
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("✅ Vector Database setup is complete!")
    return vector_db

# Standard Python idiom to execute the function when the script is run directly
if __name__ == "__main__":
    build_vector_database()