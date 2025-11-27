import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_docs(file_path):
    print("Inside the load_docs method")
    
    # Determine loader based on file extension
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .pdf or .txt")
    
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {file_path}")
    return docs

def split_docs(docs, chunk_size=800, chunk_overlap=200):
    print("Inside the split_docs method")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks")
    return chunks



## Project entry point
if __name__ == "__main__":
    
    # Path to your research paper PDF (EEG-VLM)
    file_path = "data/EEG-VLM.pdf"

    #check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    # 1. Load documents 
    docs = load_docs(file_path)
    # print(docs)

    # 2. Chunking
    chunks = split_docs(docs)
    print(chunks)