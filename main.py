import os

# silence tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ========= CONFIG =========
load_dotenv()

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
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks")
    return chunks

def create_embeddings():
    print("Inside the create_embeddings method")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    # )
    return embeddings

def build_vector_store(embeddings, chunks):
    print("Inside the build_vector_store method")
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store built successfully")
    return vector_db

def build_llm():
    """
    Uses Groq API (LLaMA 3.1) as the chat LLM.
    Requires GROQ_API_KEY in .env.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
    )
    return llm

def setup_rag_pipeline(vector_db):
    print("Inside the setup_rag_pipeline method")
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = build_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
    return qa_chain


def ask_and_print(qa_chain, question: str, chat_history: list):
    print("\n" + "=" * 80)
    print("Question:", question)

    # Call the RAG chain with just the query.
    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    sources = result["source_documents"]

    print("\nAnswer:\n")
    print(answer)

    print("\nSources used:")
    for i, doc in enumerate(sources, start=1):
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "N/A")
        print(f"- Source {i}: file={source}, page={page}")

    # Update conversation history (maintained in Python)
    chat_history.append((question, answer))

def format_history(chat_history):
    if not chat_history:
        return "No previous conversation yet."
    lines = []
    for i, (q, a) in enumerate(chat_history, start=1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a}")
    return "\n".join(lines)


if __name__ == "__main__":
    
    file_path = "data/AI.pdf"

    #check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    # 1. Load documents 
    docs = load_docs(file_path)

    # 2. Chunking
    chunks = split_docs(docs)

    # 3. Embedding
    embeddings = create_embeddings()

    # 4. Build FAISS vector DB
    vector_db = build_vector_store(embeddings, chunks)
    # print(vector_db.index.ntotal)

    # 5. Build a RAG pipeline
    qa_chain = setup_rag_pipeline(vector_db)

    chat_history = []

    questions = [
        "Summarize this document.",
        "What are the key findings of this document?",
        "What is the definition of SCADA Systems as per the document?",
    ]

    for q in questions:
        ask_and_print(qa_chain, q, chat_history)
    
    # print("\n" + "=" * 80)
    # print("Conversation history:\n")
    # print(format_history(chat_history))
