# ai_research_assistant_using_langchain

This project aims to build a Research Assistant that can answer questions from a set of documents (PDF/TXT), cite the sources, and maintain conversation history

## Features

- \*\*Document Loading:\*\* Supports PDF and TXT documents
- **Smart Splitting:** Uses `RecursiveCharacterTextSplitter` for context-aware chunking.
- **Vector Search:** Uses FAISS for efficient similarity search.
- **Source Citation:** Returns specific page numbers used to answer the question.

## Prerequisites

1. Python 3.10+
2. GROQ API Key

## Installation

1. **Create a Project Folder:**

   ```bash
    mkdir ai-research-assistant
    cd ai-research-assistant
    python -m venv langchain
    source langchain/bin/activate  # On Windows: langchain\Scripts\activate
   ```

2. **Install Dependencies from requirements.txt**
3. **Setup Environment Variables: Create a .env file in the root directory** and copy the content below
   GROQ_API_KEY=your_groq_api_key_here
4. **Run the script**
   python main.py

## Modification Required in Code

1. Path of the document
   Update the path | Rename the file
   _file_path = "data/EEG-VLM.pdf"._
   as per your needs

2. Change Questions as per your needs
   _questions = [
   "Summarize this document.",
   "What are the key findings of this document?",
   "What is the definition of EEG-VLM as per the document?",
   ]._
