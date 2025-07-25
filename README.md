# Pdf-Reader

Pdf-Reader is a Streamlit web application that allows you to upload multiple PDF files, process their content, and interact with them using a conversational AI. The app leverages state-of-the-art open-source language models and embeddings for efficient document understanding and retrieval.

## Features

- 📄 Upload and process multiple PDF files at once
- 💬 Ask questions about your PDFs and get instant, context-aware answers
- ⚡ Fast and accurate retrieval using BAAI/bge-base-en-v1.5 embeddings
- 🤖 Powered by open-source LLMs (e.g., Mistral-7B-Instruct)
- 🖥️ Clean, chat-style user interface

## Tech Stack

- [Streamlit](https://streamlit.io/) for the web UI
- [LangChain](https://python.langchain.com/) for chaining and retrieval
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF parsing
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage
- [Hugging Face models](https://huggingface.co/) for embeddings and LLMs

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/pdf_reader.git
    cd pdf_reader
    ```

2. **Create and activate a virtual environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**

    Create a `.env` file in the project root and add any required API keys (if using OpenAI or Hugging Face endpoints):

    ```
    # Example for OpenAI (if used)
    OPENAI_API_KEY=your_openai_api_key
    ```

5. **Run the app**
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the app in your browser (Streamlit will provide a local URL).
2. Upload one or more PDF files using the sidebar.
3. Click "Process" to extract and index the content.
4. Ask questions about your documents in the chat box.
5. View answers and chat history in a conversational format.

## Customization

- **Embeddings Model:** Change the model in `get_vector_store()` in `app.py`.
- **LLM Model:** Change the model in `get_conversation_chain()` in `app.py`.
