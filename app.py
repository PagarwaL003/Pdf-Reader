import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from htmlTemplate import css, bot_template, user_template

# Load environment variables (Hugging Face token etc.)
load_dotenv()

# Step 1: Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


# Step 2: Split long text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Step 3: Create vector store using Hugging Face embeddings
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


# Step 4: Setup Conversational Chain 
def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="tiiuae/falcon-7b-instruct",  
        task="text-generation",
        model_kwargs={
            "temperature": 0.5,
            "max_new_tokens": 512
        }
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )

    return conversation_chain


# Step 5: Handle chat UI and response display
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF files first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# Streamlit App Entry Point
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                # PDF text extraction
                raw_text = get_pdf_text(pdf_docs)

                # Text chunking
                text_chunks = get_text_chunks(raw_text)

                # Embedding and vector store
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

            st.success("PDFs processed. You can now ask questions!")

        if st.button("Clear Chat"):
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.success("Chat history cleared.")

if __name__ == '__main__':
    main()
