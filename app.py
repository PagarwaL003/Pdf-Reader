import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain.chat_models import ChatOpenAI


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts = chunks , embeddings = embeddings)
    return vector_store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True)
    conversation_chain = conversational_retrieval.from_llm(
        llm=llm,
        memory=memory,
        retriever = vectorstore.as_retriever()
    )
    return conversation_chain
    

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs" , page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask question about your documents")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'" , accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):    
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                 
                #get text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vector_store = get_vector_store(text_chunks)
                
                #create conversation chain
                conversation = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()