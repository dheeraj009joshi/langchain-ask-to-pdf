import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
openai_api_key = "sk-proj-aNNfNbMDDTZAw8jGTDnET3BlbkFJAquDOf8lrwHva61niNsd"  # Update your .env file with OPENAI_API_KEY
openai.api_key = openai_api_key

# Constants
collection_name = "faiss_index"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def construct_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(collection_name)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide incorrect information.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    # Using OpenAI's ChatGPT model (gpt-3.5-turbo)
    from langchain.chat_models import ChatOpenAI
    
    model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def load_collection(collectionName):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    collection = FAISS.load_local(collectionName, embeddings, allow_dangerous_deserialization=True)
    return collection

def process_input(user_question):
    collecion = load_collection(collection_name)
    docs = collecion.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using OpenAI 💁")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        process_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                construct_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
