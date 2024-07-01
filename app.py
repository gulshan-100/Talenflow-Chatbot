import streamlit as st
from langchain_community.llms import GooglePalm
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
#from langchain import Chroma  # Adjusted import
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Setting up API keys
os.environ['GOOGLE_API_KEY'] = "Gemini api"
os.environ['HUGGINGFACE_API_KEY'] = "Hugging face api"

# Initialize embeddings and LLM model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = GooglePalm(
    model = "gemini-pro"
)

# Streamlit header and input field for query
st.header("Talenflow Chatbot")
query = st.text_input("Enter your query:")

# Function to load documents from a text file
def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load()

dataset_path = "data.txt"
data = load_documents(dataset_path)

# Splitting documents using a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Creating a Chroma database from documents
db = Chroma.from_documents(all_splits, embeddings)

# Function to retrieve similar documents based on a query
def get_similar_docs(query, k=2):
    similar_docs = db.similarity_search(query, k=k)
    return similar_docs

# Loading QA chain for answering questions
chain = load_qa_chain(
    llm,
    chain_type="stuff"
)

# Function to get answer from the QA chain based on the query
def get_answer(query):
    relevant_docs = get_similar_docs(query)
    response = chain.run(input_documents=relevant_docs, question=query)
    return response

# Handling user interaction with a button press
if st.button("Enter"):
    answer = get_answer(query)
    st.write("Answer:", answer)
