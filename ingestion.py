from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain_openai import ChatOpenAI
import os

load_dotenv()

pdf_files = [
    "data/EMI - Smart Invoice-FAQ V6.0.pdf",
    "data/USER_MANUAL_-_SMART_INVOICE_APP_v3.0.pdf"
]

# Load documents from PDF files
docs = []
for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())
    else:
        print(f"Warning: File {pdf_file} not found")

print(f"Loaded {len(docs)} document pages")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, 
    chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs)

print(f"Split into {len(doc_splits)} chunks")

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="smartfund-rag-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma"
# )

chroma_retriever = Chroma(
    collection_name="smartfund-rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create history-aware retriever with your Chroma retriever
rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
history_aware_retriever = create_history_aware_retriever(
    llm=chat, 
    retriever=chroma_retriever,  
    prompt=rephrase_prompt
)

print("Retriever loaded successfully")