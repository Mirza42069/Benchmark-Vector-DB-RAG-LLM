# import basics
import os
from dotenv import load_dotenv

# import chromadb
import chromadb
from chromadb.config import Settings

# import langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize ChromaDB client (local persistent storage)
CHROMA_PATH = "chroma_db"  # Local directory to store ChromaDB data
COLLECTION_NAME = "its_guidebook"  # Collection name

# Create ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Initialize embeddings model (using Ollama)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Check if collection exists, delete if it does (for fresh ingestion)
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except:
    print(f"No existing collection found. Creating new one: {COLLECTION_NAME}")

# Initialize vector store with ChromaDB
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

print("Loading PDF documents...")
# loading the PDF document
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} documents")

print("Splitting documents into chunks...")
# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
documents = text_splitter.split_documents(raw_documents)
print(f"Created {len(documents)} chunks")

# generate unique id's
uuids = [f"id{i+1}" for i in range(len(documents))]

print("Adding documents to ChromaDB...")
# add to database
vector_store.add_documents(documents=documents, ids=uuids)

print(f"✅ Successfully ingested {len(documents)} document chunks into ChromaDB!")
print(f"📁 ChromaDB stored at: {CHROMA_PATH}")