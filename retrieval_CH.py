# import basics
import os
from dotenv import load_dotenv

# import chromadb
import chromadb

# import langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Initialize ChromaDB client (local persistent storage)
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "its_guidebook"

# Create ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize vector store
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Check database status
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✅ ChromaDB Collection: {COLLECTION_NAME}")
    print(f"📚 Total documents: {collection.count()}")
    print("\n" + "="*50 + "\n")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("Please run ingestion.py first!")
    exit()

# Retrieval with different thresholds
print("Testing retrieval with different similarity thresholds:\n")

test_queries = [
    "Explain how to register a phone in Indonesian Airport?",
    "What documents do I need to bring when arriving in Surabaya?",
    "What is the tuition fee for undergraduate programs?",  # Not in document
]

for query in test_queries:
    print(f"🔍 Query: {query}")
    print("-" * 50)
    
    # Test with threshold 0.5
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    
    results = retriever.invoke(query)
    
    print(f"Results found: {len(results)}\n")
    
    if results:
        for i, res in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Content: {res.page_content[:200]}...")
            print(f"Metadata: {res.metadata}")
            print()
    else:
        print("❌ No relevant documents found for this query.\n")
    
    print("="*50 + "\n")