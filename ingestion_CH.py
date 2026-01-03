"""
Document Ingestion for ChromaDB
"""

import os
import sys
from dotenv import load_dotenv
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.document_processor import DocumentProcessor

load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "its_guidebook"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))  # qwen3-embedding:8b=4096, mxbai=1024

print("\n" + "="*80)
print("📚 CHROMADB DOCUMENT INGESTION")
print("="*80)

# Initialize embeddings FIRST (for fair benchmark comparison)
print(f"\n🤖 Initializing embedding model: {EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialize ChromaDB client
print(f"\n🔌 Initializing ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Check if collection exists and has data
try:
    existing_collection = client.get_collection(name=COLLECTION_NAME)
    existing_count = existing_collection.count()
    
    if existing_count > 0:
        print(f"\n✅ Collection '{COLLECTION_NAME}' already exists with {existing_count} documents.")
        print("   ⏭️  Skipping ingestion. Delete 'chroma_db' folder to re-ingest.")
        
        # Initialize vector store for testing
        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        
        # Skip to test retrieval
        print("\n" + "="*80)
        print("🧪 Testing retrieval with sample queries...")
        print("-" * 80)
        
        test_queries = [
            ("🇮🇩", "Bagaimana cara mengubah password myITS Portal?"),
            ("🇬🇧", "What documents do I need to bring when arriving in Surabaya?"),
        ]
        
        for lang_flag, query in test_queries:
            print(f"\n{lang_flag} Testing: \"{query}\"")
            try:
                results = vector_store.similarity_search(query, k=3)
                if results:
                    print(f"   ✅ Found {len(results)} relevant chunks")
                else:
                    print("   ❌ No results found!")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        print("\n" + "="*80)
        print("✨ Ready to use! Run: streamlit run chatbot_chroma.py")
        print("="*80)
        sys.exit(0)
except Exception:
    pass  # Collection doesn't exist, proceed with ingestion

# Delete existing collection if exists (fresh start)
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"   🗑️  Deleted existing collection '{COLLECTION_NAME}'")
except:
    pass

# Initialize vector store
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)
print(f"   ✓ ChromaDB initialized")
print(f"   • Storage path: {CHROMA_PATH}")
print(f"   • Collection: {COLLECTION_NAME}")

# Process documents
print("\n" + "="*80)
processor = DocumentProcessor()
chunks = processor.process_documents()

# Clean documents to remove NUL bytes (for consistency with other databases)
print("\n🧹 Cleaning documents (removing NUL bytes)...")
for chunk in chunks:
    # Remove NUL bytes from content
    chunk.page_content = chunk.page_content.replace('\x00', '')
    
    # Clean metadata strings
    if chunk.metadata:
        for key, value in chunk.metadata.items():
            if isinstance(value, str):
                chunk.metadata[key] = value.replace('\x00', '')

print("   ✓ Documents cleaned")

# Add documents to ChromaDB
print("\n" + "="*80)
print("💾 Adding documents to ChromaDB...")

uuids = [f"chunk_{i+1:05d}" for i in range(len(chunks))]

batch_size = 50
total_batches = (len(chunks) + batch_size - 1) // batch_size

print(f"   Processing {len(chunks)} chunks in {total_batches} batches...")

for i in range(0, len(chunks), batch_size):
    batch_docs = chunks[i:i + batch_size]
    batch_ids = uuids[i:i + batch_size]
    current_batch = (i // batch_size) + 1
    
    try:
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"   ✓ Batch {current_batch}/{total_batches} completed")
    except Exception as e:
        print(f"   ✗ Error in batch {current_batch}: {str(e)}")

# Verify ingestion
print("\n🔍 Verifying ingestion...")
collection = client.get_collection(name=COLLECTION_NAME)
doc_count = collection.count()
print(f"   ✓ Documents stored: {doc_count}")

# Calculate language distribution
lang_distribution = {}
for chunk in chunks:
    lang = chunk.metadata.get('chunk_language', 'unknown')
    lang_distribution[lang] = lang_distribution.get(lang, 0) + 1

print("\n" + "="*80)
print("✅ INGESTION COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   • Total chunks created: {len(chunks)}")
print(f"   • Vectors stored: {doc_count}")
print(f"   • Storage path: {CHROMA_PATH}")
print(f"   • Collection: {COLLECTION_NAME}")
print(f"   • Embedding model: {EMBEDDING_MODEL}")
print(f"   • Vector dimension: {EMBEDDING_DIM}")
print(f"\n🌍 Language Distribution:")
for lang, count in sorted(lang_distribution.items()):
    lang_name = {"id": "🇮🇩 Indonesian", "en": "🇬🇧 English", "mixed": "🌍 Mixed"}.get(lang, f"❓ {lang}")
    percentage = (count / len(chunks)) * 100
    print(f"   • {lang_name}: {count} chunks ({percentage:.1f}%)")

# Test retrieval
print("\n" + "="*80)
print("🧪 Testing retrieval with sample queries...")
print("-" * 80)

test_queries = [
    ("🇮🇩", "Bagaimana cara mengubah password myITS Portal?"),
    ("🇬🇧", "What documents do I need to bring when arriving in Surabaya?"),
]

for lang_flag, query in test_queries:
    print(f"\n{lang_flag} Testing: \"{query}\"")
    
    try:
        results = vector_store.similarity_search(query, k=3)
        
        if results:
            print(f"   ✅ Found {len(results)} relevant chunks")
            for idx, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                chunk_lang = doc.metadata.get('chunk_language', '?')
                lang_emoji = {"id": "🇮🇩", "en": "🇬🇧", "mixed": "🌍"}.get(chunk_lang, "❓")
                preview = doc.page_content[:80].replace('\n', ' ')
                print(f"      {idx}. [{lang_emoji}] {source}: {preview}...")
        else:
            print("   ❌ No results found!")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

print("\n" + "="*80)
print("✨ Ready to use! Run: streamlit run chatbot_chroma.py")
print("="*80)