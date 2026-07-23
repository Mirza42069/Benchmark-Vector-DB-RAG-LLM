"""
Document Ingestion for Pinecone Vector Database
"""

import os
import time
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.benchmark_config import (
    SCALABILITY_DOC_COUNTS,
    build_scalability_namespace,
    filter_chunks_for_doc_count,
)
from utils.document_processor import DocumentProcessor
from utils.security import require_env

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-vector-benchmark")
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")

print("\n" + "="*80)
print("📚 PINECONE DOCUMENT INGESTION")
print("="*80)

# Validate API key
if not PINECONE_API_KEY:
    print("\n❌ Error: PINECONE_API_KEY not found in .env file!")
    print("Please add your Pinecone API key to .env file")
    sys.exit(1)

# Initialize embeddings FIRST (for fair benchmark comparison)
print(f"\n🤖 Initializing embedding model: {EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

print("\n📏 Detecting embedding dimension...")
try:
    EMBEDDING_DIMENSION = len(embeddings.embed_query("dimension check"))
    print(f"   ✓ Detected dimension: {EMBEDDING_DIMENSION}")
except Exception as e:
    print(f"\n❌ Error detecting embedding dimension: {str(e)}")
    sys.exit(1)

# Initialize Pinecone
print(f"\n🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists and delete if wrong dimension
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME in existing_indexes:
    # Check dimension of existing index
    existing_index_info = pc.describe_index(INDEX_NAME)
    existing_dim = existing_index_info.dimension
    if existing_dim != EMBEDDING_DIMENSION:
        print(f"\n⚠️  Existing index has dimension {existing_dim}, need {EMBEDDING_DIMENSION}")
        print(f"   🗑️  Deleting old index...")
        pc.delete_index(INDEX_NAME)
        print(f"   ✓ Old index deleted")
        time.sleep(2)  # Wait for deletion to complete
        existing_indexes = []  # Force recreation

if INDEX_NAME not in existing_indexes:
    print(f"\n🆕 Creating new index: {INDEX_NAME}")
    print(f"   Embedding dimension: {EMBEDDING_DIMENSION} ({EMBEDDING_MODEL})")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("   ⏳ Waiting for index to be ready...")
        time.sleep(1)
    print(f"   ✓ Index created successfully")
else:
    print(f"\n✅ Using existing index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# Check if index already has data
stats = index.describe_index_stats()
existing_count = stats.total_vector_count

if existing_count > 0:
    print(f"\n⚠️  Index '{INDEX_NAME}' has {existing_count} existing vectors.")
    print("   🗑️  Clearing for fresh ingestion...")

# Clear existing vectors for fresh start
print("\n🗑️  Clearing existing vectors...")
try:
    index.delete(delete_all=True)
    print("   ✓ Cleared existing data")
except:
    print("   ℹ️  No existing data to clear")

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

def ingest_store(label: str, vector_store: PineconeVectorStore, documents):
    uuids = [f"chunk_{i+1:05d}" for i in range(len(documents))]
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"\n💾 Adding documents to {label}...")
    print(f"   Processing {len(documents)} chunks in {total_batches} batches...")

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = uuids[i:i + batch_size]
        current_batch = (i // batch_size) + 1

        try:
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"   ✓ Batch {current_batch}/{total_batches} completed")
        except Exception as e:
            print(f"   ✗ Error in batch {current_batch}: {str(e)}")


print("\n" + "="*80)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
ingest_store("Pinecone", vector_store, chunks)

scalability_namespace_sizes = {}
for doc_count in SCALABILITY_DOC_COUNTS:
    namespace = build_scalability_namespace(doc_count)
    subset_chunks = filter_chunks_for_doc_count(chunks, doc_count)

    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass

    subset_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    ingest_store(f"Pinecone namespace '{namespace}'", subset_store, subset_chunks)
    scalability_namespace_sizes[namespace] = len(subset_chunks)

# Verify ingestion
print("\n🔍 Verifying ingestion...")
stats = index.describe_index_stats()
stored_count = stats.total_vector_count

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
print(f"   • Vectors stored in Pinecone: {stored_count}")
print(f"   • Index name: {INDEX_NAME}")
print(f"   • Embedding model: {EMBEDDING_MODEL}")
print(f"   • Vector dimension: {EMBEDDING_DIMENSION}")
print(f"\n📈 Scalability Namespaces:")
for namespace, chunk_count in scalability_namespace_sizes.items():
    print(f"   • {namespace}: {chunk_count} chunks")
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
print("✨ Ready to use! Run: streamlit run Benchmark.py")
print("="*80)
