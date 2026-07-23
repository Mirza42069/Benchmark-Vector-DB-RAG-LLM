"""
Document Ingestion for PostgreSQL with pgvector
"""

import os
import sys
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.benchmark_config import (
    SCALABILITY_DOC_COUNTS,
    build_scalability_collection_name,
    filter_chunks_for_doc_count,
)
from utils.document_processor import DocumentProcessor
from utils.security import build_pg_connection_string, require_env

load_dotenv()

# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = require_env("DB_PASSWORD")  # Required - no default for security
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")

# Create connection string with properly encoded credentials
connection_string = build_pg_connection_string(
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

print("\n" + "="*80)
print("📚 POSTGRESQL DOCUMENT INGESTION")
print("="*80)

# Initialize embeddings FIRST (for fair benchmark comparison)
print(f"\n🤖 Initializing embedding model: {EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
EMBEDDING_DIMENSION = len(embeddings.embed_query("dimension check"))

print(f"\n🔌 Connecting to PostgreSQL...")
try:
    PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=connection_string,
        use_jsonb=True,
    )
    print(f"   ✓ Connected successfully")
    print(f"   • Database: {DB_NAME}")
    print(f"   • Collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"\n❌ Error connecting to database: {str(e)}")
    print("\n💡 Did you run setup_postgresql.py first?")
    sys.exit(1)

# Process documents
print("\n" + "="*80)
processor = DocumentProcessor()
chunks = processor.process_documents()

# Clean documents to remove NUL bytes
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

def recreate_collection(collection_name: str) -> PGVector:
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )

    print(f"\n🗑️  Resetting collection '{collection_name}'...")
    try:
        vector_store.delete_collection()
    except Exception:
        pass

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )


def ingest_collection(collection_name: str, documents):
    vector_store = recreate_collection(collection_name)
    uuids = [f"{collection_name}_chunk_{i + 1:05d}" for i in range(len(documents))]
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"\n💾 Adding documents to PostgreSQL collection '{collection_name}'...")
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
            raise

    return vector_store


print("\n" + "="*80)
vector_store = ingest_collection(COLLECTION_NAME, chunks)

scalability_collection_sizes = {}
for doc_count in SCALABILITY_DOC_COUNTS:
    subset_collection = build_scalability_collection_name(COLLECTION_NAME, doc_count)
    subset_chunks = filter_chunks_for_doc_count(chunks, doc_count)
    ingest_collection(subset_collection, subset_chunks)
    scalability_collection_sizes[subset_collection] = len(subset_chunks)

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
print(f"   • Vectors stored: {len(chunks)}")
print(f"   • Database: {DB_NAME}")
print(f"   • Collection: {COLLECTION_NAME}")
print(f"   • Embedding model: {EMBEDDING_MODEL}")
print(f"   • Vector dimension: {EMBEDDING_DIMENSION}")
print(f"\n📈 Scalability Collections:")
for collection_name, chunk_count in scalability_collection_sizes.items():
    print(f"   • {collection_name}: {chunk_count} chunks")
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
