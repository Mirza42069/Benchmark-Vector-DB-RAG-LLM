"""
Document Ingestion for local SQLite vector benchmark backend.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.benchmark_config import (
    SCALABILITY_DOC_COUNTS,
    SQLITE_DB_PATH,
    build_scalability_collection_name,
    filter_chunks_for_doc_count,
)
from utils.document_processor import DocumentProcessor
from utils.local_vector_stores import SQLiteVectorStore
from utils.security import require_env

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")

print("\n" + "=" * 80)
print("SQLITE DOCUMENT INGESTION")
print("=" * 80)

print(f"\nInitializing embedding model: {EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
EMBEDDING_DIMENSION = len(embeddings.embed_query("dimension check"))

print("\n" + "=" * 80)
processor = DocumentProcessor()
chunks = processor.process_documents()

print("\nCleaning documents (removing NUL bytes)...")
for chunk in chunks:
    chunk.page_content = chunk.page_content.replace("\x00", "")
    if chunk.metadata:
        for key, value in chunk.metadata.items():
            if isinstance(value, str):
                chunk.metadata[key] = value.replace("\x00", "")
print("   Documents cleaned")


def ingest_collection(collection_name: str, documents):
    vector_store = SQLiteVectorStore(SQLITE_DB_PATH, collection_name, embeddings)
    vector_store.reset()
    uuids = [f"chunk_{i + 1:05d}" for i in range(len(documents))]
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size

    print(f"\nAdding documents to SQLite collection '{collection_name}'...")
    print(f"   Processing {len(documents)} chunks in {total_batches} batches...")

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = uuids[i:i + batch_size]
        current_batch = (i // batch_size) + 1
        try:
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"   Batch {current_batch}/{total_batches} completed")
        except Exception as e:
            print(f"   Error in batch {current_batch}: {str(e)}")

    return vector_store


print("\n" + "=" * 80)
vector_store = ingest_collection(COLLECTION_NAME, chunks)

scalability_collection_sizes = {}
for doc_count in SCALABILITY_DOC_COUNTS:
    subset_collection = build_scalability_collection_name(COLLECTION_NAME, doc_count)
    subset_chunks = filter_chunks_for_doc_count(chunks, doc_count)
    subset_store = ingest_collection(subset_collection, subset_chunks)
    scalability_collection_sizes[subset_collection] = subset_store.count()

print("\n" + "=" * 80)
print("SQLITE INGESTION COMPLETED")
print("=" * 80)
print(f"\nSummary:")
print(f"   Total chunks created: {len(chunks)}")
print(f"   Vectors stored: {vector_store.count()}")
print(f"   Database path: {SQLITE_DB_PATH}")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   Embedding model: {EMBEDDING_MODEL}")
print(f"   Vector dimension: {EMBEDDING_DIMENSION}")
print("\nScalability Collections:")
for collection_name, chunk_count in scalability_collection_sizes.items():
    print(f"   {collection_name}: {chunk_count} chunks")

print("\nReady to use! Run: streamlit run Benchmark.py")
