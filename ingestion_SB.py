"""Document ingestion for Supabase PostgreSQL with pgvector."""

import os
import sys

import psycopg
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.benchmark_config import (
    SCALABILITY_DOC_COUNTS,
    build_scalability_collection_name,
    filter_chunks_for_doc_count,
)
from utils.cloud_vector_stores import create_supabase_engine
from utils.document_processor import DocumentProcessor
from utils.security import build_pg_connection_string, require_env

load_dotenv()

SUPABASE_STATEMENT_TIMEOUT_MS = int(os.getenv("SUPABASE_STATEMENT_TIMEOUT_MS", "600000"))
SUPABASE_DB_HOST = require_env("SUPABASE_DB_HOST")
SUPABASE_DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
SUPABASE_DB_USER = require_env("SUPABASE_DB_USER")
SUPABASE_DB_PASSWORD = require_env("SUPABASE_DB_PASSWORD")
SUPABASE_SSLMODE = os.getenv("SUPABASE_SSLMODE", "require")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")

connection_string = build_pg_connection_string(
    user=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASSWORD,
    host=SUPABASE_DB_HOST,
    port=SUPABASE_DB_PORT,
    database=SUPABASE_DB_NAME,
    sslmode=SUPABASE_SSLMODE,
)
engine = create_supabase_engine(connection_string, SUPABASE_STATEMENT_TIMEOUT_MS)

print("\n" + "=" * 80)
print("SUPABASE PGVECTOR DOCUMENT INGESTION")
print("=" * 80)

with psycopg.connect(
    host=SUPABASE_DB_HOST,
    port=SUPABASE_DB_PORT,
    dbname=SUPABASE_DB_NAME,
    user=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASSWORD,
    sslmode=SUPABASE_SSLMODE,
) as connection:
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
embedding_dimension = len(embeddings.embed_query("dimension check"))
chunks = DocumentProcessor().process_documents()

for chunk in chunks:
    chunk.page_content = chunk.page_content.replace("\x00", "")
    for key, value in chunk.metadata.items():
        if isinstance(value, str):
            chunk.metadata[key] = value.replace("\x00", "")


def recreate_collection(collection_name: str) -> PGVector:
    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )
    try:
        store.delete_collection()
    except Exception:
        pass
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )


def ingest_collection(collection_name: str, documents) -> None:
    print(f"\nResetting and ingesting Supabase collection '{collection_name}'...")
    store = recreate_collection(collection_name)
    ids = [f"{collection_name}_chunk_{index + 1:05d}" for index in range(len(documents))]
    batch_size = 50
    for start in range(0, len(documents), batch_size):
        stop = start + batch_size
        store.add_documents(documents=documents[start:stop], ids=ids[start:stop])
        print(f"  Batch {(start // batch_size) + 1}/{(len(documents) + batch_size - 1) // batch_size}")


ingest_collection(COLLECTION_NAME, chunks)
collection_sizes = {COLLECTION_NAME: len(chunks)}
for doc_count in SCALABILITY_DOC_COUNTS:
    subset_name = build_scalability_collection_name(COLLECTION_NAME, doc_count)
    subset_chunks = filter_chunks_for_doc_count(chunks, doc_count)
    ingest_collection(subset_name, subset_chunks)
    collection_sizes[subset_name] = len(subset_chunks)

print("\nEnsuring collection_id index for per-collection scans...")
with psycopg.connect(
    host=SUPABASE_DB_HOST,
    port=SUPABASE_DB_PORT,
    dbname=SUPABASE_DB_NAME,
    user=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASSWORD,
    sslmode=SUPABASE_SSLMODE,
) as connection:
    connection.execute(
        "CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_collection_id "
        "ON langchain_pg_embedding (collection_id)"
    )

print("\nSupabase ingestion completed")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Native vector dimension: {embedding_dimension}")
print("Search mode: exact pgvector (4096 dimensions exceed pgvector HNSW vector limits)")
for name, count in collection_sizes.items():
    print(f"  {name}: {count} chunks")
