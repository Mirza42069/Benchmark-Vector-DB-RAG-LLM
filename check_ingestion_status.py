"""Check current vector-store ingestion counts."""

from __future__ import annotations

import os
import sqlite3

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone
from qdrant_client import QdrantClient

from utils.benchmark_config import LANCEDB_PATH, QDRANT_URL, SQLITE_DB_PATH


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
DOC_COUNTS = [20, 40, 60, 80, 100]
DISPLAY_NAMES = [COLLECTION_NAME] + [f"{COLLECTION_NAME}_docs_{doc_count:02d}" for doc_count in DOC_COUNTS]
VALIDATION_QUERY = "hnsw vector retrieval validation query"


def get_embeddings() -> OllamaEmbeddings | None:
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        print("  EMBEDDING_MODEL is not set; skipping retrieval checks")
        return None
    return OllamaEmbeddings(model=embedding_model)


def normalize_sqlite_table_name(table_name: str) -> str:
    return table_name.removeprefix("vectors_")


def normalize_pinecone_namespace(namespace: str) -> str:
    if not namespace:
        return COLLECTION_NAME
    if namespace.startswith("docs_"):
        return f"{COLLECTION_NAME}_{namespace}"
    return namespace


def print_counts(counts: dict[str, int], retrieval_status: dict[str, str] | None = None) -> None:
    found = False
    for name in DISPLAY_NAMES:
        if name in counts:
            status = f" | retrieval: {retrieval_status[name]}" if retrieval_status and name in retrieval_status else ""
            print(f"  {name}: {counts[name]}{status}")
            found = True
    if not found:
        print("  No target collections found")

    missing = [name for name in DISPLAY_NAMES if name not in counts]
    if missing:
        print(f"  Missing target collections: {', '.join(missing)}")


def retrieval_label(fn) -> str:
    try:
        results = fn()
        return f"OK ({len(results)} docs)" if results else "FAIL (0 docs)"
    except Exception as e:
        return f"FAIL ({e})"


def print_sqlite_counts() -> None:
    print("\nSQLite")
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"  Missing database: {SQLITE_DB_PATH}")
        return

    with sqlite3.connect(SQLITE_DB_PATH) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vectors_%' ORDER BY name"
        ).fetchall()
        if not tables:
            print("  No vector tables found")
            return
        counts = {}
        for (table_name,) in tables:
            count = connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            counts[normalize_sqlite_table_name(table_name)] = count
        embeddings = get_embeddings()
        retrieval_status = None
        if embeddings:
            from utils.local_vector_stores import SQLiteVectorStore

            retrieval_status = {
                name: retrieval_label(
                    lambda name=name: SQLiteVectorStore(SQLITE_DB_PATH, name, embeddings).similarity_search(
                        VALIDATION_QUERY, k=1
                    )
                )
                for name in DISPLAY_NAMES
                if name in counts
            }
        print_counts(counts, retrieval_status)


def print_chroma_counts() -> None:
    print("\nChromaDB")
    if not os.path.exists("chroma_db"):
        print("  Missing directory: chroma_db")
        return

    try:
        client = chromadb.PersistentClient(path="chroma_db")
        collections = client.list_collections()
        if not collections:
            print("  No collections found")
            return
        counts = {collection.name: collection.count() for collection in collections}
        embeddings = get_embeddings()
        retrieval_status = None
        if embeddings:
            retrieval_status = {
                name: retrieval_label(
                    lambda name=name: Chroma(
                        client=client,
                        collection_name=name,
                        embedding_function=embeddings,
                    ).similarity_search(VALIDATION_QUERY, k=1)
                )
                for name in DISPLAY_NAMES
                if name in counts
            }
        print_counts(counts, retrieval_status)
    except Exception as e:
        print(f"  Error: {e}")


def print_pinecone_counts() -> None:
    print("\nPinecone")
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-vector-benchmark")
    if not api_key:
        print("  PINECONE_API_KEY is not set")
        return

    try:
        pc = Pinecone(api_key=api_key)
        indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name not in indexes:
            print(f"  Missing index: {index_name}")
            return
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        counts = {
            normalize_pinecone_namespace(namespace): namespace_stats.get("vector_count", 0)
            for namespace, namespace_stats in stats.get("namespaces", {}).items()
        }
        embeddings = get_embeddings()
        retrieval_status = None
        if embeddings:
            from langchain_pinecone import PineconeVectorStore

            retrieval_status = {}
            for name in DISPLAY_NAMES:
                if name not in counts:
                    continue
                namespace = None if name == COLLECTION_NAME else name.removeprefix(f"{COLLECTION_NAME}_")
                retrieval_status[name] = retrieval_label(
                    lambda namespace=namespace: PineconeVectorStore(
                        index=index,
                        embedding=embeddings,
                        **({"namespace": namespace} if namespace else {}),
                    ).similarity_search(VALIDATION_QUERY, k=1)
                )
        print_counts(counts, retrieval_status)
    except Exception as e:
        print(f"  Error: {e}")


def print_postgres_counts() -> None:
    print("\nPostgreSQL")
    try:
        import psycopg
    except Exception as e:
        print(f"  psycopg import error: {e}")
        return

    password = os.getenv("DB_PASSWORD")
    if not password:
        print("  DB_PASSWORD is not set")
        return

    connection_kwargs = {
        "user": os.getenv("DB_USER", "raguser"),
        "password": password,
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "ragdb"),
    }
    try:
        with psycopg.connect(**connection_kwargs) as connection:
            rows = connection.execute(
                """
                SELECT c.name, COUNT(e.id)
                FROM langchain_pg_collection c
                LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                GROUP BY c.name
                ORDER BY c.name
                """
            ).fetchall()
            if not rows:
                print("  No collections found")
                return
            counts = {name: count for name, count in rows}
            embeddings = get_embeddings()
            retrieval_status = None
            if embeddings:
                from langchain_postgres import PGVector
                from utils.security import build_pg_connection_string

                connection_string = build_pg_connection_string(
                    user=connection_kwargs["user"],
                    password=connection_kwargs["password"],
                    host=connection_kwargs["host"],
                    port=connection_kwargs["port"],
                    database=connection_kwargs["dbname"],
                )
                retrieval_status = {
                    name: retrieval_label(
                        lambda name=name: PGVector(
                            embeddings=embeddings,
                            collection_name=name,
                            connection=connection_string,
                            use_jsonb=True,
                        ).similarity_search(VALIDATION_QUERY, k=1)
                    )
                    for name in DISPLAY_NAMES
                    if name in counts
                }
            print_counts(counts, retrieval_status)
    except Exception as e:
        print(f"  Error: {e}")


def print_lancedb_counts() -> None:
    print("\nLanceDB")
    if not os.path.exists(LANCEDB_PATH):
        print(f"  Missing directory: {LANCEDB_PATH}")
        return

    try:
        import lancedb

        db = lancedb.connect(LANCEDB_PATH)
        raw_tables = db.list_tables()
        raw_tables = raw_tables.tables if hasattr(raw_tables, "tables") else raw_tables
        existing_tables = {
            table["name"] if isinstance(table, dict) else table.name if hasattr(table, "name") else str(table)
            for table in raw_tables
        }
        counts = {}
        for table_name in existing_tables:
            if table_name in DISPLAY_NAMES:
                counts[table_name] = db.open_table(table_name).count_rows()
        embeddings = get_embeddings()
        retrieval_status = None
        if embeddings:
            from utils.local_vector_stores import LanceDBVectorStore

            retrieval_status = {
                name: retrieval_label(
                    lambda name=name: LanceDBVectorStore(LANCEDB_PATH, name, embeddings).similarity_search(
                        VALIDATION_QUERY, k=1
                    )
                )
                for name in DISPLAY_NAMES
                if name in counts
            }
        print_counts(counts, retrieval_status)
    except Exception as e:
        print(f"  Error: {e}")


def print_qdrant_counts() -> None:
    print("\nQdrant")
    try:
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections().collections
        if not collections:
            print("  No collections found")
            return
        counts = {}
        for collection in collections:
            info = client.get_collection(collection.name)
            counts[collection.name] = info.points_count
        embeddings = get_embeddings()
        retrieval_status = None
        if embeddings:
            from langchain_qdrant import QdrantVectorStore

            retrieval_status = {
                name: retrieval_label(
                    lambda name=name: QdrantVectorStore(
                        client=client,
                        collection_name=name,
                        embedding=embeddings,
                    ).similarity_search(VALIDATION_QUERY, k=1)
                )
                for name in DISPLAY_NAMES
                if name in counts
            }
        print_counts(counts, retrieval_status)
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    load_dotenv()
    print_sqlite_counts()
    print_chroma_counts()
    print_pinecone_counts()
    print_postgres_counts()
    print_lancedb_counts()
    print_qdrant_counts()
