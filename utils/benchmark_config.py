"""Benchmark configuration helpers for fixed corpus-size experiments."""

from langchain_core.documents import Document

from .document_processor import ANSWERABLE_QUERIES, BENCHMARK_DOCUMENTS, QUERY_METADATA


TARGET_SCALABILITY_DOC_COUNTS = [20, 40, 60, 80, 100]
SCALABILITY_DOC_COUNTS = [
    doc_count for doc_count in TARGET_SCALABILITY_DOC_COUNTS if doc_count <= len(BENCHMARK_DOCUMENTS)
]
if len(BENCHMARK_DOCUMENTS) not in SCALABILITY_DOC_COUNTS:
    SCALABILITY_DOC_COUNTS.append(len(BENCHMARK_DOCUMENTS))
SCALABILITY_DOC_COUNTS = sorted(set(SCALABILITY_DOC_COUNTS))

SCALABILITY_QUERY_DOC_COUNT = min(20, len(BENCHMARK_DOCUMENTS))
CONCURRENT_USER_LEVELS = [1, 3, 5]
CONCURRENT_QUERIES_PER_USER = 3
FIREBASE_VECTOR_DIMENSION = 2048
SQLITE_DB_PATH = "sqlite_vectors.db"
LANCEDB_PATH = "lancedb_data"
QDRANT_URL = "http://localhost:6333"


def get_source_files_for_doc_count(doc_count: int) -> set[str]:
    return {doc["file"] for doc in BENCHMARK_DOCUMENTS[:doc_count]}


def get_answerable_queries_for_doc_count(doc_count: int) -> list[str]:
    source_files = get_source_files_for_doc_count(doc_count)
    return [
        query
        for query in ANSWERABLE_QUERIES
        if QUERY_METADATA[query]["source_file"] in source_files
    ]


def get_scalability_query_set() -> list[str]:
    return get_answerable_queries_for_doc_count(SCALABILITY_QUERY_DOC_COUNT)


def filter_chunks_for_doc_count(chunks: list[Document], doc_count: int) -> list[Document]:
    source_files = get_source_files_for_doc_count(doc_count)
    return [chunk for chunk in chunks if chunk.metadata.get("source_file") in source_files]


def build_scalability_collection_name(base_collection_name: str, doc_count: int) -> str:
    return f"{base_collection_name}_docs_{doc_count:02d}"


def build_scalability_namespace(doc_count: int) -> str:
    return f"docs_{doc_count:02d}"
