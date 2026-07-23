"""Document ingestion for Cloud Firestore vector search."""

import os
import sys
import time

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, RetryError
from langchain_ollama import OllamaEmbeddings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.benchmark_config import (
    FIREBASE_VECTOR_DIMENSION,
    SCALABILITY_DOC_COUNTS,
    build_scalability_collection_name,
    filter_chunks_for_doc_count,
)
from utils.cloud_vector_stores import FirebaseVectorStore, TruncatedNormalizedEmbeddings
from utils.document_processor import DocumentProcessor
from utils.security import require_env

load_dotenv()

FIREBASE_PROJECT_ID = require_env("FIREBASE_PROJECT_ID")
FIREBASE_DATABASE_ID = os.getenv("FIREBASE_DATABASE_ID", "(default)")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")
FIREBASE_RESET_COLLECTIONS = os.getenv("FIREBASE_RESET_COLLECTIONS", "false").lower() == "true"
FIREBASE_WRITE_DELAY_SECONDS = float(os.getenv("FIREBASE_WRITE_DELAY_SECONDS", "0.25"))

print("\n" + "=" * 80)
print("FIREBASE FIRESTORE DOCUMENT INGESTION")
print("=" * 80)

base_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
embeddings = TruncatedNormalizedEmbeddings(base_embeddings, FIREBASE_VECTOR_DIMENSION)
chunks = DocumentProcessor().process_documents()

for chunk in chunks:
    chunk.page_content = chunk.page_content.replace("\x00", "")
    for key, value in chunk.metadata.items():
        if isinstance(value, str):
            chunk.metadata[key] = value.replace("\x00", "")


def create_store(collection_name: str) -> FirebaseVectorStore:
    return FirebaseVectorStore(
        project_id=FIREBASE_PROJECT_ID,
        database_id=FIREBASE_DATABASE_ID,
        collection_name=collection_name,
        embedding=embeddings,
    )


class FirebaseQuotaExhausted(RuntimeError):
    """Raised when Firestore's daily write quota is exhausted."""


def ingest_collection(collection_name: str, documents) -> None:
    print(f"\nChecking Firestore collection '{collection_name}'...")
    store = create_store(collection_name)
    if FIREBASE_RESET_COLLECTIONS:
        print("  FIREBASE_RESET_COLLECTIONS=true: deleting existing documents...")
        store.reset()

    existing_count = store.count()
    expected_count = len(documents)
    if existing_count > expected_count:
        raise RuntimeError(
            f"Firestore collection '{collection_name}' has {existing_count} rows; "
            f"expected at most {expected_count}. Set FIREBASE_RESET_COLLECTIONS=true to rebuild it."
        )
    if existing_count == expected_count:
        print(f"  Complete: {existing_count}/{expected_count} chunks. Skipping.")
        return

    if existing_count:
        last_existing_id = f"chunk_{existing_count:05d}"
        first_missing_id = f"chunk_{existing_count + 1:05d}"
        if not store.collection.document(last_existing_id).get().exists:
            raise RuntimeError(
                f"Collection '{collection_name}' is not a contiguous checkpoint at {last_existing_id}. "
                "Set FIREBASE_RESET_COLLECTIONS=true to rebuild it."
            )
        if store.collection.document(first_missing_id).get().exists:
            raise RuntimeError(
                f"Collection '{collection_name}' has non-contiguous IDs near {first_missing_id}. "
                "Set FIREBASE_RESET_COLLECTIONS=true to rebuild it."
            )

    print(f"  Resuming at chunk {existing_count + 1}/{expected_count}.")
    ids = [f"chunk_{index + 1:05d}" for index in range(len(documents))]
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for start in range(existing_count, len(documents), batch_size):
        stop = start + batch_size
        try:
            store.add_documents(documents=documents[start:stop], ids=ids[start:stop])
        except (ResourceExhausted, RetryError) as error:
            saved_count = store.count()
            raise FirebaseQuotaExhausted(
                f"Firestore daily write quota reached while ingesting '{collection_name}'. "
                f"Checkpoint saved at {saved_count}/{expected_count} chunks."
            ) from error
        print(f"  Batch {(start // batch_size) + 1}/{total_batches}")
        if FIREBASE_WRITE_DELAY_SECONDS > 0:
            time.sleep(FIREBASE_WRITE_DELAY_SECONDS)
    stored_count = store.count()
    if stored_count != len(documents):
        raise RuntimeError(
            f"Firestore collection '{collection_name}' has {stored_count} rows; expected {len(documents)}"
        )


collection_sizes = {COLLECTION_NAME: len(chunks)}
try:
    ingest_collection(COLLECTION_NAME, chunks)
    for doc_count in SCALABILITY_DOC_COUNTS:
        subset_name = build_scalability_collection_name(COLLECTION_NAME, doc_count)
        subset_chunks = filter_chunks_for_doc_count(chunks, doc_count)
        collection_sizes[subset_name] = len(subset_chunks)
        ingest_collection(subset_name, subset_chunks)
except FirebaseQuotaExhausted as error:
    print("\n" + "=" * 80)
    print("FIREBASE FREE-TIER WRITE QUOTA REACHED")
    print("=" * 80)
    print(str(error))
    print("All completed batches are preserved.")
    print("Rerun 'python ingestion_FB.py' after the daily quota resets.")
    print("The script will skip completed collections and resume automatically.")
    sys.exit(2)

print("\nFirebase ingestion completed")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Native vector dimension: {embeddings.source_dimension}")
print(f"Stored Firestore vector dimension: {embeddings.target_dimension}")
print("Transform: first 2048 dimensions, then L2 normalization")
for name, count in collection_sizes.items():
    print(f"  {name}: {count} chunks")

print("\nCreate one Firestore vector index for each collection before benchmarking:")
for name in collection_sizes:
    print(
        "gcloud firestore indexes composite create "
        f"--project={FIREBASE_PROJECT_ID} --database={FIREBASE_DATABASE_ID} "
        f"--collection-group={name} --query-scope=COLLECTION "
        "--field-config=field-path=embedding,vector-config='"
        f'{{"dimension":"{FIREBASE_VECTOR_DIMENSION}","flat":"{{}}"}}' "'"
    )
