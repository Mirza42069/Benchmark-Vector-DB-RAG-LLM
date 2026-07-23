"""Cloud vector-store helpers for Firebase benchmark integration."""

from __future__ import annotations

from typing import Any

import numpy as np
from google.api_core import exceptions as google_exceptions
from google.api_core import retry as google_retry
from langchain_core.documents import Document
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


def create_supabase_engine(connection_string: str, statement_timeout_ms: int) -> Engine:
    """Build an engine that raises statement_timeout on every new connection.

    Applied via a connect event rather than startup options because the
    Supavisor pooler does not reliably forward startup parameters.
    """
    engine = create_engine(connection_string)

    @event.listens_for(engine, "connect")
    def _set_statement_timeout(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
        cursor.close()

    return engine


FIRESTORE_WRITE_RETRY = google_retry.Retry(
    predicate=google_retry.if_exception_type(
        google_exceptions.Aborted,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError,
        google_exceptions.ServiceUnavailable,
    ),
    initial=1.0,
    maximum=30.0,
    multiplier=2.0,
    timeout=300.0,
)


class TruncatedNormalizedEmbeddings:
    """Reduce embeddings to a fixed dimension and restore unit length."""

    def __init__(self, embeddings: Any, target_dimension: int):
        self.embeddings = embeddings
        self.target_dimension = target_dimension
        dimension_probe = embeddings.embed_query("dimension check")
        self.source_dimension = len(dimension_probe)
        if self.source_dimension < target_dimension:
            raise ValueError(
                f"Embedding dimension {self.source_dimension} is smaller than target {target_dimension}"
            )

    def _transform(self, vector: list[float]) -> list[float]:
        reduced = np.asarray(vector[: self.target_dimension], dtype=np.float32)
        norm = np.linalg.norm(reduced)
        if norm > 0:
            reduced = reduced / norm
        return reduced.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._transform(vector) for vector in self.embeddings.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._transform(self.embeddings.embed_query(text))


class FirebaseVectorStore:
    """Minimal Cloud Firestore vector store with a LangChain-like interface."""

    def __init__(
        self,
        project_id: str,
        database_id: str,
        collection_name: str,
        embedding: TruncatedNormalizedEmbeddings,
    ):
        from google.cloud import firestore

        self.client = firestore.Client(project=project_id, database=database_id)
        self.collection_name = collection_name
        self.collection = self.client.collection(collection_name)
        self.embedding = embedding

    @property
    def source_dimension(self) -> int:
        return self.embedding.source_dimension

    @property
    def stored_dimension(self) -> int:
        return self.embedding.target_dimension

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        from google.cloud.firestore_v1.vector import Vector

        if not documents:
            return
        ids = ids or [f"chunk_{index + 1:05d}" for index in range(len(documents))]
        vectors = self.embedding.embed_documents([document.page_content for document in documents])
        batch = self.client.batch()
        for index, document in enumerate(documents):
            batch.set(
                self.collection.document(str(ids[index])),
                {
                    "text": document.page_content,
                    "metadata": document.metadata,
                    "embedding": Vector(vectors[index]),
                },
            )
        batch.commit(retry=FIRESTORE_WRITE_RETRY, timeout=120.0)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
        from google.cloud.firestore_v1.vector import Vector

        query_vector = Vector(self.embedding.embed_query(query))
        vector_query = self.collection.find_nearest(
            vector_field="embedding",
            query_vector=query_vector,
            distance_measure=DistanceMeasure.COSINE,
            limit=k,
        )
        return [
            Document(
                page_content=(snapshot.to_dict() or {}).get("text", ""),
                metadata=(snapshot.to_dict() or {}).get("metadata", {}),
            )
            for snapshot in vector_query.stream()
        ]

    def count(self) -> int:
        results = self.collection.count().get()
        for row in results:
            values = row if isinstance(row, (list, tuple)) else [row]
            for value in values:
                if hasattr(value, "value"):
                    return int(value.value)
        return 0

    def reset(self) -> None:
        while True:
            snapshots = list(self.collection.limit(400).stream())
            if not snapshots:
                return
            batch = self.client.batch()
            for snapshot in snapshots:
                batch.delete(snapshot.reference)
            batch.commit(retry=FIRESTORE_WRITE_RETRY, timeout=120.0)
