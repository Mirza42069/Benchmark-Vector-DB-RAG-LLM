"""Local SQLite and LanceDB vector-store wrappers for benchmark parity."""

from __future__ import annotations

import json
import os
import sqlite3
import numpy as np
from langchain_core.documents import Document


def _clean_collection_name(collection_name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in collection_name)


def _cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominator = matrix_norms * query_norm
    denominator[denominator == 0] = 1e-12
    return matrix @ query_vector / denominator


class SQLiteVectorStore:
    """Simple local vector search over SQLite-stored embeddings."""

    def __init__(self, db_path: str, collection_name: str, embedding):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding = embedding
        self._ensure_table()

    @property
    def table_name(self) -> str:
        return f"vectors_{_clean_collection_name(self.collection_name)}"

    def reset(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(f'DROP TABLE IF EXISTS "{self.table_name}"')
        self._ensure_table()

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        if not documents:
            return

        ids = ids or [f"chunk_{index + 1:05d}" for index in range(len(documents))]
        vectors = self.embedding.embed_documents([document.page_content for document in documents])
        rows = [
            (
                ids[index],
                document.page_content,
                json.dumps(document.metadata, ensure_ascii=False),
                np.asarray(vectors[index], dtype=np.float32).tobytes(),
            )
            for index, document in enumerate(documents)
        ]

        with sqlite3.connect(self.db_path) as connection:
            connection.executemany(
                f'INSERT OR REPLACE INTO "{self.table_name}" (id, text, metadata, embedding) VALUES (?, ?, ?, ?)',
                rows,
            )

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        query_vector = np.asarray(self.embedding.embed_query(query), dtype=np.float32)
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                f'SELECT text, metadata, embedding FROM "{self.table_name}"'
            ).fetchall()

        if not rows:
            return []

        vectors = np.vstack([np.frombuffer(row[2], dtype=np.float32) for row in rows])
        similarities = _cosine_similarity(query_vector, vectors)
        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            Document(
                page_content=rows[index][0],
                metadata=json.loads(rows[index][1]) if rows[index][1] else {},
            )
            for index in top_indices
        ]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as connection:
            return connection.execute(f'SELECT COUNT(*) FROM "{self.table_name}"').fetchone()[0]

    def _ensure_table(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                f'''
                CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                '''
            )


class LanceDBVectorStore:
    """Local LanceDB vector store with the same search interface."""

    def __init__(self, db_path: str, collection_name: str, embedding):
        import lancedb

        self.db_path = db_path
        self.collection_name = _clean_collection_name(collection_name)
        self.embedding = embedding
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.table = self._open_table()

    def reset(self) -> None:
        try:
            self.db.drop_table(self.collection_name)
        except Exception:
            pass
        self.table = None

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        if not documents:
            return

        ids = ids or [f"chunk_{index + 1:05d}" for index in range(len(documents))]
        vectors = self.embedding.embed_documents([document.page_content for document in documents])
        rows = [
            {
                "id": ids[index],
                "text": document.page_content,
                "metadata": json.dumps(document.metadata, ensure_ascii=False),
                "vector": [float(value) for value in vectors[index]],
            }
            for index, document in enumerate(documents)
        ]
        if self.table is None:
            self.table = self.db.create_table(self.collection_name, data=rows)
        else:
            self.table.add(rows)

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        if self.table is None:
            return []
        query_vector = [float(value) for value in self.embedding.embed_query(query)]
        results = self.table.search(query_vector).limit(k).to_list()
        return [
            Document(
                page_content=row.get("text", ""),
                metadata=json.loads(row.get("metadata") or "{}"),
            )
            for row in results
        ]

    def count(self) -> int:
        if self.table is None:
            return 0
        return self.table.count_rows()

    def _open_table(self):
        try:
            return self.db.open_table(self.collection_name)
        except Exception:
            return None
