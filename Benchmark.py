"""Benchmark UI for response speed, retrieval quality, and top-k sensitivity."""

import streamlit as st
import os
import sys
import logging
import json
import gc
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.security import build_pg_connection_string, require_env

# Vector databases
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector
import chromadb
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# LangChain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Utils
from utils.document_processor import (
    ANSWERABLE_QUERIES,
    BENCHMARK_DOCUMENTS,
    QUERY_METADATA,
)
from utils.benchmark_config import (
    CONCURRENT_QUERIES_PER_USER,
    CONCURRENT_USER_LEVELS,
    FIREBASE_VECTOR_DIMENSION,
    LANCEDB_PATH,
    QDRANT_URL,
    SCALABILITY_DOC_COUNTS,
    SQLITE_DB_PATH,
    build_scalability_collection_name,
    build_scalability_namespace,
    get_scalability_query_set,
)
from utils.deepeval_evaluator import (
    OllamaDeepEvalModel,
    build_reference_answer,
    evaluate_deepeval_rag_metrics,
)
from utils.resource_monitor import ResourceMonitor
from utils.local_vector_stores import LanceDBVectorStore, SQLiteVectorStore
from utils.cloud_vector_stores import (
    FirebaseVectorStore,
    TruncatedNormalizedEmbeddings,
    create_supabase_engine,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================
# SHADCN-STYLE COLOR PALETTE
# ============================================
COLORS = {
    'Pinecone': '#8B5CF6',      # Violet
    'PostgreSQL': '#10B981',    # Emerald
    'ChromaDB': '#F59E0B',      # Amber
    'SQLite': '#14B8A6',        # Teal
    'LanceDB': '#EF4444',       # Red
    'Qdrant': '#DC244C',        # Qdrant red
    'Supabase': '#3ECF8E',      # Supabase green
    'Firebase': '#FFCA28',      # Firebase amber
    'primary': '#6366F1',       # Indigo
    'success': '#10B981',       # Emerald
    'muted': '#64748B',         # Slate
    'background': '#09090B',    # Zinc-950
    'card': '#18181B',          # Zinc-900
    'border': '#27272A',        # Zinc-800
    'text': '#FAFAFA',          # Zinc-50
}

# Page configuration
st.set_page_config(
    page_title="Vector Database Benchmark",
    page_icon="▣",
    layout="wide"
)

# Shadcn-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #09090B;
    }
    
    /* Shadcn Card Style */
    .metric-card {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 8px 0;
        letter-spacing: -0.025em;
    }
    
    .metric-label {
        color: #71717A;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Database Badges - Distinct Colors */
    .db-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-pinecone { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-postgresql { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-chromadb { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-sqlite { background: rgba(20, 184, 166, 0.15); color: #2DD4BF; border: 1px solid rgba(20, 184, 166, 0.3); }
    .badge-lancedb { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-qdrant { background: rgba(220, 36, 76, 0.15); color: #FB7185; border: 1px solid rgba(220, 36, 76, 0.3); }
    .badge-supabase { background: rgba(62, 207, 142, 0.15); color: #6EE7B7; border: 1px solid rgba(62, 207, 142, 0.3); }
    .badge-firebase { background: rgba(255, 202, 40, 0.15); color: #FDE68A; border: 1px solid rgba(255, 202, 40, 0.3); }
    
    /* Section Headers - Shadcn Style */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 32px 0 16px 0;
        letter-spacing: -0.025em;
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    
    /* Input/Slider Styling */
    .stSlider > div > div {
        background-color: #27272A;
    }
    
    /* Divider */
    hr {
        border-color: #27272A;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
EMBEDDING_MODEL = require_env("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "3"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
RANDOM_SEED = 42
THESIS_TITLE = "The Effect of Vector Database Selection on Scalability and Response Speed in a Retrieval-Augmented Generation"
BENCHMARK_RESULTS_DIR = "benchmark_results"
CONCURRENT_SKIP_ERROR_RATE = 0.20
CONCURRENT_SKIP_DURATION_SECONDS = 600
DEEPEVAL_QUERY_CAP = 10
SUPABASE_STATEMENT_TIMEOUT_MS = int(os.getenv("SUPABASE_STATEMENT_TIMEOUT_MS", "600000"))

# Title
st.markdown("""
<div style="text-align: center; padding: 32px 0 24px 0;">
    <h1 style="font-size: 2rem; font-weight: 600; color: #FAFAFA; margin-bottom: 4px; letter-spacing: -0.025em;">
        Vector Database Benchmark
    </h1>
    <p style="color: #71717A; font-size: 0.875rem; font-weight: 400;">
        Comparing Retrieval Speed and Scalability in RAG Systems
    </p>
</div>
""", unsafe_allow_html=True)

# Header Configuration - Centered horizontal layout
_, config_col1, config_col2, config_col3, _ = st.columns([0.5, 1, 2, 2, 0.5])

# LLM Model
with config_col1:
    st.markdown(f"""
    <div style="
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 6px;
        padding: 12px 16px;
        text-align: center;
    ">
        <div style="color: #71717A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;">LLM</div>
        <div style="color: #FAFAFA; font-weight: 500; font-size: 0.9rem;">{CHAT_MODEL}</div>
        <div style="color: #71717A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 10px;">Embedding</div>
        <div style="color: #FAFAFA; font-weight: 500; font-size: 0.85rem;">{EMBEDDING_MODEL}</div>
    </div>
    """, unsafe_allow_html=True)

# Databases
with config_col2:
    st.markdown('<p style="color: #71717A; font-size: 0.75rem; margin-bottom: 8px;">Databases</p>', unsafe_allow_html=True)
    db_col1, db_col2, db_col3, db_col4, db_col5, db_col6, db_col7, db_col8 = st.columns(8)
    with db_col1:
        test_pinecone = st.checkbox("Pinecone", value=True)
    with db_col2:
        test_postgresql = st.checkbox("PostgreSQL", value=True)
    with db_col3:
        test_chroma = st.checkbox("ChromaDB", value=True)
    with db_col4:
        test_sqlite = st.checkbox("SQLite", value=True)
    with db_col5:
        test_lancedb = st.checkbox("LanceDB", value=True)
    with db_col6:
        test_qdrant = st.checkbox("Qdrant", value=True)
    with db_col7:
        test_supabase = st.checkbox("Supabase", value=False)
    with db_col8:
        test_firebase = st.checkbox("Firebase", value=False)

# Parameters
with config_col3:
    st.markdown('<p style="color: #71717A; font-size: 0.75rem; margin-bottom: 8px;">Parameters</p>', unsafe_allow_html=True)
    param_col1, param_col2, param_col3 = st.columns(3)
    unique_query_count = len(ANSWERABLE_QUERIES)
    query_options = [option for option in range(10, 101, 10) if option <= unique_query_count]
    if unique_query_count not in query_options:
        query_options.append(unique_query_count)
    with param_col1:
        num_queries = st.selectbox("Queries", query_options, index=len(query_options) - 1, label_visibility="collapsed")
    with param_col2:
        top_k = st.selectbox("Top K", [1, 2, 3, 4, 5], index=2, label_visibility="collapsed")
    with param_col3:
        repetitions = st.selectbox("Runs", [1, 3, 5], index=1, label_visibility="collapsed")

if not any([
    test_pinecone,
    test_postgresql,
    test_chroma,
    test_sqlite,
    test_lancedb,
    test_qdrant,
    test_supabase,
    test_firebase,
]):
    st.error("Select at least one database")

# Centered Button
st.markdown("<br>", unsafe_allow_html=True)
_, col_btn1, col_btn2, _ = st.columns([2.5, 1, 1.4, 2.5])
with col_btn1:
    run_benchmark = st.button("Run Benchmark", type="primary", width="stretch")
with col_btn2:
    run_deepeval_quality = st.button("Run DeepEval Quality Only", width="stretch")

# Documents Section - same width as button
_, col_doc, _ = st.columns([3, 1, 3])
with col_doc:
    with st.expander("View Documents"):
        for i, doc in enumerate(BENCHMARK_DOCUMENTS):
            st.markdown(f"**{doc['name']}**")
            st.caption(f"{doc['language']} • {doc['file']}")
            
            file_path = os.path.join("documents", doc['file'])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f.read(),
                        file_name=doc['file'],
                        mime="application/pdf",
                        width="stretch",
                        key=f"doc_{i}"
                    )
            if i < len(BENCHMARK_DOCUMENTS) - 1:
                st.divider()

st.markdown("<br>", unsafe_allow_html=True)


# Initialize vector stores
def create_pinecone_store(embeddings, namespace: str | None = None):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "rag-vector-benchmark"))
    store_kwargs = {"index": index, "embedding": embeddings}
    if namespace is not None:
        store_kwargs["namespace"] = namespace
    return PineconeVectorStore(**store_kwargs)


def create_postgresql_store(embeddings, collection_name: str):
    db_password = os.getenv('DB_PASSWORD')
    if not db_password:
        raise ValueError("DB_PASSWORD not set")

    connection_string = build_pg_connection_string(
        user=os.getenv('DB_USER', 'raguser'),
        password=db_password,
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'ragdb')
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )


def create_chroma_store(embeddings, collection_name: str):
    client = chromadb.PersistentClient(path="chroma_db")
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def create_sqlite_store(embeddings, collection_name: str):
    return SQLiteVectorStore(SQLITE_DB_PATH, collection_name, embeddings)


def create_lancedb_store(embeddings, collection_name: str):
    return LanceDBVectorStore(LANCEDB_PATH, collection_name, embeddings)


def create_qdrant_store(embeddings, collection_name: str):
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def create_supabase_store(embeddings, collection_name: str):
    password = os.getenv("SUPABASE_DB_PASSWORD")
    if not password:
        raise ValueError("SUPABASE_DB_PASSWORD not set")
    connection_string = build_pg_connection_string(
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=password,
        host=os.getenv("SUPABASE_DB_HOST", ""),
        port=os.getenv("SUPABASE_DB_PORT", "5432"),
        database=os.getenv("SUPABASE_DB_NAME", "postgres"),
        sslmode=os.getenv("SUPABASE_SSLMODE", "require"),
    )
    engine = create_supabase_engine(connection_string, SUPABASE_STATEMENT_TIMEOUT_MS)
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )


def create_firebase_store(embeddings, collection_name: str):
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    if not project_id:
        raise ValueError("FIREBASE_PROJECT_ID not set")
    return FirebaseVectorStore(
        project_id=project_id,
        database_id=os.getenv("FIREBASE_DATABASE_ID", "(default)"),
        collection_name=collection_name,
        embedding=embeddings,
    )


def get_target_collection_names() -> list[str]:
    return [COLLECTION_NAME] + [
        build_scalability_collection_name(COLLECTION_NAME, doc_count)
        for doc_count in SCALABILITY_DOC_COUNTS
    ]


def get_firestore_collection_count(collection) -> int:
    results = collection.count().get()
    for row in results:
        values = row if isinstance(row, (list, tuple)) else [row]
        for value in values:
            if hasattr(value, "value"):
                return int(value.value)
    return 0


def get_benchmark_inventory(selected_databases: list[str]) -> dict[str, dict[str, int | str]]:
    """Collect pre-run vector counts without exposing credentials."""
    inventory = {}
    target_collections = get_target_collection_names()

    if "SQLite" in selected_databases:
        try:
            import sqlite3

            sqlite_counts = {}
            with sqlite3.connect(SQLITE_DB_PATH) as connection:
                for collection_name in target_collections:
                    table_name = f"vectors_{collection_name}"
                    row = connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                        (table_name,),
                    ).fetchone()
                    if row:
                        sqlite_counts[collection_name] = connection.execute(
                            f'SELECT COUNT(*) FROM "{table_name}"'
                        ).fetchone()[0]
            inventory["SQLite"] = sqlite_counts
        except Exception as e:
            inventory["SQLite"] = {"error": str(e)}

    if "ChromaDB" in selected_databases:
        try:
            client = chromadb.PersistentClient(path="chroma_db")
            chroma_counts = {}
            existing = {collection.name for collection in client.list_collections()}
            for collection_name in target_collections:
                if collection_name in existing:
                    chroma_counts[collection_name] = client.get_collection(collection_name).count()
            inventory["ChromaDB"] = chroma_counts
        except Exception as e:
            inventory["ChromaDB"] = {"error": str(e)}

    if "Pinecone" in selected_databases:
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "rag-vector-benchmark"))
            stats = index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            pinecone_counts = {}
            if "" in namespaces:
                pinecone_counts[COLLECTION_NAME] = namespaces[""].get("vector_count", 0)
            elif None in namespaces:
                pinecone_counts[COLLECTION_NAME] = namespaces[None].get("vector_count", 0)
            elif "default" in namespaces:
                pinecone_counts[COLLECTION_NAME] = namespaces["default"].get("vector_count", 0)
            for doc_count in SCALABILITY_DOC_COUNTS:
                namespace = build_scalability_namespace(doc_count)
                collection_name = build_scalability_collection_name(COLLECTION_NAME, doc_count)
                if namespace in namespaces:
                    pinecone_counts[collection_name] = namespaces[namespace].get("vector_count", 0)
            inventory["Pinecone"] = pinecone_counts
        except Exception as e:
            inventory["Pinecone"] = {"error": str(e)}

    if "PostgreSQL" in selected_databases:
        try:
            import psycopg

            db_password = os.getenv("DB_PASSWORD")
            if not db_password:
                raise ValueError("DB_PASSWORD not set")
            with psycopg.connect(
                user=os.getenv("DB_USER", "raguser"),
                password=db_password,
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                dbname=os.getenv("DB_NAME", "ragdb"),
            ) as connection:
                rows = connection.execute(
                    """
                    SELECT c.name, COUNT(e.id)
                    FROM langchain_pg_collection c
                    LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                    WHERE c.name = ANY(%s)
                    GROUP BY c.name
                    """,
                    (target_collections,),
                ).fetchall()
                inventory["PostgreSQL"] = {name: count for name, count in rows}
        except Exception as e:
            inventory["PostgreSQL"] = {"error": str(e)}

    if "LanceDB" in selected_databases:
        try:
            lancedb_counts = {}
            if os.path.exists(LANCEDB_PATH):
                import lancedb

                db = lancedb.connect(LANCEDB_PATH)
                raw_tables = db.list_tables()
                raw_tables = raw_tables.tables if hasattr(raw_tables, "tables") else raw_tables
                existing = set(raw_tables)
                for collection_name in target_collections:
                    if collection_name in existing:
                        lancedb_counts[collection_name] = db.open_table(collection_name).count_rows()
            inventory["LanceDB"] = lancedb_counts
        except Exception as e:
            inventory["LanceDB"] = {"error": str(e)}

    if "Qdrant" in selected_databases:
        try:
            client = QdrantClient(url=QDRANT_URL)
            existing = {collection.name for collection in client.get_collections().collections}
            qdrant_counts = {}
            for collection_name in target_collections:
                if collection_name in existing:
                    qdrant_counts[collection_name] = client.get_collection(collection_name).points_count
            inventory["Qdrant"] = qdrant_counts
        except Exception as e:
            inventory["Qdrant"] = {"error": str(e)}

    if "Supabase" in selected_databases:
        try:
            import psycopg

            password = os.getenv("SUPABASE_DB_PASSWORD")
            if not password:
                raise ValueError("SUPABASE_DB_PASSWORD not set")
            with psycopg.connect(
                host=os.getenv("SUPABASE_DB_HOST", ""),
                port=os.getenv("SUPABASE_DB_PORT", "5432"),
                dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
                user=os.getenv("SUPABASE_DB_USER", "postgres"),
                password=password,
                sslmode=os.getenv("SUPABASE_SSLMODE", "require"),
            ) as connection:
                rows = connection.execute(
                    """
                    SELECT c.name, COUNT(e.id)
                    FROM langchain_pg_collection c
                    LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                    WHERE c.name = ANY(%s)
                    GROUP BY c.name
                    """,
                    (target_collections,),
                ).fetchall()
                inventory["Supabase"] = {name: count for name, count in rows}
        except Exception as e:
            inventory["Supabase"] = {"error": str(e)}

    if "Firebase" in selected_databases:
        try:
            from google.cloud import firestore

            project_id = os.getenv("FIREBASE_PROJECT_ID")
            if not project_id:
                raise ValueError("FIREBASE_PROJECT_ID not set")
            client = firestore.Client(
                project=project_id,
                database=os.getenv("FIREBASE_DATABASE_ID", "(default)"),
            )
            inventory["Firebase"] = {
                collection_name: get_firestore_collection_count(client.collection(collection_name))
                for collection_name in target_collections
            }
        except Exception as e:
            inventory["Firebase"] = {"error": str(e)}

    return inventory


def get_query_manifest(queries: list[str]) -> list[dict[str, str]]:
    return [
        {
            "query": query,
            "query_type": QUERY_METADATA.get(query, {}).get("query_type", "unknown"),
            "source_file": QUERY_METADATA.get(query, {}).get("source_file", ""),
            "language": QUERY_METADATA.get(query, {}).get("language", ""),
        }
        for query in queries
    ]


def get_document_manifest() -> list[dict[str, str]]:
    return [
        {
            "name": document.get("name", ""),
            "file": document.get("file", ""),
            "language": document.get("language", ""),
        }
        for document in BENCHMARK_DOCUMENTS
    ]


def validate_benchmark_inventory(inventory: dict[str, dict[str, int | str]]) -> list[str]:
    target_collections = get_target_collection_names()
    errors = []

    for db_name, counts in inventory.items():
        if "error" in counts:
            errors.append(f"{db_name}: inventory error: {counts['error']}")
            continue
        for collection_name in target_collections:
            count = counts.get(collection_name)
            if count is None:
                errors.append(f"{db_name}: missing collection {collection_name}")
            elif int(count) <= 0:
                errors.append(f"{db_name}: collection {collection_name} has {count} vectors")

    for collection_name in target_collections:
        observed = {
            db_name: int(counts[collection_name])
            for db_name, counts in inventory.items()
            if "error" not in counts and collection_name in counts
        }
        if observed and len(set(observed.values())) > 1:
            errors.append(f"{collection_name}: inconsistent counts across DBs: {observed}")

    return errors


def dataframe_success_summary(df: pd.DataFrame) -> dict[str, int | float]:
    total_rows = len(df)
    success_rows = int(df.get('success', pd.Series(dtype=bool)).sum())
    failed_rows = total_rows - success_rows
    return {
        'total_rows': total_rows,
        'success_rows': success_rows,
        'failed_rows': failed_rows,
        'success_rate': success_rows / total_rows if total_rows else 0,
    }


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'success' not in df:
        return df[df['total_time'] > 0]
    return df[(df['success'] == True) & (df['total_time'] > 0)]


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def safe_max(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_checkpoint(run_dir: str, filename: str, data: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(sanitize_for_json(data), file, indent=2, ensure_ascii=False, allow_nan=False)
    return path


def write_latest_checkpoint(run_dir: str, data: dict) -> None:
    write_checkpoint(run_dir, "latest_partial.json", data)


def get_database_limitations(databases_tested: list[str]) -> dict[str, dict]:
    limitations = {}
    if "Firebase" in databases_tested:
        native_dimension = 4096 if EMBEDDING_MODEL == "qwen3-embedding:8b" else None
        limitations["Firebase"] = {
            "embedding_model": EMBEDDING_MODEL,
            "native_dimension": native_dimension,
            "stored_dimension": FIREBASE_VECTOR_DIMENSION,
            "transformation": "first_2048_dimensions_then_l2_normalization",
            "ingestion_mode": "resumable_across_free_tier_daily_quota_windows",
            "free_tier_daily_write_quota": 20000,
            "ingestion_note": (
                "Firebase collections were ingested across multiple daily quota windows "
                "because the full and scalability collections require more writes than "
                "the Firestore free-tier daily allowance. This affects ingestion duration "
                "only; ingestion time is excluded from benchmark retrieval metrics."
            ),
            "limitation": (
                "Cloud Firestore supports at most 2048-dimensional vector embeddings. "
                "Firebase therefore used a reduced representation and is not "
                "dimension-identical to databases evaluated with the native embedding."
            ),
        }
    if "Supabase" in databases_tested:
        limitations["Supabase"] = {
            "embedding_model": EMBEDDING_MODEL,
            "stored_dimension": 4096 if EMBEDDING_MODEL == "qwen3-embedding:8b" else None,
            "search_mode": "exact_pgvector",
            "statement_timeout_ms": SUPABASE_STATEMENT_TIMEOUT_MS,
            "limitation": (
                "The native 4096-dimensional vector exceeds pgvector HNSW index limits "
                "for vector (2000) and halfvec (4000), so Supabase used exact "
                "nearest-neighbor search."
            ),
        }
    return limitations


def build_partial_export(
    status: str,
    benchmark_started_at: datetime,
    databases_tested: list[str],
    num_queries: int,
    top_k: int,
    repetitions: int,
    scalability_queries: list[str],
    database_inventory_before_run: dict,
    completed_phases: dict,
    current_phase: str,
    error: str | None = None,
) -> dict:
    now = datetime.now()
    return {
        "metadata": {
            "status": status,
            "run_id": benchmark_started_at.strftime("%Y%m%d_%H%M%S"),
            "benchmark_started_at": benchmark_started_at.isoformat(),
            "checkpoint_at": now.isoformat(),
            "duration_seconds": round((now - benchmark_started_at).total_seconds(), 2),
            "llm_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": COLLECTION_NAME,
            "num_queries": num_queries,
            "repetitions": repetitions,
            "top_k": top_k,
            "databases_tested": databases_tested,
            "current_phase": current_phase,
            "error": error,
        },
        "benchmark_plan": {
            "thesis_title": THESIS_TITLE,
            "databases_tested": databases_tested,
            "num_queries": num_queries,
            "repetitions": repetitions,
            "top_k": top_k,
            "scalability_doc_counts": SCALABILITY_DOC_COUNTS,
            "concurrent_user_levels": CONCURRENT_USER_LEVELS,
            "concurrent_queries_per_user": CONCURRENT_QUERIES_PER_USER,
            "concurrent_skip_error_rate": CONCURRENT_SKIP_ERROR_RATE,
            "concurrent_skip_duration_seconds": CONCURRENT_SKIP_DURATION_SECONDS,
            "random_seed": RANDOM_SEED,
        },
        "database_inventory_before_run": database_inventory_before_run,
        "database_limitations": get_database_limitations(databases_tested),
        "scalability_query_manifest": get_query_manifest(scalability_queries),
        "document_manifest": get_document_manifest(),
        "completed_phases": completed_phases,
    }


def init_all_vector_stores(
    test_pinecone,
    test_postgresql,
    test_chroma,
    test_sqlite,
    test_lancedb,
    test_qdrant,
    test_supabase,
    test_firebase,
    embedding_model,
):
    stores = {}
    embeddings = OllamaEmbeddings(model=embedding_model)

    if test_pinecone:
        try:
            stores['Pinecone'] = create_pinecone_store(embeddings)
        except Exception as e:
            logger.error(f"Pinecone connection error: {str(e)}")

    if test_postgresql:
        try:
            stores['PostgreSQL'] = create_postgresql_store(embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"PostgreSQL connection error: {str(e)}")

    if test_chroma:
        try:
            stores['ChromaDB'] = create_chroma_store(embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"ChromaDB connection error: {str(e)}")

    if test_sqlite:
        try:
            stores['SQLite'] = create_sqlite_store(embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"SQLite connection error: {str(e)}")

    if test_lancedb:
        try:
            stores['LanceDB'] = create_lancedb_store(embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"LanceDB connection error: {str(e)}")

    if test_qdrant:
        try:
            stores['Qdrant'] = create_qdrant_store(embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Qdrant connection error: {str(e)}")

    if test_supabase:
        try:
            stores['Supabase'] = create_supabase_store(embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Supabase connection error: {str(e)}")

    if test_firebase:
        try:
            firebase_embeddings = TruncatedNormalizedEmbeddings(
                embeddings, FIREBASE_VECTOR_DIMENSION
            )
            stores['Firebase'] = create_firebase_store(firebase_embeddings, COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Firebase connection error: {str(e)}")

    return stores


def init_scalability_vector_stores(
    test_pinecone,
    test_postgresql,
    test_chroma,
    test_sqlite,
    test_lancedb,
    test_qdrant,
    test_supabase,
    test_firebase,
    doc_counts,
    embedding_model,
):
    stores = {}
    embeddings = OllamaEmbeddings(model=embedding_model)

    if test_pinecone:
        try:
            stores['Pinecone'] = {
                doc_count: create_pinecone_store(embeddings, build_scalability_namespace(doc_count))
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"Pinecone scalability connection error: {str(e)}")

    if test_postgresql:
        try:
            stores['PostgreSQL'] = {
                doc_count: create_postgresql_store(
                    embeddings, build_scalability_collection_name(COLLECTION_NAME, doc_count)
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"PostgreSQL scalability connection error: {str(e)}")

    if test_chroma:
        try:
            stores['ChromaDB'] = {
                doc_count: create_chroma_store(
                    embeddings, build_scalability_collection_name(COLLECTION_NAME, doc_count)
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"ChromaDB scalability connection error: {str(e)}")

    if test_sqlite:
        try:
            stores['SQLite'] = {
                doc_count: create_sqlite_store(
                    embeddings, build_scalability_collection_name(COLLECTION_NAME, doc_count)
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"SQLite scalability connection error: {str(e)}")

    if test_lancedb:
        try:
            stores['LanceDB'] = {
                doc_count: create_lancedb_store(
                    embeddings, build_scalability_collection_name(COLLECTION_NAME, doc_count)
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"LanceDB scalability connection error: {str(e)}")

    if test_qdrant:
        try:
            stores['Qdrant'] = {
                doc_count: create_qdrant_store(
                    embeddings, build_scalability_collection_name(COLLECTION_NAME, doc_count)
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"Qdrant scalability connection error: {str(e)}")

    if test_supabase:
        try:
            stores['Supabase'] = {
                doc_count: create_supabase_store(
                    embeddings, build_scalability_collection_name(COLLECTION_NAME, doc_count)
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"Supabase scalability connection error: {str(e)}")

    if test_firebase:
        try:
            firebase_embeddings = TruncatedNormalizedEmbeddings(
                embeddings, FIREBASE_VECTOR_DIMENSION
            )
            stores['Firebase'] = {
                doc_count: create_firebase_store(
                    firebase_embeddings,
                    build_scalability_collection_name(COLLECTION_NAME, doc_count),
                )
                for doc_count in doc_counts
            }
        except Exception as e:
            logger.error(f"Firebase scalability connection error: {str(e)}")

    return stores


def get_selected_database_names(
    test_pinecone: bool,
    test_postgresql: bool,
    test_chroma: bool,
    test_sqlite: bool,
    test_lancedb: bool,
    test_qdrant: bool,
    test_supabase: bool,
    test_firebase: bool,
) -> list[str]:
    selections = {
        "Pinecone": test_pinecone,
        "PostgreSQL": test_postgresql,
        "ChromaDB": test_chroma,
        "SQLite": test_sqlite,
        "LanceDB": test_lancedb,
        "Qdrant": test_qdrant,
        "Supabase": test_supabase,
        "Firebase": test_firebase,
    }
    return [name for name, selected in selections.items() if selected]


def fixed_top_k_search(vector_store, query: str, top_k: int):
    """Run the fixed top-k retrieval used for latency and answerable-query evaluation."""
    return vector_store.similarity_search(query, k=top_k)


def measure_performance(vector_store, query, llm, top_k):
    try:
        start = time.time()
        docs = fixed_top_k_search(vector_store, query, top_k)
        retrieval_time = time.time() - start
        
        if docs and len(docs) > 0:
            docs_text = "\n\n".join(d.page_content for d in docs)
            system_prompt = f"Context: {docs_text}\n\nAnswer based only on the context. Respond in the same language as the question."
            messages = [SystemMessage(system_prompt), HumanMessage(query)]
            
            start = time.time()
            response = llm.invoke(messages)
            llm_time = time.time() - start
        else:
            llm_time = 0
        
        total_time = retrieval_time + llm_time
        
        return {
            'retrieval_time': retrieval_time * 1000,
            'llm_time': llm_time * 1000,
            'total_time': total_time * 1000,
            'num_docs': len(docs),
            'success': len(docs) > 0,
            'answer': response.content if docs and len(docs) > 0 else '',
            'retrieval_context': [doc.page_content for doc in docs],
            'retrieved_sources': [doc.metadata.get('source_file', '') for doc in docs],
        }
    except Exception as e:
        return {
            'retrieval_time': 0,
            'llm_time': 0,
            'total_time': 0,
            'num_docs': 0,
            'success': False,
            'answer': '',
            'retrieval_context': [],
            'retrieved_sources': [],
            'error': str(e)
        }


def measure_top_k_sensitivity_single_k(vector_store, queries, k):
    """
    Measure response time for a single Top-K level.
    """
    level_times = []
    for query in queries:
        try:
            start = time.time()
            docs = fixed_top_k_search(vector_store, query, k)
            retrieval_time = (time.time() - start) * 1000
            level_times.append(retrieval_time)
        except Exception as e:
            raise RuntimeError(f"Top-K sensitivity failed at k={k}: {e}") from e
    
    if level_times:
        return {
            'avg_time': np.mean(level_times),
            'std_time': np.std(level_times),
            'min_time': np.min(level_times),
            'max_time': np.max(level_times),
            'query_count': len(level_times),
        }
    return None


def measure_corpus_scalability_single_run(vector_store, queries, top_k):
    """Measure retrieval latency for a fixed query set as corpus size grows."""
    level_times = []
    for query in queries:
        try:
            start = time.time()
            fixed_top_k_search(vector_store, query, top_k)
            retrieval_time = (time.time() - start) * 1000
            level_times.append(retrieval_time)
        except Exception as e:
            raise RuntimeError(f"Corpus scalability retrieval failed: {e}") from e

    if level_times:
        return {
            'avg_time': np.mean(level_times),
            'median_time': np.median(level_times),
            'std_time': np.std(level_times),
            'p95_time': np.quantile(level_times, 0.95),
            'min_time': np.min(level_times),
            'max_time': np.max(level_times),
            'query_count': len(level_times),
        }
    return None


def summarize_repeated_measurements(measurements, level_key, level_value):
    """Aggregate repeated latency measurements for one benchmark level."""
    avg_times = [measurement['avg_time'] for measurement in measurements]
    median_times = [measurement.get('median_time', measurement['avg_time']) for measurement in measurements]
    p95_times = [measurement.get('p95_time', measurement['avg_time']) for measurement in measurements]

    return {
        level_key: level_value,
        'runs': len(measurements),
        'mean_avg_time': float(np.mean(avg_times)),
        'mean_median_time': float(np.mean(median_times)),
        'mean_p95_time': float(np.mean(p95_times)),
        'std_avg_time': float(np.std(avg_times)),
        'min_avg_time': float(np.min(avg_times)),
        'max_avg_time': float(np.max(avg_times)),
        'query_count': int(measurements[0].get('query_count', 10)),
        'per_run': measurements,
    }


def measure_deepeval_quality_single(vector_store, query, llm, judge_model, top_k):
    """
    Measure answer quality using DeepEval RAG metrics.
    """
    query_metadata = QUERY_METADATA.get(query)
    if not query_metadata:
        return None

    try:
        performance = measure_performance(vector_store, query, llm, top_k)
        expected_output = query_metadata.get('reference_answer') or build_reference_answer(
            query, query_metadata.get('keywords', [])
        )
        deepeval_scores = evaluate_deepeval_rag_metrics(
            query=query,
            actual_output=performance['answer'],
            retrieval_context=performance['retrieval_context'],
            expected_output=expected_output,
            judge_model=judge_model,
        )
        return {
            'query': query,
            'expected_output': expected_output,
            'actual_output': performance['answer'],
            'retrieved_sources': performance['retrieved_sources'],
            'retrieval_time': performance['retrieval_time'],
            'llm_time': performance['llm_time'],
            'total_time': performance['total_time'],
            **deepeval_scores,
        }
    except Exception as e:
        return {
            'query': query,
            'AnswerRelevancy_score': 0,
            'Faithfulness_score': 0,
            'ContextualRelevancy_score': 0,
            'ContextualPrecision_score': 0,
            'ContextualRecall_score': 0,
            'error': str(e)
        }


def measure_concurrent_users(vector_store, queries, top_k, concurrent_users, queries_per_user):
    """Measure retrieval-only latency under concurrent load."""
    selected_queries = [queries[i % len(queries)] for i in range(concurrent_users * queries_per_user)]

    def run_query(query):
        start = time.time()
        fixed_top_k_search(vector_store, query, top_k)
        return (time.time() - start) * 1000

    monitor = ResourceMonitor()
    monitor.start()
    test_start = time.time()
    latencies = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(run_query, query) for query in selected_queries]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:
                errors += 1

    duration_seconds = time.time() - test_start
    resources = monitor.stop()
    total_requests = len(selected_queries)

    if not latencies:
        return {
            'concurrent_users': concurrent_users,
            'total_requests': total_requests,
            'successful_requests': 0,
            'error_rate': 1.0,
            **resources,
        }

    return {
        'concurrent_users': concurrent_users,
        'queries_per_user': queries_per_user,
        'total_requests': total_requests,
        'successful_requests': len(latencies),
        'errors': errors,
        'error_rate': errors / total_requests if total_requests else 0,
        'duration_seconds': duration_seconds,
        'throughput_rps': len(latencies) / duration_seconds if duration_seconds else 0,
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'p95_latency_ms': float(np.quantile(latencies, 0.95)),
        'p99_latency_ms': float(np.quantile(latencies, 0.99)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        **resources,
    }


def run_deepeval_quality_only(
    test_pinecone: bool,
    test_postgresql: bool,
    test_chroma: bool,
    test_sqlite: bool,
    test_lancedb: bool,
    test_qdrant: bool,
    test_supabase: bool,
    test_firebase: bool,
    top_k: int,
) -> str:
    """Run DeepEval as a separate quality-only benchmark."""
    benchmark_started_at = datetime.now()
    run_id = benchmark_started_at.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(BENCHMARK_RESULTS_DIR, f"{run_id}_deepeval_quality_only")

    vector_stores = init_all_vector_stores(
        test_pinecone,
        test_postgresql,
        test_chroma,
        test_sqlite,
        test_lancedb,
        test_qdrant,
        test_supabase,
        test_firebase,
        EMBEDDING_MODEL,
    )
    if not vector_stores:
        raise RuntimeError("No vector stores initialized for DeepEval")

    selected_databases = get_selected_database_names(
        test_pinecone,
        test_postgresql,
        test_chroma,
        test_sqlite,
        test_lancedb,
        test_qdrant,
        test_supabase,
        test_firebase,
    )
    missing_databases = sorted(set(selected_databases) - set(vector_stores))
    if missing_databases:
        raise RuntimeError("Failed to initialize: " + ", ".join(missing_databases))

    databases_tested = list(vector_stores.keys())
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    judge_model = OllamaDeepEvalModel(CHAT_MODEL, temperature=0.0)

    randomized_queries = ANSWERABLE_QUERIES.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(randomized_queries)
    deepeval_queries = randomized_queries[:DEEPEVAL_QUERY_CAP]

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(deepeval_queries) * len(vector_stores)
    current_step = 0
    deepeval_quality_results = {db_name: [] for db_name in vector_stores.keys()}

    for q_idx, query in enumerate(deepeval_queries):
        status_text.text(f"DeepEval Quality [Query {q_idx + 1}/{len(deepeval_queries)}]...")
        for db_name, vector_store in vector_stores.items():
            result = measure_deepeval_quality_single(vector_store, query, llm, judge_model, top_k)
            if result:
                deepeval_quality_results[db_name].append(result)
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            partial_data = {
                'metadata': {
                    'status': 'running',
                    'run_id': run_id,
                    'benchmark_started_at': benchmark_started_at.isoformat(),
                    'checkpoint_at': datetime.now().isoformat(),
                    'llm_model': CHAT_MODEL,
                    'embedding_model': EMBEDDING_MODEL,
                    'collection_name': COLLECTION_NAME,
                    'top_k': top_k,
                    'deepeval_query_cap': DEEPEVAL_QUERY_CAP,
                    'deepeval_query_count': len(deepeval_queries),
                    'databases_tested': databases_tested,
                    'current_phase': f"deepeval_answer_quality:{db_name}:query_{q_idx + 1}",
                },
                'database_limitations': get_database_limitations(databases_tested),
                'deepeval_query_manifest': get_query_manifest(deepeval_queries),
                'deepeval_answer_quality_partial': deepeval_quality_results,
            }
            write_checkpoint(run_dir, "deepeval_quality_partial.json", partial_data)
            gc.collect()

    export_data = {
        'metadata': {
            'status': 'completed',
            'run_id': run_id,
            'benchmark_started_at': benchmark_started_at.isoformat(),
            'benchmark_completed_at': datetime.now().isoformat(),
            'llm_model': CHAT_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'collection_name': COLLECTION_NAME,
            'top_k': top_k,
            'deepeval_query_cap': DEEPEVAL_QUERY_CAP,
            'deepeval_query_count': len(deepeval_queries),
            'databases_tested': databases_tested,
        },
        'benchmark_plan': {
            'thesis_title': THESIS_TITLE,
            'databases_tested': databases_tested,
            'top_k': top_k,
            'deepeval_query_cap': DEEPEVAL_QUERY_CAP,
            'deepeval_query_count': len(deepeval_queries),
            'random_seed': RANDOM_SEED,
            'run_mode': 'deepeval_quality_only',
        },
        'database_limitations': get_database_limitations(databases_tested),
        'deepeval_query_manifest': get_query_manifest(deepeval_queries),
        'deepeval_answer_quality': {
            db_name: {
                'avg_answer_relevancy': round(np.mean([r.get('AnswerRelevancy_score', 0) for r in results]), 4),
                'avg_faithfulness': round(np.mean([r.get('Faithfulness_score', 0) for r in results]), 4),
                'avg_contextual_relevancy': round(np.mean([r.get('ContextualRelevancy_score', 0) for r in results]), 4),
                'avg_contextual_precision': round(np.mean([r.get('ContextualPrecision_score', 0) for r in results]), 4),
                'avg_contextual_recall': round(np.mean([r.get('ContextualRecall_score', 0) for r in results]), 4),
                'per_query': results,
            } for db_name, results in deepeval_quality_results.items()
        },
    }
    final_path = write_checkpoint(run_dir, "deepeval_quality_final.json", export_data)
    st.session_state['deepeval_quality_results'] = deepeval_quality_results
    st.session_state['deepeval_export_data'] = sanitize_for_json(export_data)
    st.session_state['deepeval_export_path'] = final_path
    status_text.text("DeepEval quality-only completed.")
    progress_bar.progress(1.0)
    return final_path


if run_deepeval_quality:
    with st.spinner("Running DeepEval quality-only benchmark..."):
        try:
            deepeval_quality_path = run_deepeval_quality_only(
                test_pinecone=test_pinecone,
                test_postgresql=test_postgresql,
                test_chroma=test_chroma,
                test_sqlite=test_sqlite,
                test_lancedb=test_lancedb,
                test_qdrant=test_qdrant,
                test_supabase=test_supabase,
                test_firebase=test_firebase,
                top_k=top_k,
            )
        except Exception as e:
            logger.exception("DeepEval quality-only benchmark failed")
            st.error(f"DeepEval quality-only benchmark failed: {e}")
            st.stop()
    st.success(f"Saved DeepEval quality-only result to {deepeval_quality_path}")


# Main benchmark
if run_benchmark:
    benchmark_started_at = datetime.now()
    with st.spinner("Initializing..."):
        vector_stores = init_all_vector_stores(
            test_pinecone,
            test_postgresql,
            test_chroma,
            test_sqlite,
            test_lancedb,
            test_qdrant,
            test_supabase,
            test_firebase,
            EMBEDDING_MODEL,
        )
        scalability_vector_stores = init_scalability_vector_stores(
            test_pinecone,
            test_postgresql,
            test_chroma,
            test_sqlite,
            test_lancedb,
            test_qdrant,
            test_supabase,
            test_firebase,
            SCALABILITY_DOC_COUNTS,
            EMBEDDING_MODEL,
        )

    if not vector_stores:
        st.error("No vector stores initialized")
        st.stop()

    selected_databases = get_selected_database_names(
        test_pinecone,
        test_postgresql,
        test_chroma,
        test_sqlite,
        test_lancedb,
        test_qdrant,
        test_supabase,
        test_firebase,
    )
    missing_databases = sorted(set(selected_databases) - set(vector_stores))
    if missing_databases:
        st.error("Failed to initialize selected databases: " + ", ".join(missing_databases))
        st.stop()

    databases_tested = list(vector_stores.keys())
    database_inventory_before_run = get_benchmark_inventory(databases_tested)
    run_id = benchmark_started_at.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(BENCHMARK_RESULTS_DIR, run_id)
    completed_phases = {}
    write_latest_checkpoint(
        run_dir,
        build_partial_export(
            status="running",
            benchmark_started_at=benchmark_started_at,
            databases_tested=databases_tested,
            num_queries=num_queries,
            top_k=top_k,
            repetitions=repetitions,
            scalability_queries=[],
            database_inventory_before_run=database_inventory_before_run,
            completed_phases=completed_phases,
            current_phase="initializing",
        ),
    )
    inventory_errors = validate_benchmark_inventory(database_inventory_before_run)
    if inventory_errors:
        st.error("Preflight collection validation failed. Fix ingestion before running the thesis benchmark.")
        st.json({"errors": inventory_errors, "inventory": database_inventory_before_run})
        st.stop()

    missing_scalability = sorted(set(vector_stores) - set(scalability_vector_stores))
    if missing_scalability:
        st.error(
            "Missing corpus-size scalability stores for: "
            + ", ".join(missing_scalability)
            + ". Reingest the databases before running the thesis benchmark."
        )
        st.stop()

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    st.success(
        f"Initialized {len(vector_stores)} database(s) with fixed embedding '{EMBEDDING_MODEL}'"
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_results = {db_name: [] for db_name in vector_stores.keys()}
    total_tests = len(vector_stores) * num_queries * repetitions
    current_test = 0

    # Randomize queries with fixed seed for reproducibility
    randomized_queries = ANSWERABLE_QUERIES.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(randomized_queries)
    benchmark_queries = randomized_queries[:num_queries]
    scalability_queries = get_scalability_query_set()
    write_latest_checkpoint(
        run_dir,
        build_partial_export(
            status="running",
            benchmark_started_at=benchmark_started_at,
            databases_tested=databases_tested,
            num_queries=num_queries,
            top_k=top_k,
            repetitions=repetitions,
            scalability_queries=scalability_queries,
            database_inventory_before_run=database_inventory_before_run,
            completed_phases=completed_phases,
            current_phase="warmup",
        ),
    )

    # ========== WARM-UP PHASE ==========
    # Run a few queries on each DB to reduce cold start bias
    status_text.text("Warming up databases...")
    for db_name, vector_store in vector_stores.items():
        try:
            fixed_top_k_search(vector_store, randomized_queries[0], top_k)
        except Exception:
            pass

    for db_name, stores_by_size in scalability_vector_stores.items():
        try:
            fixed_top_k_search(stores_by_size[SCALABILITY_DOC_COUNTS[0]], scalability_queries[0], top_k)
        except Exception:
            pass

    top_k_levels = [1, 2, 3, 5, 8, 10, 15, 20]
    total_steps = (
        total_tests
        + len(top_k_levels) * len(vector_stores) * repetitions
        + len(SCALABILITY_DOC_COUNTS) * len(scalability_vector_stores) * repetitions
        + len(CONCURRENT_USER_LEVELS) * len(vector_stores) * repetitions
    )

    # Track and clear old results if configuration changed
    current_config = (
        test_pinecone,
        test_postgresql,
        test_chroma,
        test_sqlite,
        test_lancedb,
        test_qdrant,
        test_supabase,
        test_firebase,
        num_queries,
        top_k,
        repetitions,
        EMBEDDING_MODEL,
    )
    if 'tested_config' in st.session_state and st.session_state['tested_config'] != current_config:
        for key in [
            'benchmark_results',
            'dfs',
            'top_k_sensitivity_results',
            'corpus_scalability_results',
            'concurrent_scalability_results',
            'deepeval_quality_results',
            'export_data',
        ]:
            if key in st.session_state:
                del st.session_state[key]

    # FAIR BENCHMARK: Interleave queries across databases
    # Instead of testing all queries on DB1, then DB2, etc.
    # We test query1 on all DBs, then query2 on all DBs, etc.
    # This eliminates cold start bias and LLM caching advantages

    for repetition in range(1, repetitions + 1):
        for i in range(num_queries):
            query = benchmark_queries[i]
            query_metadata = QUERY_METADATA.get(query, {})
            global_query_num = ((repetition - 1) * num_queries) + i + 1

            for db_name, vector_store in vector_stores.items():
                status_text.text(
                    f"Speed Test [Run {repetition}/{repetitions}] [{db_name}] Query {i + 1}/{num_queries}..."
                )

                metrics = measure_performance(vector_store, query, llm, top_k)
                if not metrics.get('success'):
                    failure_message = (
                        f"Speed test failed for {db_name} on query {i + 1}: "
                        + metrics.get('error', 'no documents retrieved')
                    )
                    if db_name == 'Supabase':
                        # Free-tier disk I/O throttling makes worst-case latency
                        # unbounded; record the failure as data and keep the run alive.
                        st.warning(failure_message + " — recorded as failed query, continuing.")
                    else:
                        st.error(failure_message)
                        st.stop()
                all_results[db_name].append({
                    'repetition': repetition,
                    'query_num': i + 1,
                    'global_query_num': global_query_num,
                    'query': query,
                    'database': db_name,
                    'query_type': query_metadata.get('query_type', 'unknown'),
                    'query_answerable': True,
                    **metrics
                })

                current_test += 1
                progress_bar.progress(current_test / total_steps)

        completed_phases['speed_test_last_completed_repetition'] = repetition
        write_latest_checkpoint(
            run_dir,
            build_partial_export(
                status="running",
                benchmark_started_at=benchmark_started_at,
                databases_tested=databases_tested,
                num_queries=num_queries,
                top_k=top_k,
                repetitions=repetitions,
                scalability_queries=scalability_queries,
                database_inventory_before_run=database_inventory_before_run,
                completed_phases={
                    **completed_phases,
                    'speed_test_partial': all_results,
                },
                current_phase="speed_test",
            ),
        )

    status_text.text("Speed test completed, running top-k sensitivity test...")

    dfs = {db_name: pd.DataFrame(results) for db_name, results in all_results.items()}
    combined_df = pd.concat(dfs.values(), ignore_index=True)

    st.session_state['benchmark_results'] = combined_df
    st.session_state['dfs'] = dfs
    st.session_state['tested_config'] = current_config
    completed_phases['speed_test'] = combined_df.to_dict(orient='records')
    write_checkpoint(run_dir, "01_speed_test.json", {'speed_test': completed_phases['speed_test']})
    write_latest_checkpoint(
        run_dir,
        build_partial_export(
            status="running",
            benchmark_started_at=benchmark_started_at,
            databases_tested=databases_tested,
            num_queries=num_queries,
            top_k=top_k,
            repetitions=repetitions,
            scalability_queries=scalability_queries,
            database_inventory_before_run=database_inventory_before_run,
            completed_phases=completed_phases,
            current_phase="top_k_sensitivity",
        ),
    )

    # ========== PHASE 2: TOP-K SENSITIVITY TEST ==========
    top_k_trials = {
        db_name: {k: [] for k in top_k_levels} for db_name in vector_stores.keys()
    }

    for repetition in range(1, repetitions + 1):
        for k in top_k_levels:
            status_text.text(f"Top-K Sensitivity [Run {repetition}/{repetitions}] [k={k}]...")
            for db_name, vector_store in vector_stores.items():
                try:
                    result = measure_top_k_sensitivity_single_k(vector_store, benchmark_queries, k)
                except RuntimeError as e:
                    if db_name != 'Supabase':
                        raise
                    st.warning(
                        f"Top-K sensitivity failed for Supabase at k={k} "
                        f"(run {repetition}): {e} — skipping this level, continuing."
                    )
                    result = None
                if result:
                    top_k_trials[db_name][k].append({'run': repetition, **result})
                current_test += 1
                progress_bar.progress(current_test / total_steps)

    top_k_sensitivity_results = {
        db_name: [
            summarize_repeated_measurements(top_k_trials[db_name][k], 'top_k', k)
            for k in top_k_levels
            if top_k_trials[db_name][k]
        ]
        for db_name in vector_stores.keys()
    }
    st.session_state['top_k_sensitivity_results'] = top_k_sensitivity_results
    completed_phases['top_k_sensitivity_test'] = top_k_sensitivity_results
    write_checkpoint(run_dir, "02_top_k_sensitivity.json", {'top_k_sensitivity_test': top_k_sensitivity_results})
    write_latest_checkpoint(
        run_dir,
        build_partial_export(
            status="running",
            benchmark_started_at=benchmark_started_at,
            databases_tested=databases_tested,
            num_queries=num_queries,
            top_k=top_k,
            repetitions=repetitions,
            scalability_queries=scalability_queries,
            database_inventory_before_run=database_inventory_before_run,
            completed_phases=completed_phases,
            current_phase="corpus_size_scalability",
        ),
    )

    # ========== PHASE 3: CORPUS-SIZE SCALABILITY ==========
    status_text.text("Running corpus-size scalability test...")

    corpus_scalability_trials = {
        db_name: {doc_count: [] for doc_count in SCALABILITY_DOC_COUNTS}
        for db_name in scalability_vector_stores.keys()
    }

    for repetition in range(1, repetitions + 1):
        for doc_count in SCALABILITY_DOC_COUNTS:
            status_text.text(
                f"Corpus Scalability [Run {repetition}/{repetitions}] [docs={doc_count}]..."
            )
            for db_name, stores_by_size in scalability_vector_stores.items():
                try:
                    result = measure_corpus_scalability_single_run(
                        stores_by_size[doc_count], scalability_queries, top_k
                    )
                except Exception as e:
                    collection_name = build_scalability_collection_name(COLLECTION_NAME, doc_count)
                    failure_message = (
                        f"Corpus scalability failed for {db_name}, docs={doc_count}, "
                        f"collection={collection_name}: {e}"
                    )
                    if db_name != 'Supabase':
                        st.error(failure_message)
                        raise
                    st.warning(failure_message + " — skipping this level, continuing.")
                    result = None
                if result:
                    corpus_scalability_trials[db_name][doc_count].append(
                        {'run': repetition, **result}
                    )
                current_test += 1
                progress_bar.progress(current_test / total_steps)

    corpus_scalability_results = {
        db_name: [
            summarize_repeated_measurements(
                corpus_scalability_trials[db_name][doc_count], 'doc_count', doc_count
            )
            for doc_count in SCALABILITY_DOC_COUNTS
            if corpus_scalability_trials[db_name][doc_count]
        ]
        for db_name in scalability_vector_stores.keys()
    }
    st.session_state['corpus_scalability_results'] = corpus_scalability_results
    completed_phases['corpus_size_scalability_test'] = corpus_scalability_results
    write_checkpoint(run_dir, "03_corpus_scalability.json", {'corpus_size_scalability_test': corpus_scalability_results})
    write_latest_checkpoint(
        run_dir,
        build_partial_export(
            status="running",
            benchmark_started_at=benchmark_started_at,
            databases_tested=databases_tested,
            num_queries=num_queries,
            top_k=top_k,
            repetitions=repetitions,
            scalability_queries=scalability_queries,
            database_inventory_before_run=database_inventory_before_run,
            completed_phases=completed_phases,
            current_phase="concurrent_user_scalability",
        ),
    )

    # ========== PHASE 4: CONCURRENT USER SCALABILITY ==========
    status_text.text("Running concurrent-user scalability test...")

    concurrent_scalability_trials = {
        db_name: {user_count: [] for user_count in CONCURRENT_USER_LEVELS}
        for db_name in vector_stores.keys()
    }
    concurrent_skip_reasons = {db_name: None for db_name in vector_stores.keys()}

    for repetition in range(1, repetitions + 1):
        for user_count in CONCURRENT_USER_LEVELS:
            status_text.text(
                f"Concurrent Users [Run {repetition}/{repetitions}] [users={user_count}]..."
            )
            for db_name, vector_store in vector_stores.items():
                if concurrent_skip_reasons[db_name]:
                    result = {
                        'concurrent_users': user_count,
                        'queries_per_user': CONCURRENT_QUERIES_PER_USER,
                        'total_requests': user_count * CONCURRENT_QUERIES_PER_USER,
                        'successful_requests': 0,
                        'errors': 0,
                        'error_rate': None,
                        'skipped': True,
                        'skip_reason': concurrent_skip_reasons[db_name],
                    }
                else:
                    result = measure_concurrent_users(
                        vector_store,
                        benchmark_queries,
                        top_k,
                        user_count,
                        CONCURRENT_QUERIES_PER_USER,
                    )
                    result['skipped'] = False

                    if result.get('successful_requests', 0) == 0:
                        concurrent_skip_reasons[db_name] = (
                            f"users={user_count} had zero successful requests"
                        )
                    elif result.get('error_rate', 0) is not None and result.get('error_rate', 0) >= CONCURRENT_SKIP_ERROR_RATE:
                        concurrent_skip_reasons[db_name] = (
                            f"users={user_count} error_rate={result.get('error_rate'):.2%} "
                            f">= {CONCURRENT_SKIP_ERROR_RATE:.2%}"
                        )
                    elif result.get('duration_seconds', 0) >= CONCURRENT_SKIP_DURATION_SECONDS:
                        concurrent_skip_reasons[db_name] = (
                            f"users={user_count} duration={result.get('duration_seconds'):.2f}s "
                            f">= {CONCURRENT_SKIP_DURATION_SECONDS}s"
                        )

                concurrent_scalability_trials[db_name][user_count].append(
                    {'run': repetition, **result}
                )
                current_test += 1
                progress_bar.progress(current_test / total_steps)

            write_latest_checkpoint(
                run_dir,
                build_partial_export(
                    status="running",
                    benchmark_started_at=benchmark_started_at,
                    databases_tested=databases_tested,
                    num_queries=num_queries,
                    top_k=top_k,
                    repetitions=repetitions,
                    scalability_queries=scalability_queries,
                    database_inventory_before_run=database_inventory_before_run,
                    completed_phases={
                        **completed_phases,
                        'concurrent_user_scalability_partial': concurrent_scalability_trials,
                        'concurrent_skip_reasons': concurrent_skip_reasons,
                    },
                    current_phase="concurrent_user_scalability",
                ),
            )

    concurrent_scalability_results = {
        db_name: [
            {
                'concurrent_users': user_count,
                'runs': len(results),
                'skipped_runs': int(sum(1 for r in results if r.get('skipped'))),
                'skip_reasons': sorted({r.get('skip_reason') for r in results if r.get('skip_reason')}),
                'mean_latency_ms': safe_mean([r['mean_latency_ms'] for r in results if 'mean_latency_ms' in r]),
                'p95_latency_ms': safe_mean([r['p95_latency_ms'] for r in results if 'p95_latency_ms' in r]),
                'p99_latency_ms': safe_mean([r['p99_latency_ms'] for r in results if 'p99_latency_ms' in r]),
                'throughput_rps': safe_mean([r['throughput_rps'] for r in results if 'throughput_rps' in r]),
                'error_rate': safe_mean([r['error_rate'] for r in results if r.get('error_rate') is not None]),
                'avg_cpu_percent': safe_mean([r.get('avg_cpu_percent', 0) for r in results]),
                'max_cpu_percent': safe_max([r.get('max_cpu_percent', 0) for r in results]),
                'avg_ram_used_mb': safe_mean([r.get('avg_ram_used_mb', 0) for r in results]),
                'max_ram_used_mb': safe_max([r.get('max_ram_used_mb', 0) for r in results]),
                'avg_gpu_util_percent': safe_mean([r.get('avg_gpu_util_percent', 0) for r in results]),
                'max_gpu_util_percent': safe_max([r.get('max_gpu_util_percent', 0) for r in results]),
                'avg_gpu_memory_used_mb': safe_mean([r.get('avg_gpu_memory_used_mb', 0) for r in results]),
                'max_gpu_memory_used_mb': safe_max([r.get('max_gpu_memory_used_mb', 0) for r in results]),
                'per_run': results,
            }
            for user_count, results in user_results.items()
            if results
        ]
        for db_name, user_results in concurrent_scalability_trials.items()
    }
    st.session_state['concurrent_scalability_results'] = concurrent_scalability_results
    completed_phases['concurrent_user_scalability_test'] = concurrent_scalability_results
    write_checkpoint(
        run_dir,
        "04_concurrent_users.json",
        {'concurrent_user_scalability_test': concurrent_scalability_results},
    )
    # ========== AUTO-SAVE JSON ==========
    status_text.text("Saving results...")

    benchmark_completed_at = datetime.now()
    valid_dfs = {db_name: successful_rows(df) for db_name, df in dfs.items()}
    failure_summary = {db_name: dataframe_success_summary(df) for db_name, df in dfs.items()}
    avg_retrieval_times = {db_name: df['retrieval_time'].mean() for db_name, df in valid_dfs.items()}
    winner = min(avg_retrieval_times, key=avg_retrieval_times.get)
    winner_time = avg_retrieval_times[winner]
    other_dbs = {k: v for k, v in avg_retrieval_times.items() if k != winner}
    speed_improvement = ((max(other_dbs.values()) - winner_time) / max(other_dbs.values()) * 100) if other_dbs else 0

    export_data = {
        'metadata': {
            'status': 'completed',
            'run_id': benchmark_started_at.strftime('%Y%m%d_%H%M%S'),
            'benchmark_started_at': benchmark_started_at.isoformat(),
            'benchmark_completed_at': benchmark_completed_at.isoformat(),
            'duration_seconds': round((benchmark_completed_at - benchmark_started_at).total_seconds(), 2),
            'llm_model': CHAT_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'collection_name': COLLECTION_NAME,
            'num_queries': num_queries,
            'unique_queries_available': len(ANSWERABLE_QUERIES),
            'repetitions': repetitions,
            'top_k': top_k,
            'scalability_doc_counts': SCALABILITY_DOC_COUNTS,
            'scalability_query_count': len(scalability_queries),
            'databases_tested': databases_tested,
            'quality_evaluation_included': False,
        },
        'benchmark_plan': {
            'thesis_title': THESIS_TITLE,
            'databases_tested': databases_tested,
            'num_queries': num_queries,
            'repetitions': repetitions,
            'top_k': top_k,
            'top_k_levels': top_k_levels,
            'scalability_doc_counts': SCALABILITY_DOC_COUNTS,
            'concurrent_user_levels': CONCURRENT_USER_LEVELS,
            'concurrent_queries_per_user': CONCURRENT_QUERIES_PER_USER,
            'deepeval_run_mode': 'separate',
            'random_seed': RANDOM_SEED,
            'speed_test_total_rows_expected': len(databases_tested) * num_queries * repetitions,
            'top_k_sensitivity_query_count': len(benchmark_queries),
            'corpus_scalability_query_count': len(scalability_queries),
            'metric_scope_notes': {
                'retrieval_time_ms': 'Includes query embedding plus vector-store similarity search because LangChain similarity_search embeds the query internally.',
                'resource_metrics': 'CPU, RAM, and GPU metrics are host/client-side measurements during retrieval-only concurrent tests; managed backend server resources such as Pinecone infrastructure are not directly measured.',
                'response_time_ms': 'Sequential speed test total_time is retrieval_time plus LLM generation time.'
            },
        },
        'database_inventory_before_run': database_inventory_before_run,
        'database_limitations': get_database_limitations(databases_tested),
        'query_manifest': get_query_manifest(randomized_queries[:num_queries]),
        'scalability_query_manifest': get_query_manifest(scalability_queries),
        'document_manifest': get_document_manifest(),
        'environment': {
            'score_threshold': SCORE_THRESHOLD,
            'top_k_env_default': TOP_K,
            'sqlite_db_path': SQLITE_DB_PATH,
            'lancedb_path': LANCEDB_PATH,
            'qdrant_url': QDRANT_URL,
            'pinecone_index_name': os.getenv("PINECONE_INDEX_NAME", "rag-vector-benchmark"),
            'postgres_host': os.getenv('DB_HOST', 'localhost'),
            'postgres_port': os.getenv('DB_PORT', '5432'),
            'postgres_database': os.getenv('DB_NAME', 'ragdb'),
            'supabase_host': os.getenv('SUPABASE_DB_HOST', ''),
            'supabase_port': os.getenv('SUPABASE_DB_PORT', '5432'),
            'firebase_project_id': os.getenv('FIREBASE_PROJECT_ID', ''),
            'firebase_database_id': os.getenv('FIREBASE_DATABASE_ID', '(default)'),
        },
        'speed_test': {
            'winner': {'database': winner, 'avg_retrieval_ms': round(winner_time, 2), 'speed_improvement_percent': round(speed_improvement, 1)},
            'failure_summary': failure_summary,
            'summary': [
                {
                    'database': db_name,
                    **failure_summary[db_name],
                    'mean_total_ms': round(valid_df['total_time'].mean(), 2),
                    'median_total_ms': round(valid_df['total_time'].median(), 2),
                    'std_total_ms': round(valid_df['total_time'].std(), 2),
                    'min_total_ms': round(valid_df['total_time'].min(), 2),
                    'max_total_ms': round(valid_df['total_time'].max(), 2),
                    'p95_total_ms': round(valid_df['total_time'].quantile(0.95), 2),
                    'mean_retrieval_ms': round(valid_df['retrieval_time'].mean(), 2),
                    'p95_retrieval_ms': round(valid_df['retrieval_time'].quantile(0.95), 2),
                    'mean_llm_ms': round(valid_df['llm_time'].mean(), 2),
                } for db_name, valid_df in valid_dfs.items()
            ],
            'per_repetition_summary': {
                db_name: [
                    {
                        'repetition': int(run_id),
                        **dataframe_success_summary(run_df),
                        'mean_retrieval_ms': round(successful_rows(run_df)['retrieval_time'].mean(), 2),
                        'mean_total_ms': round(successful_rows(run_df)['total_time'].mean(), 2),
                        'p95_retrieval_ms': round(successful_rows(run_df)['retrieval_time'].quantile(0.95), 2),
                        'p95_total_ms': round(successful_rows(run_df)['total_time'].quantile(0.95), 2),
                    }
                    for run_id, run_df in df.groupby('repetition')
                ]
                for db_name, df in dfs.items()
            },
            'query_type_summary': {
                db_name: {
                    query_type: {
                        'queries_tested': int(len(type_df)),
                        **dataframe_success_summary(type_df),
                        'mean_retrieval_ms': round(successful_rows(type_df)['retrieval_time'].mean(), 2),
                        'mean_total_ms': round(successful_rows(type_df)['total_time'].mean(), 2),
                    }
                    for query_type, type_df in df.groupby('query_type')
                }
                for db_name, df in dfs.items()
            },
            'raw_results': combined_df.to_dict(orient='records')
        },
        'top_k_sensitivity_test': top_k_sensitivity_results,
        'corpus_size_scalability_test': corpus_scalability_results,
        'concurrent_user_scalability_test': concurrent_scalability_results,
    }
    # Store export data in session for download button
    st.session_state['export_data'] = sanitize_for_json(export_data)
    final_path = write_checkpoint(run_dir, "benchmark_final.json", st.session_state['export_data'])
    write_latest_checkpoint(run_dir, st.session_state['export_data'])
    st.session_state['export_path'] = final_path

    status_text.text("Completed!")
    progress_bar.progress(1.0)


# Display Results
if 'benchmark_results' in st.session_state:
    combined_df = st.session_state['benchmark_results']
    dfs = st.session_state['dfs']
    valid_dfs = {db_name: successful_rows(df) for db_name, df in dfs.items()}
    
    st.markdown("---")
    
    # 🏆 Winner Banner - Based on RETRIEVAL TIME (database performance only)
    avg_retrieval_times = {db_name: df['retrieval_time'].mean() for db_name, df in valid_dfs.items()}
    winner = min(avg_retrieval_times, key=avg_retrieval_times.get)
    winner_time = avg_retrieval_times[winner]
    
    # Calculate how much faster winner is compared to others
    other_dbs = {k: v for k, v in avg_retrieval_times.items() if k != winner}
    if other_dbs:
        slowest = max(other_dbs.values())
        speed_improvement = ((slowest - winner_time) / slowest) * 100
    else:
        speed_improvement = 0
    
    winner_color = COLORS.get(winner, '#10B981')
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {winner_color}22 0%, {winner_color}11 100%);
        border: 2px solid {winner_color};
        border-radius: 16px;
        padding: 24px 32px;
        margin: 20px 0 30px 0;
        text-align: center;
    ">
        <div style="font-size: 2rem; margin-bottom: 8px; color: #10B981;">WINNER</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {winner_color}; margin-bottom: 4px;">
            {winner} Wins!
        </div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #F8FAFC; margin: 8px 0;">
            {winner_time:.1f}ms Retrieval
        </div>
        <div style="font-size: 1rem; color: #94A3B8;">
            {speed_improvement:.1f}% faster than the slowest database
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Cards
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(valid_dfs))
    for idx, (db_name, df) in enumerate(valid_dfs.items()):
        badge_class = f'badge-{db_name.lower().replace(" ", "").replace("+", "")}'
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <span class="db-badge {badge_class}">{db_name}</span>
                <div class="metric-value">{df['total_time'].mean():.0f}ms</div>
                <div class="metric-label">Average Response</div>
                <div style="margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div>
                        <div style="color: #3B82F6; font-weight: 600; font-size: 1.25rem;">{df['retrieval_time'].mean():.1f}ms</div>
                        <div style="color: #94A3B8; font-size: 0.75rem;">Retrieval</div>
                    </div>
                    <div>
                        <div style="color: #EC4899; font-weight: 600; font-size: 1.25rem;">{df['llm_time'].mean():.1f}ms</div>
                        <div style="color: #94A3B8; font-size: 0.75rem;">LLM</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Chart 1: Sequential latency overview
    st.markdown('<div class="section-header">Sequential Query Latency</div>', unsafe_allow_html=True)
    
    fig1 = go.Figure()
    
    for db_name, df in dfs.items():
        color = COLORS.get(db_name, '#94A3B8')
        
        # Filter out failed queries (where total_time == 0)
        df_valid = df[df['total_time'] > 0]
        
        # Calculate average for this database
        avg_time = df_valid['total_time'].mean()
        
        # Main response time line
        fig1.add_trace(go.Scatter(
            x=df_valid['global_query_num'],
            y=df_valid['total_time'],
            mode='lines+markers',
            name=f'{db_name}',
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate=f'<b>{db_name}</b><br>Sequence: %{{x}}<br>Time: %{{y:.1f}}ms<extra></extra>'
        ))
        
        # Average line (red dashed)
        fig1.add_trace(go.Scatter(
            x=[df_valid['global_query_num'].min(), df_valid['global_query_num'].max()],
            y=[avg_time, avg_time],
            mode='lines',
            name=f'Average ({avg_time:.0f}ms)',
            line=dict(color='#EF4444', width=2, dash='dash'),
            hovertemplate=f'<b>Average</b>: {avg_time:.1f}ms<extra></extra>'
        ))
    
    fig1.update_layout(
        title=dict(
            text='<b>Response Time Across Sequential Queries</b>',
            font=dict(size=24, color='#F8FAFC')
        ),
        xaxis=dict(
            title='Global Query Sequence',
            gridcolor='rgba(148, 163, 184, 0.1)',
            showgrid=True,
            title_font=dict(size=14, color='#94A3B8'),
            tickfont=dict(size=12, color='#94A3B8')
        ),
        yaxis=dict(
            title='Response Time (ms)',
            gridcolor='rgba(148, 163, 184, 0.1)',
            showgrid=True,
            title_font=dict(size=14, color='#94A3B8'),
            tickfont=dict(size=12, color='#94A3B8')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#F8FAFC'),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(148, 163, 184, 0.3)',
            borderwidth=1,
            font=dict(size=13)
        )
    )
    
    st.plotly_chart(fig1, width="stretch")
    
    # Chart 2: Stacked Bar Comparison
    st.markdown('<div class="section-header">Speed Breakdown</div>', unsafe_allow_html=True)
    
    fig2 = go.Figure()
    
    db_names = list(valid_dfs.keys())
    
    # Retrieval bars
    fig2.add_trace(go.Bar(
        name='Retrieval Time',
        x=db_names,
        y=[valid_dfs[db]['retrieval_time'].mean() for db in db_names],
        marker=dict(
            color=COLORS['primary'],
            line=dict(color='rgba(255,255,255,0.2)', width=2)
        ),
        text=[f"{valid_dfs[db]['retrieval_time'].mean():.1f}ms" for db in db_names],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{x}</b><br>Retrieval: %{y:.1f}ms<extra></extra>'
    ))
    
    # LLM bars
    fig2.add_trace(go.Bar(
        name='LLM Generation',
        x=db_names,
        y=[valid_dfs[db]['llm_time'].mean() for db in db_names],
        marker=dict(
            color=COLORS['muted'],
            line=dict(color='rgba(255,255,255,0.2)', width=2)
        ),
        text=[f"{valid_dfs[db]['llm_time'].mean():.1f}ms" for db in db_names],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{x}</b><br>LLM: %{y:.1f}ms<extra></extra>'
    ))
    
    fig2.update_layout(
        title=dict(
            text='<b>Average Time Components</b>',
            font=dict(size=24, color='#F8FAFC')
        ),
        barmode='stack',
        xaxis=dict(
            title='',
            tickfont=dict(size=13, color='#F8FAFC')
        ),
        yaxis=dict(
            title='Time (ms)',
            gridcolor='rgba(148, 163, 184, 0.1)',
            title_font=dict(size=14, color='#94A3B8'),
            tickfont=dict(size=12, color='#94A3B8')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#F8FAFC'),
        height=450,
        bargap=0.2,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(148, 163, 184, 0.3)',
            borderwidth=1,
            font=dict(size=13)
        )
    )
    
    st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="section-header">Repeated Runs</div>', unsafe_allow_html=True)

    repetition_stats = []
    for db_name, df in valid_dfs.items():
        for repetition, repetition_df in df.groupby('repetition'):
            repetition_stats.append({
                'Database': db_name,
                'Run': int(repetition),
                'Mean Retrieval (ms)': repetition_df['retrieval_time'].mean(),
                'P95 Retrieval (ms)': repetition_df['retrieval_time'].quantile(0.95),
                'Mean Total (ms)': repetition_df['total_time'].mean(),
                'P95 Total (ms)': repetition_df['total_time'].quantile(0.95),
            })

    repetition_stats_df = pd.DataFrame(repetition_stats)
    st.dataframe(
        repetition_stats_df.style.format({
            'Mean Retrieval (ms)': '{:.2f}',
            'P95 Retrieval (ms)': '{:.2f}',
            'Mean Total (ms)': '{:.2f}',
            'P95 Total (ms)': '{:.2f}',
        }),
        width="stretch",
        hide_index=True,
    )
    
    # Statistics Table
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    
    stats_data = []
    for db_name, df in dfs.items():
        stats_data.append({
            'Database': db_name,
            'Mean (ms)': df['total_time'].mean(),
            'Median (ms)': df['total_time'].median(),
            'P95 (ms)': df['total_time'].quantile(0.95),
            'Std Dev': df['total_time'].std(),
            'Min (ms)': df['total_time'].min(),
            'Max (ms)': df['total_time'].max()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    st.dataframe(
        stats_df.style.format({
            'Mean (ms)': '{:.2f}',
            'Median (ms)': '{:.2f}',
            'P95 (ms)': '{:.2f}',
            'Std Dev': '{:.2f}',
            'Min (ms)': '{:.2f}',
            'Max (ms)': '{:.2f}'
        }),
        width="stretch",
        hide_index=True
    )

    st.markdown('<div class="section-header">Latency By Query Type</div>', unsafe_allow_html=True)

    latency_by_type = []
    for db_name, df in dfs.items():
        for query_type, type_df in df.groupby('query_type'):
            latency_by_type.append({
                'Database': db_name,
                'Query Type': query_type,
                'Queries Tested': len(type_df),
                'Mean Retrieval (ms)': type_df['retrieval_time'].mean(),
                'Mean Total (ms)': type_df['total_time'].mean(),
            })

    latency_by_type_df = pd.DataFrame(latency_by_type)
    st.dataframe(
        latency_by_type_df.style.format({
            'Mean Retrieval (ms)': '{:.2f}',
            'Mean Total (ms)': '{:.2f}',
        }),
        width="stretch",
        hide_index=True,
    )

    st.caption("All benchmark queries in this run are answerable ground-truth queries.")
    
    # Query Scoreboard - Show all queries sorted by fastest
    st.markdown("---")
    
    st.markdown('<div class="section-header">Query Scoreboard</div>', unsafe_allow_html=True)
    
    # Create scoreboard dataframe - sorted by fastest (default)
    scoreboard_df = combined_df[['repetition', 'query_num', 'database', 'query_type', 'query', 'retrieval_time', 'llm_time', 'total_time']].copy()
    scoreboard_df = scoreboard_df.rename(columns={
        'repetition': 'Run',
        'query_num': '#',
        'database': 'Database',
        'query_type': 'Type',
        'query': 'Query',
        'retrieval_time': 'Retrieval (ms)',
        'llm_time': 'LLM (ms)',
        'total_time': 'Total (ms)'
    })
    
    # Sort by fastest (ascending total time)
    scoreboard_df = scoreboard_df.sort_values('Total (ms)', ascending=True)
    
    # Truncate query text for display
    scoreboard_df['Query'] = scoreboard_df['Query'].apply(lambda x: x[:40] + '...' if len(x) > 40 else x)
    
    st.dataframe(
        scoreboard_df.style.format({
            'Retrieval (ms)': '{:.1f}',
            'LLM (ms)': '{:.1f}',
            'Total (ms)': '{:.1f}'
        }),
        width="stretch",
        hide_index=True,
        height=400
    )

if 'top_k_sensitivity_results' in st.session_state:
    top_k_sensitivity_results = st.session_state['top_k_sensitivity_results']
    
    st.markdown("---")
    st.markdown('<div class="section-header">Top-K Sensitivity</div>', unsafe_allow_html=True)
    
    fig_scale = go.Figure()
    
    for db_name, results in top_k_sensitivity_results.items():
        color = COLORS.get(db_name, '#94A3B8')
        x_vals = [r['top_k'] for r in results]
        y_vals = [r['mean_avg_time'] for r in results]
        
        fig_scale.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers', name=db_name,
            line=dict(color=color, width=3), marker=dict(size=10, color=color),
            hovertemplate=f'<b>{db_name}</b><br>Top-K: %{{x}}<br>Mean Avg Time: %{{y:.2f}}ms<extra></extra>'
        ))
    
    fig_scale.update_layout(
        title=dict(text='<b>Retrieval Time vs Retrieval Volume (Top-K)</b>', font=dict(size=24, color='#F8FAFC')),
        xaxis=dict(title='Top-K (Documents Retrieved)', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        yaxis=dict(title='Average Retrieval Time (ms)', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#F8FAFC'), height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.3)', borderwidth=1, font=dict(size=13))
    )
    
    st.plotly_chart(fig_scale, width="stretch")
    
    # Top-K sensitivity statistics
    st.markdown('<div class="section-header">Top-K Sensitivity Statistics</div>', unsafe_allow_html=True)
    
    scale_stats = []
    for db_name, results in top_k_sensitivity_results.items():
        if results:
            avg_times = [r['mean_avg_time'] for r in results]
            scale_stats.append({
                'Database': db_name,
                'Min Avg (ms)': min(avg_times),
                'Max Avg (ms)': max(avg_times),
                'Growth Rate': f"{((max(avg_times) - min(avg_times)) / min(avg_times) * 100):.1f}%"
            })
    
    scale_df = pd.DataFrame(scale_stats)
    st.dataframe(scale_df.style.format({'Min Avg (ms)': '{:.2f}', 'Max Avg (ms)': '{:.2f}'}), width="stretch", hide_index=True)

if 'corpus_scalability_results' in st.session_state:
    corpus_scalability_results = st.session_state['corpus_scalability_results']

    st.markdown("---")
    st.markdown('<div class="section-header">Corpus-Size Scalability</div>', unsafe_allow_html=True)

    fig_corpus = go.Figure()

    for db_name, results in corpus_scalability_results.items():
        color = COLORS.get(db_name, '#94A3B8')
        x_vals = [r['doc_count'] for r in results]
        y_vals = [r['mean_avg_time'] for r in results]

        fig_corpus.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=db_name,
            line=dict(color=color, width=3),
            marker=dict(size=10, color=color),
            hovertemplate=f'<b>{db_name}</b><br>Documents: %{{x}}<br>Mean Avg Retrieval: %{{y:.2f}}ms<extra></extra>'
        ))

    fig_corpus.update_layout(
        title=dict(text='<b>Retrieval Latency vs Corpus Size</b>', font=dict(size=24, color='#F8FAFC')),
        xaxis=dict(title='Number of Documents in Corpus', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        yaxis=dict(title='Average Retrieval Time (ms)', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#F8FAFC'), height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.3)', borderwidth=1, font=dict(size=13))
    )

    st.plotly_chart(fig_corpus, width="stretch")

    corpus_stats = []
    for db_name, results in corpus_scalability_results.items():
        for result in results:
            corpus_stats.append({
                'Database': db_name,
                'Documents': result['doc_count'],
                'Queries Tested': result['query_count'],
                'Runs': result['runs'],
                'Mean Avg Retrieval (ms)': result['mean_avg_time'],
                'Mean Median Retrieval (ms)': result['mean_median_time'],
                'Mean P95 Retrieval (ms)': result['mean_p95_time'],
                'Std Avg Retrieval (ms)': result['std_avg_time'],
            })

    corpus_stats_df = pd.DataFrame(corpus_stats)
    st.dataframe(
        corpus_stats_df.style.format({
            'Mean Avg Retrieval (ms)': '{:.2f}',
            'Mean Median Retrieval (ms)': '{:.2f}',
            'Mean P95 Retrieval (ms)': '{:.2f}',
            'Std Avg Retrieval (ms)': '{:.2f}',
        }),
        width="stretch",
        hide_index=True,
    )

if 'concurrent_scalability_results' in st.session_state:
    concurrent_scalability_results = st.session_state['concurrent_scalability_results']

    st.markdown("---")
    st.markdown('<div class="section-header">Concurrent User Scalability</div>', unsafe_allow_html=True)

    fig_concurrent = go.Figure()
    for db_name, results in concurrent_scalability_results.items():
        color = COLORS.get(db_name, '#94A3B8')
        fig_concurrent.add_trace(go.Scatter(
            x=[r['concurrent_users'] for r in results],
            y=[r['p95_latency_ms'] for r in results],
            mode='lines+markers',
            name=f'{db_name} P95 Latency',
            line=dict(color=color, width=3),
            marker=dict(size=10, color=color),
            hovertemplate=f'<b>{db_name}</b><br>Users: %{{x}}<br>P95: %{{y:.2f}}ms<extra></extra>'
        ))

    fig_concurrent.update_layout(
        title=dict(text='<b>P95 Retrieval Latency vs Concurrent Users</b>', font=dict(size=24, color='#F8FAFC')),
        xaxis=dict(title='Concurrent Users', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        yaxis=dict(title='P95 Retrieval Latency (ms)', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#F8FAFC'), height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.3)', borderwidth=1, font=dict(size=13))
    )
    st.plotly_chart(fig_concurrent, width="stretch")

    concurrent_stats = []
    for db_name, results in concurrent_scalability_results.items():
        for result in results:
            concurrent_stats.append({
                'Database': db_name,
                'Concurrent Users': result['concurrent_users'],
                'Runs': result['runs'],
                'Throughput (req/s)': result['throughput_rps'],
                'Mean Latency (ms)': result['mean_latency_ms'],
                'P95 Latency (ms)': result['p95_latency_ms'],
                'P99 Latency (ms)': result['p99_latency_ms'],
                'Error Rate': result['error_rate'],
                'Avg CPU (%)': result['avg_cpu_percent'],
                'Max CPU (%)': result['max_cpu_percent'],
                'Avg RAM (MB)': result['avg_ram_used_mb'],
                'Max RAM (MB)': result['max_ram_used_mb'],
                'Avg GPU (%)': result['avg_gpu_util_percent'],
                'Max GPU (%)': result['max_gpu_util_percent'],
                'Avg GPU Mem (MB)': result['avg_gpu_memory_used_mb'],
                'Max GPU Mem (MB)': result['max_gpu_memory_used_mb'],
            })

    concurrent_stats_df = pd.DataFrame(concurrent_stats)
    st.dataframe(
        concurrent_stats_df.style.format({
            'Throughput (req/s)': '{:.2f}',
            'Mean Latency (ms)': '{:.2f}',
            'P95 Latency (ms)': '{:.2f}',
            'P99 Latency (ms)': '{:.2f}',
            'Error Rate': '{:.2%}',
            'Avg CPU (%)': '{:.2f}',
            'Max CPU (%)': '{:.2f}',
            'Avg RAM (MB)': '{:.2f}',
            'Max RAM (MB)': '{:.2f}',
            'Avg GPU (%)': '{:.2f}',
            'Max GPU (%)': '{:.2f}',
            'Avg GPU Mem (MB)': '{:.2f}',
            'Max GPU Mem (MB)': '{:.2f}',
        }),
        width="stretch",
        hide_index=True,
    )

# ========== DEEPEVAL QUALITY RESULTS ==========
if 'deepeval_quality_results' in st.session_state:
    deepeval_quality_results = st.session_state['deepeval_quality_results']
    
    st.markdown("---")
    st.markdown('<div class="section-header">DeepEval Answer Quality</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(deepeval_quality_results))
    for idx, (db_name, results) in enumerate(deepeval_quality_results.items()):
        if results:
            avg_answer_relevancy = np.mean([r.get('AnswerRelevancy_score', 0) for r in results])
            avg_faithfulness = np.mean([r.get('Faithfulness_score', 0) for r in results])
            avg_contextual_recall = np.mean([r.get('ContextualRecall_score', 0) for r in results])
            
            badge_class = f'badge-{db_name.lower().replace(" ", "").replace("+", "")}'
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="db-badge {badge_class}">{db_name}</span>
                    <div class="metric-value">{avg_faithfulness:.2%}</div>
                    <div class="metric-label">Faithfulness</div>
                    <div style="margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div>
                            <div style="color: #22C55E; font-weight: 600; font-size: 1.25rem;">{avg_answer_relevancy:.2%}</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Answer Relevancy</div>
                        </div>
                        <div>
                            <div style="color: #3B82F6; font-weight: 600; font-size: 1.25rem;">{avg_contextual_recall:.2%}</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Contextual Recall</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    fig_quality = go.Figure()
    
    db_names = list(deepeval_quality_results.keys())
    answer_relevancy = [np.mean([r.get('AnswerRelevancy_score', 0) for r in deepeval_quality_results[db]]) for db in db_names]
    faithfulness = [np.mean([r.get('Faithfulness_score', 0) for r in deepeval_quality_results[db]]) for db in db_names]
    contextual_relevancy = [np.mean([r.get('ContextualRelevancy_score', 0) for r in deepeval_quality_results[db]]) for db in db_names]
    contextual_precision = [np.mean([r.get('ContextualPrecision_score', 0) for r in deepeval_quality_results[db]]) for db in db_names]
    contextual_recall = [np.mean([r.get('ContextualRecall_score', 0) for r in deepeval_quality_results[db]]) for db in db_names]
    
    fig_quality.add_trace(go.Bar(name='Answer Relevancy', x=db_names, y=answer_relevancy, marker=dict(color='#22C55E'), text=[f"{v:.1%}" for v in answer_relevancy], textposition='outside', textfont=dict(size=12, color='#22C55E')))
    fig_quality.add_trace(go.Bar(name='Faithfulness', x=db_names, y=faithfulness, marker=dict(color='#3B82F6'), text=[f"{v:.1%}" for v in faithfulness], textposition='outside', textfont=dict(size=12, color='#3B82F6')))
    fig_quality.add_trace(go.Bar(name='Contextual Relevancy', x=db_names, y=contextual_relevancy, marker=dict(color='#F59E0B'), text=[f"{v:.1%}" for v in contextual_relevancy], textposition='outside', textfont=dict(size=12, color='#F59E0B')))
    fig_quality.add_trace(go.Bar(name='Contextual Precision', x=db_names, y=contextual_precision, marker=dict(color='#A855F7'), text=[f"{v:.1%}" for v in contextual_precision], textposition='outside', textfont=dict(size=12, color='#A855F7')))
    fig_quality.add_trace(go.Bar(name='Contextual Recall', x=db_names, y=contextual_recall, marker=dict(color='#EC4899'), text=[f"{v:.1%}" for v in contextual_recall], textposition='outside', textfont=dict(size=12, color='#EC4899')))
    
    fig_quality.update_layout(
        title=dict(text='<b>DeepEval RAG Quality Comparison</b>', font=dict(size=24, color='#F8FAFC')),
        barmode='group', xaxis=dict(title='', tickfont=dict(size=13, color='#F8FAFC')),
        yaxis=dict(title='Score', range=[0, 1.1], gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#F8FAFC'), height=450, bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.3)', borderwidth=1, font=dict(size=13))
    )
    
    st.plotly_chart(fig_quality, width="stretch")
    
    answerable_quality_stats = []
    for db_name, results in deepeval_quality_results.items():
        if results:
            answerable_quality_stats.append({
                'Database': db_name,
                'Avg Answer Relevancy': np.mean([r.get('AnswerRelevancy_score', 0) for r in results]),
                'Avg Faithfulness': np.mean([r.get('Faithfulness_score', 0) for r in results]),
                'Avg Contextual Relevancy': np.mean([r.get('ContextualRelevancy_score', 0) for r in results]),
                'Avg Contextual Precision': np.mean([r.get('ContextualPrecision_score', 0) for r in results]),
                'Avg Contextual Recall': np.mean([r.get('ContextualRecall_score', 0) for r in results]),
                'Queries Tested': len(results)
            })
    
    answerable_quality_df = pd.DataFrame(answerable_quality_stats)
    st.dataframe(
        answerable_quality_df.style.format({
            'Avg Answer Relevancy': '{:.2%}',
            'Avg Faithfulness': '{:.2%}',
            'Avg Contextual Relevancy': '{:.2%}',
            'Avg Contextual Precision': '{:.2%}',
            'Avg Contextual Recall': '{:.2%}',
        }),
        width="stretch",
        hide_index=True,
    )


# Download JSON Button
if 'export_data' in st.session_state:
    st.markdown("---")
    import json
    json_data = json.dumps(st.session_state['export_data'], indent=2, allow_nan=False)
    st.download_button(
        label="Download Results (JSON)",
        data=json_data,
        file_name=f"benchmark_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        width="stretch"
    )

if 'deepeval_export_data' in st.session_state:
    st.markdown("---")
    import json
    deepeval_json_data = json.dumps(st.session_state['deepeval_export_data'], indent=2, allow_nan=False)
    st.download_button(
        label="Download DeepEval Quality (JSON)",
        data=deepeval_json_data,
        file_name=f"deepeval_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        width="stretch",
    )
