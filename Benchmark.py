"""
Comprehensive Benchmark System
Comparing Pinecone, PostgreSQL+pgvector, and ChromaDB
For thesis: The Effect of Vector Database Selection on Scalability and Response Speed
"""

import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import random

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.security import build_pg_connection_string, sanitize_error_message

# Vector databases
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector
import chromadb
from langchain_chroma import Chroma

# LangChain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Utils
from utils.document_processor import SAMPLE_QUERIES
from utils.ground_truth import GROUND_TRUTH, calculate_retrieval_metrics

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Title
st.markdown("""
<div style="text-align: center; padding: 32px 0 24px 0;">
    <h1 style="font-size: 2rem; font-weight: 600; color: #FAFAFA; margin-bottom: 4px; letter-spacing: -0.025em;">
        Vector Database Benchmark
    </h1>
    <p style="color: #71717A; font-size: 0.875rem; font-weight: 400;">
        Comparing Scalability and Response Speed
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
    </div>
    """, unsafe_allow_html=True)

# Databases
with config_col2:
    st.markdown('<p style="color: #71717A; font-size: 0.75rem; margin-bottom: 8px;">Databases</p>', unsafe_allow_html=True)
    db_col1, db_col2, db_col3 = st.columns(3)
    with db_col1:
        test_pinecone = st.checkbox("Pinecone", value=True)
    with db_col2:
        test_postgresql = st.checkbox("PostgreSQL", value=True)
    with db_col3:
        test_chroma = st.checkbox("ChromaDB", value=True)

# Parameters
with config_col3:
    st.markdown('<p style="color: #71717A; font-size: 0.75rem; margin-bottom: 8px;">Parameters</p>', unsafe_allow_html=True)
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        num_queries = st.selectbox("Queries", [10, 20, 30, 40, 50, 60, 80, 100], index=7, label_visibility="collapsed")
    with param_col2:
        top_k = st.selectbox("Top K", [1, 2, 3, 4, 5], index=2, label_visibility="collapsed")

if not any([test_pinecone, test_postgresql, test_chroma]):
    st.error("Select at least one database")

# Centered Button
st.markdown("<br>", unsafe_allow_html=True)
_, col_btn1, _ = st.columns([3, 1, 3])
with col_btn1:
    run_benchmark = st.button("Run Benchmark", type="primary", use_container_width=True)

# Documents Section - same width as button
_, col_doc, _ = st.columns([3, 1, 3])
with col_doc:
    with st.expander("View Documents"):
        documents = [
            {
                "name": "General Guidebook for International Students",
                "file": "General-Guidebook-for-International-Students_July-2024.pdf",
                "language": "English",
                "date": "July 2024"
            },
            {
                "name": "Panduan Mahasiswa Baru DPTSI 2025",
                "file": "Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf",
                "language": "Indonesian",
                "date": "2025"
            },
            {
                "name": "Perjanjian Angkutan MRT Jakarta",
                "file": "MAN 01 - Perjanjian Angkutan dengan Penumpang 2021.pdf",
                "language": "Indonesian",
                "date": "2021"
            }
        ]
        
        for i, doc in enumerate(documents):
            st.markdown(f"**{doc['name']}**")
            st.caption(f"{doc['language']} • {doc['date']}")
            
            file_path = os.path.join("documents", doc['file'])
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f.read(),
                        file_name=doc['file'],
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"doc_{i}"
                    )
            if i == 0:
                st.divider()

st.markdown("<br>", unsafe_allow_html=True)


# Initialize vector stores
@st.cache_resource(hash_funcs={bool: lambda x: x})
def init_all_vector_stores(test_pinecone, test_postgresql, test_chroma):
    stores = {}
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if test_pinecone:
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "its-helpdesk-chatbot"))
            stores['Pinecone'] = PineconeVectorStore(index=index, embedding=embeddings)
        except Exception as e:
            logger.error(f"Pinecone connection error: {str(e)}")
    
    if test_postgresql:
        db_password = os.getenv('DB_PASSWORD')
        if not db_password:
            logger.error("PostgreSQL: DB_PASSWORD not set")
        else:
            try:
                connection_string = build_pg_connection_string(
                    user=os.getenv('DB_USER', 'raguser'),
                    password=db_password,
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=os.getenv('DB_PORT', '5432'),
                    database=os.getenv('DB_NAME', 'ragdb')
                )
                stores['PostgreSQL'] = PGVector(
                    embeddings=embeddings,
                    collection_name=os.getenv("COLLECTION_NAME", "its_guidebook"),
                    connection=connection_string,
                    use_jsonb=True,
                )
            except Exception as e:
                logger.error(f"PostgreSQL connection error: {str(e)}")
    
    if test_chroma:
        try:
            client = chromadb.PersistentClient(path="chroma_db")
            stores['ChromaDB'] = Chroma(
                client=client,
                collection_name="its_guidebook",
                embedding_function=embeddings,
            )
        except Exception as e:
            logger.error(f"ChromaDB connection error: {str(e)}")
    
    return stores

def measure_performance(vector_store, query, llm, top_k):
    try:
        start = time.time()
        # Use similarity search (without threshold) for fair comparison
        # All databases will return exactly top_k documents
        docs = vector_store.similarity_search(query, k=top_k)
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
            'success': len(docs) > 0
        }
    except Exception as e:
        return {
            'retrieval_time': 0,
            'llm_time': 0,
            'total_time': 0,
            'num_docs': 0,
            'success': False,
            'error': str(e)
        }


def measure_scalability(vector_store, queries, top_k_levels):
    """
    Measure response time at different Top-K levels (simulating document volume scaling)
    """
    results = []
    for k in top_k_levels:
        level_times = []
        for query in queries[:10]:  # Use subset for speed
            try:
                start = time.time()
                docs = vector_store.similarity_search(query, k=k)
                retrieval_time = (time.time() - start) * 1000
                level_times.append(retrieval_time)
            except:
                pass
        
        if level_times:
            results.append({
                'top_k': k,
                'avg_time': np.mean(level_times),
                'std_time': np.std(level_times),
                'min_time': np.min(level_times),
                'max_time': np.max(level_times)
            })
    
    return results


def measure_retrieval_quality(vector_store, queries, top_k):
    """
    Measure Precision, Recall, and F1-Score for each query using ground truth
    """
    results = []
    for query in queries:
        if query not in GROUND_TRUTH:
            continue
        
        try:
            docs = vector_store.similarity_search(query, k=top_k)
            metrics = calculate_retrieval_metrics(docs, query, top_k)
            results.append({
                'query': query,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'relevant_retrieved': metrics.get('relevant_retrieved', 0),
                'total_retrieved': metrics.get('total_retrieved', 0)
            })
        except Exception as e:
            results.append({
                'query': query,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'error': str(e)
            })
    
    return results




# Main benchmark
if run_benchmark:
    with st.spinner("Initializing..."):
        vector_stores = init_all_vector_stores(test_pinecone, test_postgresql, test_chroma)
    
    if not vector_stores:
        st.error("No vector stores initialized")
        st.stop()
    
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    st.success(f"Initialized {len(vector_stores)} database(s)")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {db_name: [] for db_name in vector_stores.keys()}
    total_tests = len(vector_stores) * num_queries
    current_test = 0
    
    # Randomize queries with fixed seed for reproducibility
    randomized_queries = SAMPLE_QUERIES.copy()
    random.seed(42)
    random.shuffle(randomized_queries)
    
    # Track and clear old results if configuration changed
    current_config = (test_pinecone, test_postgresql, test_chroma)
    if 'tested_config' in st.session_state and st.session_state['tested_config'] != current_config:
        if 'benchmark_results' in st.session_state:
            del st.session_state['benchmark_results']
        if 'dfs' in st.session_state:
            del st.session_state['dfs']
    
    # FAIR BENCHMARK: Interleave queries across databases
    # Instead of testing all queries on DB1, then DB2, etc.
    # We test query1 on all DBs, then query2 on all DBs, etc.
    # This eliminates cold start bias and LLM caching advantages
    
    for i in range(num_queries):
        query = randomized_queries[i % len(randomized_queries)]
        
        for db_name, vector_store in vector_stores.items():
            status_text.text(f"[{db_name}] Query {i+1}/{num_queries}...")
            
            metrics = measure_performance(vector_store, query, llm, top_k)
            all_results[db_name].append({
                'query_num': i + 1,
                'query': query,
                'database': db_name,
                **metrics
            })
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
    
    status_text.text("Speed test completed, running scalability test...")
    
    dfs = {db_name: pd.DataFrame(results) for db_name, results in all_results.items()}
    combined_df = pd.concat(dfs.values(), ignore_index=True)
    
    st.session_state['benchmark_results'] = combined_df
    st.session_state['dfs'] = dfs
    st.session_state['tested_config'] = current_config
    
    # ========== PHASE 2: SCALABILITY TEST ==========
    top_k_levels = [1, 2, 3, 5, 8, 10, 15, 20]
    scalability_results = {}
    
    for idx, (db_name, vector_store) in enumerate(vector_stores.items()):
        status_text.text(f"Scalability Test [{db_name}]...")
        scalability_results[db_name] = measure_scalability(vector_store, randomized_queries, top_k_levels)
        progress_bar.progress((total_tests + idx + 1) / (total_tests + len(vector_stores) * 2))
    
    st.session_state['scalability_results'] = scalability_results
    
    # ========== PHASE 3: RETRIEVAL QUALITY TEST ==========
    status_text.text("Running Retrieval Quality Test...")
    
    quality_results = {}
    
    for idx, (db_name, vector_store) in enumerate(vector_stores.items()):
        status_text.text(f"Quality Test [{db_name}]...")
        quality_results[db_name] = measure_retrieval_quality(vector_store, randomized_queries[:num_queries], top_k)
        progress_bar.progress((total_tests + len(vector_stores) + idx + 1) / (total_tests + len(vector_stores) * 2))
    
    st.session_state['quality_results'] = quality_results
    
    # ========== AUTO-SAVE JSON ==========
    status_text.text("Saving results...")
    
    import json
    
    avg_retrieval_times = {db_name: df['retrieval_time'].mean() for db_name, df in dfs.items()}
    winner = min(avg_retrieval_times, key=avg_retrieval_times.get)
    winner_time = avg_retrieval_times[winner]
    other_dbs = {k: v for k, v in avg_retrieval_times.items() if k != winner}
    speed_improvement = ((max(other_dbs.values()) - winner_time) / max(other_dbs.values()) * 100) if other_dbs else 0
    
    export_data = {
        'metadata': {
            'benchmark_date': datetime.now().isoformat(),
            'llm_model': CHAT_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'num_queries': num_queries,
            'top_k': top_k,
            'databases_tested': list(vector_stores.keys())
        },
        'speed_test': {
            'winner': {'database': winner, 'avg_retrieval_ms': round(winner_time, 2), 'speed_improvement_percent': round(speed_improvement, 1)},
            'summary': [
                {
                    'database': db_name,
                    'mean_total_ms': round(df['total_time'].mean(), 2),
                    'median_total_ms': round(df['total_time'].median(), 2),
                    'std_total_ms': round(df['total_time'].std(), 2),
                    'min_total_ms': round(df['total_time'].min(), 2),
                    'max_total_ms': round(df['total_time'].max(), 2),
                    'mean_retrieval_ms': round(df['retrieval_time'].mean(), 2),
                    'mean_llm_ms': round(df['llm_time'].mean(), 2),
                } for db_name, df in dfs.items()
            ],
            'raw_results': combined_df.to_dict(orient='records')
        },
        'scalability_test': scalability_results,
        'retrieval_quality': {
            db_name: {
                'avg_precision': round(np.mean([r['precision'] for r in results]), 4),
                'avg_recall': round(np.mean([r['recall'] for r in results]), 4),
                'avg_f1_score': round(np.mean([r['f1_score'] for r in results]), 4),
                'per_query': results
            } for db_name, results in quality_results.items()
        }
    }
    # Store export data in session for download button
    st.session_state['export_data'] = export_data
    
    status_text.text("Completed!")
    progress_bar.progress(1.0)


# Display Results
if 'benchmark_results' in st.session_state:
    combined_df = st.session_state['benchmark_results']
    dfs = st.session_state['dfs']
    
    st.markdown("---")
    
    # 🏆 Winner Banner - Based on RETRIEVAL TIME (database performance only)
    avg_retrieval_times = {db_name: df['retrieval_time'].mean() for db_name, df in dfs.items()}
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
    
    cols = st.columns(len(dfs))
    for idx, (db_name, df) in enumerate(dfs.items()):
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
    
    # Chart 1: Scalability with Rolling Average
    st.markdown('<div class="section-header">Scalability Analysis</div>', unsafe_allow_html=True)
    
    fig1 = go.Figure()
    
    for db_name, df in dfs.items():
        color = COLORS.get(db_name, '#94A3B8')
        
        # Filter out failed queries (where total_time == 0)
        df_valid = df[df['total_time'] > 0]
        
        # Calculate average for this database
        avg_time = df_valid['total_time'].mean()
        
        # Main response time line
        fig1.add_trace(go.Scatter(
            x=df_valid['query_num'],
            y=df_valid['total_time'],
            mode='lines+markers',
            name=f'{db_name}',
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate=f'<b>{db_name}</b><br>Query: %{{x}}<br>Time: %{{y:.1f}}ms<extra></extra>'
        ))
        
        # Average line (red dashed)
        fig1.add_trace(go.Scatter(
            x=[df_valid['query_num'].min(), df_valid['query_num'].max()],
            y=[avg_time, avg_time],
            mode='lines',
            name=f'Average ({avg_time:.0f}ms)',
            line=dict(color='#EF4444', width=2, dash='dash'),
            hovertemplate=f'<b>Average</b>: {avg_time:.1f}ms<extra></extra>'
        ))
    
    fig1.update_layout(
        title=dict(
            text='<b>Response Time Under Load</b>',
            font=dict(size=24, color='#F8FAFC')
        ),
        xaxis=dict(
            title='Query Number',
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
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Stacked Bar Comparison
    st.markdown('<div class="section-header">Speed Breakdown</div>', unsafe_allow_html=True)
    
    fig2 = go.Figure()
    
    db_names = list(dfs.keys())
    
    # Retrieval bars
    fig2.add_trace(go.Bar(
        name='Retrieval Time',
        x=db_names,
        y=[dfs[db]['retrieval_time'].mean() for db in db_names],
        marker=dict(
            color=COLORS['primary'],
            line=dict(color='rgba(255,255,255,0.2)', width=2)
        ),
        text=[f"{dfs[db]['retrieval_time'].mean():.1f}ms" for db in db_names],
        textposition='inside',
        textfont=dict(size=14, color='white', family='Inter'),
        hovertemplate='<b>%{x}</b><br>Retrieval: %{y:.1f}ms<extra></extra>'
    ))
    
    # LLM bars
    fig2.add_trace(go.Bar(
        name='LLM Generation',
        x=db_names,
        y=[dfs[db]['llm_time'].mean() for db in db_names],
        marker=dict(
            color=COLORS['muted'],
            line=dict(color='rgba(255,255,255,0.2)', width=2)
        ),
        text=[f"{dfs[db]['llm_time'].mean():.1f}ms" for db in db_names],
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
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics Table
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    
    stats_data = []
    for db_name, df in dfs.items():
        stats_data.append({
            'Database': db_name,
            'Mean (ms)': df['total_time'].mean(),
            'Median (ms)': df['total_time'].median(),
            'Std Dev': df['total_time'].std(),
            'Min (ms)': df['total_time'].min(),
            'Max (ms)': df['total_time'].max()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    st.dataframe(
        stats_df.style.format({
            'Mean (ms)': '{:.2f}',
            'Median (ms)': '{:.2f}',
            'Std Dev': '{:.2f}',
            'Min (ms)': '{:.2f}',
            'Max (ms)': '{:.2f}'
        }).background_gradient(subset=['Mean (ms)'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )
    
    # Query Scoreboard - Show all queries sorted by fastest
    st.markdown("---")
    
    st.markdown('<div class="section-header">Query Scoreboard</div>', unsafe_allow_html=True)
    
    # Create scoreboard dataframe - sorted by fastest (default)
    scoreboard_df = combined_df[['query_num', 'database', 'query', 'retrieval_time', 'llm_time', 'total_time']].copy()
    scoreboard_df = scoreboard_df.rename(columns={
        'query_num': '#',
        'database': 'Database',
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
        }).background_gradient(subset=['Total (ms)'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True,
        height=400
    )

if 'scalability_results' in st.session_state:
    scalability_results = st.session_state['scalability_results']
    
    st.markdown("---")
    st.markdown('<div class="section-header">Scalability Analysis</div>', unsafe_allow_html=True)
    
    fig_scale = go.Figure()
    
    for db_name, results in scalability_results.items():
        color = COLORS.get(db_name, '#94A3B8')
        x_vals = [r['top_k'] for r in results]
        y_vals = [r['avg_time'] for r in results]
        
        fig_scale.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers', name=db_name,
            line=dict(color=color, width=3), marker=dict(size=10, color=color),
            hovertemplate=f'<b>{db_name}</b><br>Top-K: %{{x}}<br>Avg Time: %{{y:.2f}}ms<extra></extra>'
        ))
    
    fig_scale.update_layout(
        title=dict(text='<b>Response Time vs Retrieval Volume (Top-K)</b>', font=dict(size=24, color='#F8FAFC')),
        xaxis=dict(title='Top-K (Documents Retrieved)', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        yaxis=dict(title='Average Response Time (ms)', gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#F8FAFC'), height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.3)', borderwidth=1, font=dict(size=13))
    )
    
    st.plotly_chart(fig_scale, use_container_width=True)
    
    # Scalability Statistics
    st.markdown('<div class="section-header">Scalability Statistics</div>', unsafe_allow_html=True)
    
    scale_stats = []
    for db_name, results in scalability_results.items():
        if results:
            avg_times = [r['avg_time'] for r in results]
            scale_stats.append({
                'Database': db_name,
                'Min Avg (ms)': min(avg_times),
                'Max Avg (ms)': max(avg_times),
                'Growth Rate': f"{((max(avg_times) - min(avg_times)) / min(avg_times) * 100):.1f}%"
            })
    
    scale_df = pd.DataFrame(scale_stats)
    st.dataframe(scale_df.style.format({'Min Avg (ms)': '{:.2f}', 'Max Avg (ms)': '{:.2f}'}), use_container_width=True, hide_index=True)

# ========== RETRIEVAL QUALITY RESULTS ==========
if 'quality_results' in st.session_state:
    quality_results = st.session_state['quality_results']
    
    st.markdown("---")
    st.markdown('<div class="section-header">Retrieval Quality Metrics</div>', unsafe_allow_html=True)
    
    # Summary Cards
    cols = st.columns(len(quality_results))
    for idx, (db_name, results) in enumerate(quality_results.items()):
        if results:
            avg_precision = np.mean([r['precision'] for r in results])
            avg_recall = np.mean([r['recall'] for r in results])
            avg_f1 = np.mean([r['f1_score'] for r in results])
            
            badge_class = f'badge-{db_name.lower().replace(" ", "").replace("+", "")}'
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="db-badge {badge_class}">{db_name}</span>
                    <div class="metric-value">{avg_f1:.2%}</div>
                    <div class="metric-label">F1-Score</div>
                    <div style="margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div>
                            <div style="color: #22C55E; font-weight: 600; font-size: 1.25rem;">{avg_precision:.2%}</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Precision</div>
                        </div>
                        <div>
                            <div style="color: #3B82F6; font-weight: 600; font-size: 1.25rem;">{avg_recall:.2%}</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Recall</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Quality Bar Chart
    fig_quality = go.Figure()
    
    db_names = list(quality_results.keys())
    precisions = [np.mean([r['precision'] for r in quality_results[db]]) for db in db_names]
    recalls = [np.mean([r['recall'] for r in quality_results[db]]) for db in db_names]
    f1_scores = [np.mean([r['f1_score'] for r in quality_results[db]]) for db in db_names]
    
    fig_quality.add_trace(go.Bar(name='Precision', x=db_names, y=precisions, marker=dict(color='#22C55E'), text=[f"{p:.1%}" for p in precisions], textposition='outside', textfont=dict(size=12, color='#22C55E')))
    fig_quality.add_trace(go.Bar(name='Recall', x=db_names, y=recalls, marker=dict(color='#3B82F6'), text=[f"{r:.1%}" for r in recalls], textposition='outside', textfont=dict(size=12, color='#3B82F6')))
    fig_quality.add_trace(go.Bar(name='F1-Score', x=db_names, y=f1_scores, marker=dict(color='#F59E0B'), text=[f"{f:.1%}" for f in f1_scores], textposition='outside', textfont=dict(size=12, color='#F59E0B')))
    
    fig_quality.update_layout(
        title=dict(text='<b>Precision, Recall & F1-Score Comparison</b>', font=dict(size=24, color='#F8FAFC')),
        barmode='group', xaxis=dict(title='', tickfont=dict(size=13, color='#F8FAFC')),
        yaxis=dict(title='Score', range=[0, 1.1], gridcolor='rgba(148, 163, 184, 0.1)', title_font=dict(size=14, color='#94A3B8'), tickfont=dict(size=12, color='#94A3B8')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#F8FAFC'), height=450, bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor='rgba(30, 41, 59, 0.8)', bordercolor='rgba(148, 163, 184, 0.3)', borderwidth=1, font=dict(size=13))
    )
    
    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Quality Statistics
    st.markdown('<div class="section-header">Quality Statistics</div>', unsafe_allow_html=True)
    
    quality_stats = []
    for db_name, results in quality_results.items():
        if results:
            quality_stats.append({
                'Database': db_name,
                'Avg Precision': np.mean([r['precision'] for r in results]),
                'Avg Recall': np.mean([r['recall'] for r in results]),
                'Avg F1-Score': np.mean([r['f1_score'] for r in results]),
                'Queries Tested': len(results)
            })
    
    quality_df = pd.DataFrame(quality_stats)
    st.dataframe(
        quality_df.style.format({'Avg Precision': '{:.2%}', 'Avg Recall': '{:.2%}', 'Avg F1-Score': '{:.2%}'}),
        use_container_width=True, hide_index=True
    )

# Download JSON Button
if 'export_data' in st.session_state:
    st.markdown("---")
    import json
    json_data = json.dumps(st.session_state['export_data'], indent=2, default=str)
    st.download_button(
        label="Download Results (JSON)",
        data=json_data,
        file_name=f"benchmark_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
