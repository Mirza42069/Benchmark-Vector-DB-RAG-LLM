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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================
# MODERN COLOR PALETTE & STYLING
# ============================================
COLORS = {
    'Pinecone': '#6366F1',      # Indigo
    'PostgreSQL': '#10B981',    # Emerald
    'ChromaDB': '#F59E0B',      # Amber
    'retrieval': '#3B82F6',     # Blue
    'llm': '#EC4899',           # Pink
    'background': '#0F172A',
    'card': '#1E293B',
    'text': '#F8FAFC',
    'muted': '#94A3B8',
}

# Page configuration
st.set_page_config(
    page_title="RAG Benchmark System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .metric-label {
        color: #94A3B8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .db-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 4px;
    }
    
    .badge-pinecone { background: rgba(99, 102, 241, 0.2); color: #6366F1; border: 2px solid #6366F1; }
    .badge-postgresql { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 2px solid #10B981; }
    .badge-chromadb { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 2px solid #F59E0B; }
    
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 40px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid #6366F1;
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
<div style="text-align: center; padding: 40px 0 20px 0;">
    <h1 style="font-size: 3rem; font-weight: 700; background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;">
        📊 Vector Database Benchmark
    </h1>
    <p style="color: #94A3B8; font-size: 1.125rem; font-weight: 400;">
        Benchmark Vector Database on Scalability and Response Speed
    </p>
    <div style="display: flex; justify-content: center; gap: 12px; margin-top: 16px;">
        <span class="db-badge badge-pinecone">Pinecone</span>
        <span class="db-badge badge-postgresql">PostgreSQL + pgvector</span>
        <span class="db-badge badge-chromadb">ChromaDB</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 📂 Databases")
    test_pinecone = st.checkbox("Pinecone", value=True)
    test_postgresql = st.checkbox("PostgreSQL + pgvector", value=True)
    test_chroma = st.checkbox("ChromaDB", value=True)
    
    if not any([test_pinecone, test_postgresql, test_chroma]):
        st.error("⚠️ Select at least one database!")
    
    st.divider()
    
    st.markdown("### 🔧 Parameters")
    num_queries = st.slider("Test Queries", 10, 100, 60, 10)
    score_threshold = st.slider("Similarity Threshold", 0.0, 1.0, SCORE_THRESHOLD, 0.05)
    top_k = st.slider("Documents per Query", 1, 10, TOP_K)
    
    st.divider()
    
    run_benchmark = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

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
            st.sidebar.success("✅ Pinecone")
        except Exception as e:
            st.sidebar.error(f"❌ Pinecone: {str(e)[:30]}...")
    
    if test_postgresql:
        db_password = os.getenv('DB_PASSWORD')
        if not db_password:
            st.sidebar.error("❌ PostgreSQL: DB_PASSWORD not set")
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
                st.sidebar.success("✅ PostgreSQL")
            except Exception as e:
                logger.error(f"PostgreSQL connection error: {str(e)}")
                st.sidebar.error("❌ PostgreSQL: Connection failed")
    
    if test_chroma:
        try:
            client = chromadb.PersistentClient(path="chroma_db")
            stores['ChromaDB'] = Chroma(
                client=client,
                collection_name="its_guidebook",
                embedding_function=embeddings,
            )
            st.sidebar.success("✅ ChromaDB")
        except Exception as e:
            st.sidebar.error(f"❌ ChromaDB: {str(e)[:30]}...")
    
    return stores

def measure_performance(vector_store, query, llm, score_threshold, top_k):
    try:
        start = time.time()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold}
        )
        docs = retriever.invoke(query)
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



# Main benchmark
if run_benchmark:
    with st.spinner("🔄 Initializing..."):
        vector_stores = init_all_vector_stores(test_pinecone, test_postgresql, test_chroma)
    
    if not vector_stores:
        st.error("❌ No vector stores initialized!")
        st.stop()
    
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    st.success(f"✅ Initialized {len(vector_stores)} database(s)")
    
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
            
            metrics = measure_performance(vector_store, query, llm, score_threshold, top_k)
            all_results[db_name].append({
                'query_num': i + 1,
                'query': query,
                'database': db_name,
                **metrics
            })
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
    
    status_text.text("✅ Completed!")
    
    dfs = {db_name: pd.DataFrame(results) for db_name, results in all_results.items()}
    combined_df = pd.concat(dfs.values(), ignore_index=True)
    
    st.session_state['benchmark_results'] = combined_df
    st.session_state['dfs'] = dfs
    st.session_state['tested_config'] = current_config  # Save tested configuration

# Display Results
if 'benchmark_results' in st.session_state:
    combined_df = st.session_state['benchmark_results']
    dfs = st.session_state['dfs']
    
    st.markdown("---")
    
    # Summary Cards
    st.markdown('<div class="section-header">📊 Performance Summary</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header">📈 Scalability Analysis</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="section-header">⚡ Speed Breakdown</div>', unsafe_allow_html=True)
    
    fig2 = go.Figure()
    
    db_names = list(dfs.keys())
    
    # Retrieval bars
    fig2.add_trace(go.Bar(
        name='Retrieval Time',
        x=db_names,
        y=[dfs[db]['retrieval_time'].mean() for db in db_names],
        marker=dict(
            color=COLORS['retrieval'],
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
            color=COLORS['llm'],
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
    st.markdown('<div class="section-header">📋 Statistical Summary</div>', unsafe_allow_html=True)
    
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
    
    # Export
    st.markdown("---")
    st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_combined = combined_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Results (CSV)",
            data=csv_combined,
            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Statistics (CSV)",
            data=csv_stats,
            file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #94A3B8;">
    <p>📊 <strong>Vector Database Benchmark</strong></p>
    <p style="font-size: 0.875rem;">Powered by Streamlit, Plotly & Ollama</p>
</div>
""", unsafe_allow_html=True)
