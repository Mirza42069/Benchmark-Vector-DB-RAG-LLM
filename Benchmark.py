"""
Comprehensive Benchmark System
Comparing Pinecone, PostgreSQL+pgvector, and ChromaDB
For thesis: The Effect of Vector Database Selection on Scalability and Response Speed
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.document_processor import SAMPLE_QUERIES

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Benchmark System",
    page_icon="📊",
    layout="wide"
)

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Title
st.title("📊 RAG System Benchmark Comparison")
st.markdown("### The Effect of Vector Database Selection on Scalability and Response Speed")
st.caption("Comparing Pinecone vs PostgreSQL+pgvector vs ChromaDB")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Benchmark Configuration")
    
    # Database selection
    st.subheader("📂 Select Databases to Test")
    test_pinecone = st.checkbox("Pinecone", value=True)
    test_postgresql = st.checkbox("PostgreSQL + pgvector", value=True)
    test_chroma = st.checkbox("ChromaDB", value=True)
    
    if not any([test_pinecone, test_postgresql, test_chroma]):
        st.error("⚠️ Select at least one database!")
    
    st.divider()
    
    # Test parameters
    st.subheader("🔧 Test Parameters")
    num_queries = st.slider(
        "Number of Test Queries",
        min_value=10,
        max_value=100,
        value=60,
        step=10,
        help="More queries = better statistical significance"
    )
    
    score_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=SCORE_THRESHOLD,
        step=0.05
    )
    
    top_k = st.slider(
        "Documents per Query",
        min_value=1,
        max_value=10,
        value=TOP_K
    )
    
    st.divider()
    
    # Run button
    run_benchmark = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)
    
    st.divider()
    
    st.markdown("""
    ### 📋 Metrics Measured
    
    **1. Response Speed**
    - Retrieval Time (vector search)
    - LLM Generation Time
    - Total Response Time
    
    **2. Scalability**
    - Performance under load
    - Consistency across queries
    
    **3. Statistical Analysis**
    - Mean, Median, Std Deviation
    - Min/Max response times
    """)

# Initialize vector stores
@st.cache_resource
def init_all_vector_stores():
    """Initialize all vector stores"""
    stores = {}
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Pinecone
    if test_pinecone:
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "its-helpdesk-chatbot"))
            stores['Pinecone'] = PineconeVectorStore(index=index, embedding=embeddings)
            st.sidebar.success("✅ Pinecone loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Pinecone: {str(e)}")
    
    # PostgreSQL
    if test_postgresql:
        try:
            connection_string = f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            stores['PostgreSQL'] = PGVector(
                embeddings=embeddings,
                collection_name=os.getenv("COLLECTION_NAME", "its_guidebook"),
                connection=connection_string,
                use_jsonb=True,
            )
            st.sidebar.success("✅ PostgreSQL loaded")
        except Exception as e:
            st.sidebar.error(f"❌ PostgreSQL: {str(e)}")
    
    # ChromaDB
    if test_chroma:
        try:
            client = chromadb.PersistentClient(path="chroma_db")
            stores['ChromaDB'] = Chroma(
                client=client,
                collection_name="its_guidebook",
                embedding_function=embeddings,
            )
            st.sidebar.success("✅ ChromaDB loaded")
        except Exception as e:
            st.sidebar.error(f"❌ ChromaDB: {str(e)}")
    
    return stores

def measure_performance(vector_store, query, llm, score_threshold, top_k):
    """Measure complete query performance"""
    try:
        # Retrieval time
        start = time.time()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold}
        )
        docs = retriever.invoke(query)
        retrieval_time = time.time() - start
        
        # LLM time (if docs found)
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
            'retrieval_time': retrieval_time * 1000,  # ms
            'llm_time': llm_time * 1000,  # ms
            'total_time': total_time * 1000,  # ms
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

# Main benchmark execution
if run_benchmark:
    # Initialize stores
    with st.spinner("🔄 Initializing vector stores..."):
        vector_stores = init_all_vector_stores()
    
    if not vector_stores:
        st.error("❌ No vector stores initialized! Check your configuration.")
        st.stop()
    
    # Initialize LLM
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    
    st.success(f"✅ Initialized {len(vector_stores)} database(s)")
    st.info(f"📊 Running {num_queries} queries on each database...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Store results
    all_results = {db_name: [] for db_name in vector_stores.keys()}
    
    # Run benchmark
    total_tests = len(vector_stores) * num_queries
    current_test = 0
    
    for db_name, vector_store in vector_stores.items():
        st.markdown(f"### 🔄 Testing {db_name}...")
        
        for i in range(num_queries):
            query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            
            status_text.text(f"[{db_name}] Query {i+1}/{num_queries}: {query[:50]}...")
            
            metrics = measure_performance(vector_store, query, llm, score_threshold, top_k)
            
            all_results[db_name].append({
                'query_num': i + 1,
                'query': query,
                'database': db_name,
                **metrics
            })
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
    
    status_text.text("✅ Benchmark completed!")
    
    # Convert to DataFrames
    dfs = {db_name: pd.DataFrame(results) for db_name, results in all_results.items()}
    combined_df = pd.concat(dfs.values(), ignore_index=True)
    
    # Store in session state
    st.session_state['benchmark_results'] = combined_df
    st.session_state['dfs'] = dfs

# Display results
if 'benchmark_results' in st.session_state:
    combined_df = st.session_state['benchmark_results']
    dfs = st.session_state['dfs']
    
    st.markdown("---")
    st.header("📈 Benchmark Results")
    
    # Summary metrics
    st.subheader("📊 Performance Summary")
    
    cols = st.columns(len(dfs))
    for idx, (db_name, df) in enumerate(dfs.items()):
        with cols[idx]:
            st.markdown(f"### {db_name}")
            st.metric("Avg Response Time", f"{df['total_time'].mean():.2f} ms")
            st.metric("Avg Retrieval Time", f"{df['retrieval_time'].mean():.2f} ms")
            st.metric("Avg LLM Time", f"{df['llm_time'].mean():.2f} ms")
            st.metric("Avg Docs Retrieved", f"{df['num_docs'].mean():.1f}")
    
    st.markdown("---")
    
    # === SCALABILITY ANALYSIS ===
    st.subheader("📈 Scalability Analysis")
    st.caption("How response time behaves as the number of queries increases")
    
    fig_scalability = go.Figure()
    
    colors = {'Pinecone': '#3498db', 'PostgreSQL': '#2ecc71', 'ChromaDB': '#e74c3c'}
    
    for db_name, df in dfs.items():
        # Add response time line
        fig_scalability.add_trace(go.Scatter(
            x=df['query_num'],
            y=df['total_time'],
            mode='lines+markers',
            name=f'{db_name}',
            line=dict(color=colors.get(db_name, '#95a5a6'), width=2),
            marker=dict(size=4)
        ))
        
        # Add average line
        avg_time = df['total_time'].mean()
        fig_scalability.add_trace(go.Scatter(
            x=df['query_num'],
            y=[avg_time] * len(df),
            mode='lines',
            name=f'{db_name} Avg ({avg_time:.2f} ms)',
            line=dict(color=colors.get(db_name, '#95a5a6'), width=2, dash='dash'),
            showlegend=True
        ))
    
    fig_scalability.update_layout(
        title='Response Time Scalability (Query Load Test)',
        xaxis_title='Query Number (Load)',
        yaxis_title='Response Time (ms)',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_scalability, use_container_width=True)
    
    st.markdown("---")
    
    # === RESPONSE SPEED BREAKDOWN ===
    st.subheader("⚡ Response Speed Breakdown")
    st.caption("Time distribution between retrieval and LLM generation")
    
    # Create subplots for each database
    fig_speed = go.Figure()
    
    for db_name, df in dfs.items():
        fig_speed.add_trace(go.Bar(
            name=f'{db_name} - Retrieval',
            x=df['query_num'],
            y=df['retrieval_time'],
            marker_color=colors.get(db_name, '#95a5a6'),
            legendgroup=db_name,
        ))
        
        fig_speed.add_trace(go.Bar(
            name=f'{db_name} - LLM',
            x=df['query_num'],
            y=df['llm_time'],
            marker_color=colors.get(db_name, '#95a5a6'),
            marker_pattern_shape="/",
            legendgroup=db_name,
        ))
    
    fig_speed.update_layout(
        barmode='group',
        title='Response Speed Component Breakdown by Database',
        xaxis_title='Query Number',
        yaxis_title='Time (ms)',
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_speed, use_container_width=True)
    
    st.markdown("---")
    
    
    # === STATISTICS TABLE ===
    st.subheader("📊 Detailed Performance Statistics")
    
    stats_data = []
    for db_name, df in dfs.items():
        stats_data.append({
            'Database': db_name,
            'Mean (ms)': df['total_time'].mean(),
            'Median (ms)': df['total_time'].median(),
            'Std Dev (ms)': df['total_time'].std(),
            'Min (ms)': df['total_time'].min(),
            'Max (ms)': df['total_time'].max(),
            'Avg Docs Retrieved': df['num_docs'].mean()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    st.dataframe(
        stats_df.style.format({
            'Mean (ms)': '{:.2f}',
            'Median (ms)': '{:.2f}',
            'Std Dev (ms)': '{:.2f}',
            'Min (ms)': '{:.2f}',
            'Max (ms)': '{:.2f}',
            'Avg Docs Retrieved': '{:.2f}'
        }).background_gradient(subset=['Mean (ms)'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # === RETRIEVAL VS LLM TIME COMPARISON ===
    st.subheader("⚖️ Retrieval vs LLM Time Comparison")
    
    fig_comparison = go.Figure()
    
    x_labels = list(dfs.keys())
    retrieval_times = [df['retrieval_time'].mean() for df in dfs.values()]
    llm_times = [df['llm_time'].mean() for df in dfs.values()]
    
    fig_comparison.add_trace(go.Bar(
        name='Retrieval Time',
        x=x_labels,
        y=retrieval_times,
        marker_color='#2ecc71',
        text=[f'{t:.2f} ms' for t in retrieval_times],
        textposition='auto',
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='LLM Time',
        x=x_labels,
        y=llm_times,
        marker_color='#e74c3c',
        text=[f'{t:.2f} ms' for t in llm_times],
        textposition='auto',
    ))
    
    fig_comparison.update_layout(
        barmode='group',
        title='Average Time Components by Database',
        xaxis_title='Database',
        yaxis_title='Time (ms)',
        height=400
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # === DOWNLOAD RESULTS ===
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Combined results
        csv_combined = combined_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Results (CSV)",
            data=csv_combined,
            file_name=f"benchmark_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Statistics summary
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics Summary (CSV)",
            data=csv_stats,
            file_name=f"benchmark_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.divider()
st.caption("🎓 ITS RAG Benchmark System | Powered by Streamlit & Ollama")