import streamlit as st
import os
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="RAG Performance Analysis", layout="wide")
st.title("📊 RAG Chatbot: Scalability & Response Speed Analysis")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Test configuration
st.sidebar.header("Test Configuration")
test_queries = st.sidebar.number_input("Number of Test Queries", min_value=5, max_value=100, value=30)

# Sample queries
SAMPLE_QUERIES = [
        # RELEVANT queries (from PDF content)
    "Jelaskan Alur Pengajuan Cuti Akademik?",
    "Bagaimana cara pembayaran UKT melalui Bank Mandiri?",
    "Apa yang terjadi jika mengajukan cuti di minggu ke 0?",
    "Bagaimana prosedur drop mata kuliah?",
    "Siapa saja yang harus menyetujui pengajuan cuti?",
    "Berapa biaya UKT jika cuti di minggu ke 5?",
    "Bagaimana cara bayar UKT lewat ATM BNI?",
    "Apa itu Program Magang Reguler?",
    "Berapa lama maksimal waktu pelaksanaan magang?",
    "Bagaimana pembayaran UKT melalui BSM Mobile Banking?",
    
    # IRRELEVANT queries (not in PDF - should get "tidak tersedia" response)
    "Siapa rektor ITS saat ini?",
    "Bagaimana cara mengajukan beasiswa?",
    "Kapan jadwal UTS semester ini?",
    "Berapa passing grade jurusan Teknik Informatika?",
    "Bagaimana cara daftar organisasi mahasiswa?",
    "Apa syarat kelulusan mahasiswa?",
    "Dimana lokasi perpustakaan pusat?",
    "Bagaimana cara mengubah data pribadi di sistem?",
    "Siapa ketua jurusan Teknik Komputer?",
    "Kapan wisuda tahun ini?"
]

def measure_performance(query):
    """Measure complete query performance"""
    # Retrieval time
    start = time.time()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.75}
    )
    docs = retriever.invoke(query)
    retrieval_time = time.time() - start
    
    # LLM time (if docs found)
    if docs and len(docs) > 0:
        docs_text = "\n\n".join(d.page_content for d in docs)
        system_prompt = f"Konteks: {docs_text}\n\nJawabkan dengan detail."
        messages = [SystemMessage(system_prompt), HumanMessage(query)]
        
        llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
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
        'num_docs': len(docs)
    }

# Run test
if st.sidebar.button("🚀 Run Performance Test", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i in range(test_queries):
        status_text.text(f"Testing query {i+1}/{test_queries}...")
        
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        metrics = measure_performance(query)
        
        results.append({
            'query_num': i + 1,
            'retrieval_time': metrics['retrieval_time'],
            'llm_time': metrics['llm_time'],
            'total_time': metrics['total_time'],
            'num_docs': metrics['num_docs']
        })
        
        progress_bar.progress((i + 1) / test_queries)
    
    status_text.text("✅ Test completed!")
    df = pd.DataFrame(results)
    st.session_state['results'] = df

# Display graphs
if 'results' in st.session_state:
    df = st.session_state['results']
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Response Time", f"{df['total_time'].mean():.2f} ms")
    with col2:
        st.metric("Avg Retrieval Time", f"{df['retrieval_time'].mean():.2f} ms")
    with col3:
        st.metric("Avg LLM Time", f"{df['llm_time'].mean():.2f} ms")
    
    st.markdown("---")
    
    # SCALABILITY GRAPH
    st.subheader("📈 Scalability Analysis")
    st.caption("Shows how response time behaves as the number of queries increases")
    
    fig_scalability = go.Figure()
    
    # Add total response time line
    fig_scalability.add_trace(go.Scatter(
        x=df['query_num'],
        y=df['total_time'],
        mode='lines+markers',
        name='Response Time',
        line=dict(color='#3498db', width=3),
        marker=dict(size=6)
    ))
    
    # Add average line
    avg_time = df['total_time'].mean()
    fig_scalability.add_trace(go.Scatter(
        x=df['query_num'],
        y=[avg_time] * len(df),
        mode='lines',
        name=f'Average ({avg_time:.2f} ms)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_scalability.update_layout(
        title='Response Time Scalability (Query Load Test)',
        xaxis_title='Query Number (Load)',
        yaxis_title='Response Time (ms)',
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_scalability, use_container_width=True)
    
    st.markdown("---")
    
    # RESPONSE SPEED GRAPH
    st.subheader("⚡ Response Speed Breakdown")
    st.caption("Shows the time distribution between retrieval and LLM generation")
    
    fig_speed = go.Figure()
    
    fig_speed.add_trace(go.Bar(
        name='Retrieval Time',
        x=df['query_num'],
        y=df['retrieval_time'],
        marker_color='#2ecc71'
    ))
    
    fig_speed.add_trace(go.Bar(
        name='LLM Generation Time',
        x=df['query_num'],
        y=df['llm_time'],
        marker_color='#e74c3c'
    ))
    
    fig_speed.update_layout(
        barmode='stack',
        title='Response Speed Component Breakdown',
        xaxis_title='Query Number',
        yaxis_title='Time (ms)',
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_speed, use_container_width=True)
    
    # Statistics table
    st.markdown("---")
    st.subheader("📊 Performance Statistics")
    
    stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Total Response Time (ms)': [
            df['total_time'].mean(),
            df['total_time'].median(),
            df['total_time'].std(),
            df['total_time'].min(),
            df['total_time'].max()
        ],
        'Retrieval Time (ms)': [
            df['retrieval_time'].mean(),
            df['retrieval_time'].median(),
            df['retrieval_time'].std(),
            df['retrieval_time'].min(),
            df['retrieval_time'].max()
        ],
        'LLM Time (ms)': [
            df['llm_time'].mean(),
            df['llm_time'].median(),
            df['llm_time'].std(),
            df['llm_time'].min(),
            df['llm_time'].max()
        ]
    })
    
    st.dataframe(stats.style.format({
        'Total Response Time (ms)': '{:.2f}',
        'Retrieval Time (ms)': '{:.2f}',
        'LLM Time (ms)': '{:.2f}'
    }), use_container_width=True)
    
    # Download results
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )