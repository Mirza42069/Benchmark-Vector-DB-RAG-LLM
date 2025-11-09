import streamlit as st
import os
from dotenv import load_dotenv
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="RAG Performance Analysis", layout="wide")
st.title("📊 RAG Chatbot: Scalability & Response Speed Analysis (ChromaDB)")

# Initialize ChromaDB
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "its_guidebook"

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Verify database
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()
    st.sidebar.success(f"✅ Connected to ChromaDB")
    st.sidebar.info(f"📚 Documents: {doc_count}")
    st.sidebar.info(f"📁 Storage: {CHROMA_PATH}")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.error("Please run `ingestion.py` first!")
    st.stop()

# Test configuration
st.sidebar.header("Test Configuration")
test_queries = st.sidebar.number_input("Number of Test Queries", min_value=5, max_value=100, value=30)

# Sample queries for ITS International Students Guidebook RAG System
SAMPLE_QUERIES = [
    # RELEVANT queries (from PDF content - should return accurate answers)
    "What documents do I need to bring when arriving in Surabaya?",
    "How many types of academic VISA are available for international students?",
    "What is the ITS administration fee for international students?",
    "Where is the ITS Global Engagement Office located?",
    "How do I register my phone's IMEI for a long stay in Indonesia?",
    "What are the internet package options for Tourist SIM Card?",
    "How much does accommodation cost per month near ITS?",
    "What is the emergency hotline number in Surabaya?",
    "What is the dress code for attending classes at ITS?",
    "Which banks are available inside ITS campus?",
    "What are the nearest hospitals to ITS?",
    "What activities are included in the O-Week?",
    "How do I access the MyITS system?",
    "What is the fine for not having proof of residency within 14 days?",
    "Which transportation apps can I use in Surabaya?",
    "How many student activity units (UKM) are available at ITS?",
    "What is Rawon Setan and where can I find it?",
    "What faculties are available at ITS?",
    "How long does the admission evaluation process take?",
    "What are the steps I need to complete upon arrival at ITS?",
    
    # IRRELEVANT queries (not in PDF - should get "information not available" response)
    "What is the average GPA requirement for graduate programs?",
    "How much is the tuition fee per semester for undergraduate programs?",
    "What scholarship amounts are available for international students?",
    "Can you explain the Mechanical Engineering course curriculum?",
    "What is the acceptance rate for international students at ITS?",
    "How do I apply for internships at companies in Surabaya?",
    "What are the job prospects after graduating from ITS?",
    "How does ITS rank compared to other Indonesian universities?",
    "How do I get a work permit after graduation?",
    "What are the best universities in Jakarta?",
    "What is the weather like in Bali during summer?",
    "How do I convert Indonesian credits to ECTS credits?",
    "When is the graduation ceremony this year?",
    "Who is the current rector of ITS?",
    "What are the requirements for changing majors?",
    "How much does it cost to travel from Surabaya to Thailand?",
    "What research opportunities are available for undergraduates?",
    "Can you recommend good restaurants in Singapore?",
    "What are the requirements for obtaining Indonesian citizenship?",
    "How do I prepare for IELTS or TOEFL exams?"
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
        system_prompt = f"Context: {docs_text}\n\nAnswer in detail based only on the context."
        messages = [SystemMessage(system_prompt), HumanMessage(query)]
        
        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
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