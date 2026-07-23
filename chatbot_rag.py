"""
All-in-One RAG System - Compare Vector Databases
Pinecone vs PostgreSQL+pgvector vs ChromaDB
"""

import streamlit as st
import os
import sys
import logging
from dotenv import load_dotenv
import time

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.security import (
    sanitize_query,
    escape_html,
    build_pg_connection_string,
    sanitize_error_message
)

# Vector databases
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVector
import chromadb
from langchain_chroma import Chroma

# LangChain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG System - All Databases",
    page_icon="🤖",
    layout="wide"
)

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Database configs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-vector-benchmark")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD")  # No default - must be set in .env
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")

CHROMA_PATH = "chroma_db"

# Maximum query length for input validation
MAX_QUERY_LENGTH = 1000

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}

# Header
st.title("RAG System - All Databases")
st.caption("Compare responses from the configured RAG databases simultaneously")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📂 Enable Databases")
    use_pinecone = st.checkbox("Pinecone", value=True)
    use_postgresql = st.checkbox("PostgreSQL + pgvector", value=True)
    use_chroma = st.checkbox("ChromaDB", value=True)
    
    if not any([use_pinecone, use_postgresql, use_chroma]):
        st.error("⚠️ Select at least one database!")
    
    st.divider()
    
    st.subheader("🔧 Settings")
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
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### 💡 Sample Questions
    
    **Indonesian:**
    - Bagaimana cara mengubah password myITS Portal?
    - Apa itu Multi-Factor Authentication?
    - Berapa biaya akomodasi per bulan?
    
    **English:**
    - What documents do I need when arriving?
    - How do I register my phone's IMEI?
    - Which banks are available at ITS?
    """)

# Initialize all vector stores
@st.cache_resource
def init_all_vector_stores():
    """Initialize all vector stores"""
    stores = {}
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    return stores, embeddings

def get_pinecone_store(embeddings):
    """Initialize Pinecone"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        return PineconeVectorStore(index=index, embedding=embeddings)
    except Exception as e:
        logger.error(f"Pinecone connection error: {str(e)}")
        st.sidebar.error("❌ Pinecone: Connection failed")
        return None

def get_postgresql_store(embeddings):
    """Initialize PostgreSQL"""
    if not DB_PASSWORD:
        st.sidebar.error("❌ PostgreSQL: DB_PASSWORD not set")
        return None
    try:
        connection_string = build_pg_connection_string(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        return PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=connection_string,
            use_jsonb=True,
        )
    except Exception as e:
        logger.error(f"PostgreSQL connection error: {str(e)}")
        st.sidebar.error("❌ PostgreSQL: Connection failed")
        return None

def get_chroma_store(embeddings):
    """Initialize ChromaDB"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    except Exception as e:
        logger.error(f"ChromaDB connection error: {str(e)}")
        st.sidebar.error("❌ ChromaDB: Connection failed")
        return None

def query_database(db_name, vector_store, query, score_threshold, top_k, llm):
    """Query a single database and return results"""
    # Sanitize the query input
    query = sanitize_query(query, MAX_QUERY_LENGTH)
    """Query a single database and return results"""
    result = {
        "db_name": db_name,
        "success": False,
        "retrieval_time": 0,
        "llm_time": 0,
        "total_time": 0,
        "docs_found": 0,
        "response": "",
        "sources": [],
        "error": None
    }
    
    try:
        # Retrieval
        start_retrieval = time.time()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold}
        )
        docs = retriever.invoke(query)
        result["retrieval_time"] = (time.time() - start_retrieval) * 1000
        result["docs_found"] = len(docs)
        
        if docs and len(docs) > 0:
            # Generate response
            docs_text = "\n\n".join(d.page_content for d in docs)
            
            system_prompt = f"""You are a helpful assistant for Institut Teknologi Sepuluh Nopember (ITS) students.
Answer questions based ONLY on the provided context. If the answer is not in the context, say so politely.
Respond in the same language as the question (Indonesian or English).
Keep your response concise and clear.

Context:
{docs_text}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            start_llm = time.time()
            response = llm.invoke(messages)
            result["llm_time"] = (time.time() - start_llm) * 1000
            result["response"] = response.content
            result["success"] = True
            
            # Prepare sources
            for doc in docs:
                result["sources"].append({
                    "file": doc.metadata.get('source_file', 'Unknown'),
                    "lang": doc.metadata.get('chunk_language', 'unknown'),
                    "content": doc.page_content[:150] + "..."
                })
        else:
            result["response"] = "No relevant information found in the knowledge base."
            result["success"] = True
        
        result["total_time"] = result["retrieval_time"] + result["llm_time"]
        
    except Exception as e:
        result["error"] = str(e)
        result["response"] = f"Error: {str(e)}"
    
    return result

# Initialize embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Load selected vector stores
with st.spinner("🔄 Loading databases..."):
    active_stores = {}
    
    if use_pinecone:
        store = get_pinecone_store(embeddings)
        if store:
            active_stores["🌲 Pinecone"] = store
            st.sidebar.success("✅ Pinecone")
    
    if use_postgresql:
        store = get_postgresql_store(embeddings)
        if store:
            active_stores["🐘 PostgreSQL"] = store
            st.sidebar.success("✅ PostgreSQL")
    
    if use_chroma:
        store = get_chroma_store(embeddings)
        if store:
            active_stores["🎨 ChromaDB"] = store
            st.sidebar.success("✅ ChromaDB")

if not active_stores:
    st.error("❌ No databases available! Please check your configuration.")
    st.stop()

# Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        st.markdown("---")
        st.markdown(f"**Question:** {msg.get('query', '')}")
        
        cols = st.columns(len(msg["responses"]))
        for idx, (db_name, resp) in enumerate(msg["responses"].items()):
            with cols[idx]:
                st.markdown(f"### {db_name}")
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⏱️ Retrieval", f"{resp['retrieval_time']:.0f} ms")
                with col2:
                    st.metric("🤖 LLM", f"{resp['llm_time']:.0f} ms")
                
                st.metric("📄 Docs", resp['docs_found'])
                
                # Response
                if resp["success"]:
                    st.success(resp["response"])
                else:
                    st.error(resp["response"])
                
                # Sources
                if resp["sources"]:
                    with st.expander("📚 Sources"):
                        for src in resp["sources"]:
                            st.caption(f"**{src['file']}** ({src['lang']})")
                            st.text(src['content'])

# Chat input
if prompt := st.chat_input("Ask me anything about ITS..."):
    # Sanitize and validate user input
    prompt = sanitize_query(prompt, MAX_QUERY_LENGTH)
    if not prompt:
        st.warning("Please enter a valid question.")
        st.stop()
    
    # Display user message (escaped for safety)
    with st.chat_message("user"):
        st.markdown(escape_html(prompt))
    
    st.markdown("---")
    
    # Initialize LLM
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
    
    # Create columns for each database
    cols = st.columns(len(active_stores))
    results = {}
    
    # Query all databases
    for idx, (db_name, vector_store) in enumerate(active_stores.items()):
        with cols[idx]:
            st.markdown(f"### {db_name}")
            
            with st.spinner(f"🔍 Searching..."):
                result = query_database(
                    db_name, vector_store, prompt,
                    score_threshold, top_k, llm
                )
                results[db_name] = result
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⏱️ Retrieval", f"{result['retrieval_time']:.0f} ms")
            with col2:
                st.metric("🤖 LLM", f"{result['llm_time']:.0f} ms")
            
            st.metric("📄 Documents Found", result['docs_found'])
            
            # Display response
            if result["success"]:
                st.success(result["response"])
            else:
                st.error(result["response"])
            
            # Sources
            if result["sources"]:
                with st.expander("📚 View Sources"):
                    for src in result["sources"]:
                        st.caption(f"**{src['file']}** ({src['lang']})")
                        st.text(src['content'])
                        st.divider()
    
    # Performance comparison
    st.markdown("---")
    st.subheader("📊 Performance Comparison")
    
    perf_cols = st.columns(len(results))
    
    # Find fastest
    fastest_retrieval = min(results.values(), key=lambda x: x['retrieval_time'] if x['retrieval_time'] > 0 else float('inf'))
    fastest_total = min(results.values(), key=lambda x: x['total_time'] if x['total_time'] > 0 else float('inf'))
    
    for idx, (db_name, result) in enumerate(results.items()):
        with perf_cols[idx]:
            st.markdown(f"**{db_name}**")
            
            retrieval_label = "🏆 " if result == fastest_retrieval else ""
            total_label = "🏆 " if result == fastest_total else ""
            
            st.write(f"{retrieval_label}Retrieval: **{result['retrieval_time']:.2f} ms**")
            st.write(f"LLM: **{result['llm_time']:.2f} ms**")
            st.write(f"{total_label}Total: **{result['total_time']:.2f} ms**")
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "query": prompt,
        "responses": results
    })

# Footer
st.divider()
st.caption("RAG System | Powered by Ollama")
