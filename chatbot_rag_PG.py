"""
RAG Chatbot with PostgreSQL + pgvector
"""

import streamlit as st
import os
from dotenv import load_dotenv
import time
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Page config
st.set_page_config(
    page_title="ITS Helpdesk Chatbot - PostgreSQL",
    page_icon="🤖",
    layout="wide"
)

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "3"))

connection_string = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Header
st.title("🤖 ITS Helpdesk Chatbot")
st.caption("Powered by PostgreSQL + pgvector")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"**Vector Database:** PostgreSQL + pgvector")
    st.info(f"**Database:** {DB_NAME}")
    st.info(f"**Collection:** {COLLECTION_NAME}")
    st.info(f"**Embedding:** {EMBEDDING_MODEL}")
    st.info(f"**LLM:** {CHAT_MODEL}")
    
    st.divider()
    
    st.header("📊 Settings")
    score_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=SCORE_THRESHOLD,
        step=0.05,
        help="Minimum similarity score for retrieval"
    )
    
    top_k = st.slider(
        "Number of Documents",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="Number of documents to retrieve"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### 💡 Sample Questions
    
    **Indonesian:**
    - Bagaimana cara mengubah password myITS Portal?
    - Apa itu Multi-Factor Authentication?
    - Berapa biaya akomodasi per bulan di dekat ITS?
    
    **English:**
    - What documents do I need when arriving in Surabaya?
    - How do I register my phone's IMEI?
    - Which banks are available inside ITS campus?
    """)

# Initialize PostgreSQL vector store
@st.cache_resource
def init_vector_store():
    """Initialize PostgreSQL vector store"""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=connection_string,
            use_jsonb=True,
        )
        return vector_store
    except Exception as e:
        st.error(f"Error initializing PostgreSQL: {str(e)}")
        return None

# Load vector store
if st.session_state.vector_store is None:
    with st.spinner("🔄 Connecting to PostgreSQL..."):
        st.session_state.vector_store = init_vector_store()
        if st.session_state.vector_store:
            st.success("✅ Connected to PostgreSQL successfully!")
        else:
            st.error("❌ Failed to connect to PostgreSQL. Please check your configuration.")
            st.stop()

vector_store = st.session_state.vector_store

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source['file']}")
                    st.caption(f"Language: {source['lang']} | Similarity: {source['score']:.3f}")
                    st.text(source['content'][:200] + "...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask me anything about ITS..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            # Measure retrieval time
            start_time = time.time()
            
            try:
                # Retrieve documents
                retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": top_k, "score_threshold": score_threshold}
                )
                docs = retriever.invoke(prompt)
                
                retrieval_time = time.time() - start_time
                
                if docs and len(docs) > 0:
                    # Prepare context
                    docs_text = "\n\n".join(d.page_content for d in docs)
                    
                    # Create system prompt
                    system_prompt = f"""You are a helpful assistant for Institut Teknologi Sepuluh Nopember (ITS) students.
Answer questions based ONLY on the provided context. If the answer is not in the context, say so politely.
Respond in the same language as the question (Indonesian or English).

Context:
{docs_text}"""
                    
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt)
                    ]
                    
                    # Generate response
                    llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
                    
                    start_llm = time.time()
                    response = llm.invoke(messages)
                    llm_time = time.time() - start_llm
                    
                    # Display response
                    st.markdown(response.content)
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Retrieval Time", f"{retrieval_time*1000:.0f} ms")
                    with col2:
                        st.metric("🤖 Generation Time", f"{llm_time*1000:.0f} ms")
                    with col3:
                        st.metric("📄 Documents Found", len(docs))
                    
                    # Prepare sources
                    sources = []
                    for doc in docs:
                        sources.append({
                            "file": doc.metadata.get('source_file', 'Unknown'),
                            "lang": doc.metadata.get('chunk_language', 'unknown'),
                            "score": doc.metadata.get('score', 0.0) if hasattr(doc, 'metadata') else 0.0,
                            "content": doc.page_content
                        })
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "sources": sources
                    })
                    
                else:
                    response_text = "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing or ask something else about ITS."
                    st.warning(response_text)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.caption("🎓 ITS Helpdesk Chatbot | Using PostgreSQL + pgvector | Powered by Ollama")