import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

st.title("🤖 ITS International Student Chatbot (PostgreSQL)")
st.caption("Ask questions about the ITS International Students Guidebook")

# PostgreSQL connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")

# Create connection string
conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Test connection
try:
    with psycopg.connect(conn_str) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (COLLECTION_NAME,))
            collection_id = cur.fetchone()
            if collection_id:
                collection_id = collection_id[0]
                
                # Count embeddings
                cur.execute("""
                    SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s;
                """, (str(collection_id),))
                count = cur.fetchone()[0]
                
                st.sidebar.success(f"✅ Connected to PostgreSQL")
                st.sidebar.info(f"🗄️  Database: {DB_NAME}")
                st.sidebar.info(f"📊 Collection: {COLLECTION_NAME}")
                st.sidebar.info(f"📄 Documents: {count}")
            else:
                st.sidebar.error("Collection not found!")
                st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)}")
    st.error("Please run the ingestion script first!")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
prompt = st.chat_input("Ask about ITS International Student Guidelines...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(prompt))
    
    # Generate query embedding
    query_embedding = embeddings.embed_query(prompt)
    
    # Search for similar documents
    docs = []
    with psycopg.connect(conn_str) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    document,
                    1 - (embedding <=> %s::vector) as similarity
                FROM langchain_pg_embedding
                WHERE collection_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """, (query_embedding, str(collection_id), query_embedding))
            
            results = cur.fetchall()
            
            # Filter by similarity threshold
            for doc, similarity in results:
                if similarity >= 0.3:  # Adjust threshold as needed
                    docs.append((doc, similarity))
    
    if not docs or len(docs) == 0:
        # No relevant documents
        with st.chat_message("assistant"):
            response = "I'm sorry, I cannot answer that question as the information is not available in the ITS International Students Guidebook. Please ask questions related to the guidebook content."
            st.markdown(response)
        
        st.session_state.messages.append(AIMessage(response))
    else:
        # Relevant documents found
        docs_text = "\n\n".join([doc for doc, _ in docs])
        
        system_prompt = """You are an assistant that ONLY answers based on the provided document. STRICT RULES:

1. ONLY use information from the context below to answer
2. If the answer is NOT in the context, you MUST respond: "I'm sorry, that information is not available in the ITS International Students Guidebook."
3. DO NOT use knowledge outside the provided context
4. DO NOT make assumptions or conclusions beyond what is written in the context
5. DO NOT answer if you are not 100% sure the answer is in the context

Context from document:
{context}

REMEMBER: If the information is not in the context above, say you cannot answer!"""

        system_prompt_fmt = system_prompt.format(context=docs_text)
        
        # Debug info
        with st.sidebar:
            st.markdown("### 🔍 Debug Info")
            st.write(f"**Documents retrieved:** {len(docs)}")
            st.write(f"**Similarities:** {[f'{sim:.3f}' for _, sim in docs]}")
            with st.expander("View retrieved context"):
                st.text(docs_text[:500] + "...")
        
        # Generate response
        llm = ChatOllama(model="qwen3:8b", temperature=0.1)
        
        messages_with_context = [
            SystemMessage(system_prompt_fmt),
            HumanMessage(prompt)
        ]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in llm.stream(messages_with_context):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append(AIMessage(full_response))