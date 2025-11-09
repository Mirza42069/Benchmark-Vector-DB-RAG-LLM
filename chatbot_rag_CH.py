# import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import chromadb
import chromadb

# import langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("🤖 ITS International Student Chatbot (ChromaDB)")
st.caption("Ask questions about the ITS International Students Guidebook")

# Initialize ChromaDB client (local persistent storage)
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "its_guidebook"

# Create ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize vector store
vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Verify database is loaded
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()
    st.sidebar.success(f"✅ Connected to ChromaDB")
    st.sidebar.info(f"📚 Documents in database: {doc_count}")
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)}")
    st.error("Please run `ingestion.py` first to create the database!")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create the bar where we can type messages
prompt = st.chat_input("Ask about ITS International Student Guidelines...")

# Did the user submit a prompt?
if prompt:
    # Add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(prompt))

    # Initialize the llm (using Ollama)
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.1
    )

    # Creating and invoking the retriever with HIGHER threshold
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.75},
    )

    docs = retriever.invoke(prompt)
    
    # More strict checking
    if not docs or len(docs) == 0:
        # No relevant documents found
        with st.chat_message("assistant"):
            response = "I'm sorry, I cannot answer that question as the information is not available in the ITS International Students Guidebook. Please ask questions related to the guidebook content."
            st.markdown(response)
        
        st.session_state.messages.append(AIMessage(response))
    else:
        docs_text = "\n\n".join(d.page_content for d in docs)
        
        # Additional check: if retrieved text is too short, it's probably not relevant
        if len(docs_text.strip()) < 50:
            with st.chat_message("assistant"):
                response = "I'm sorry, I cannot answer that question as the information is not available in the ITS International Students Guidebook. Please ask questions related to the guidebook content."
                st.markdown(response)
            
            st.session_state.messages.append(AIMessage(response))
        else:
            # Relevant documents found - proceed with answering
            system_prompt = """You are an assistant that ONLY answers based on the provided document. STRICT RULES:

1. ONLY use information from the context below to answer
2. If the answer is NOT in the context, you MUST respond: "I'm sorry, that information is not available in the ITS International Students Guidebook."
3. DO NOT use knowledge outside the provided context
4. DO NOT make assumptions or conclusions beyond what is written in the context
5. DO NOT answer if you are not 100% sure the answer is in the context

Context from document:
{context}

REMEMBER: If the information is not in the context above, say you cannot answer!"""

            # Populate the system prompt with the retrieved context
            system_prompt_fmt = system_prompt.format(context=docs_text)

            # Debug info in sidebar
            with st.sidebar:
                st.markdown("### 🔍 Debug Info")
                st.write(f"**Documents retrieved:** {len(docs)}")
                st.write(f"**Context length:** {len(docs_text)} chars")
                with st.expander("View retrieved context"):
                    st.text(docs_text[:500] + "...")

            # Create a temporary message list with the system prompt
            messages_with_context = [
                SystemMessage(system_prompt_fmt),
                HumanMessage(prompt)
            ]

            # Invoking the llm with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in llm.stream(messages_with_context):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append(AIMessage(full_response))