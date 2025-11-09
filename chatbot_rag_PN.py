#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.title("Chatbot")

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")  # change if desired
index = pc.Index(index_name)

# initialize embeddings model + vector store (using Ollama)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("How are you?")

# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(prompt))

    # initialize the llm (using Ollama)
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.1
    )

    # creating and invoking the retriever with HIGHER threshold
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.75},  # Increased from 0.5 to 0.75
    )

    docs = retriever.invoke(prompt)
    
    # More strict checking
    if not docs or len(docs) == 0:
        # No relevant documents found
        with st.chat_message("assistant"):
            response = "Maaf, saya tidak dapat menjawab pertanyaan tersebut karena informasi tidak tersedia dalam dokumen yang saya miliki. Silakan tanyakan hal yang berkaitan dengan konten dokumen yang telah diunggah."
            st.markdown(response)
        
        st.session_state.messages.append(AIMessage(response))
    else:
        docs_text = "\n\n".join(d.page_content for d in docs)
        
        # Additional check: if retrieved text is too short, it's probably not relevant
        if len(docs_text.strip()) < 50:
            with st.chat_message("assistant"):
                response = "Maaf, saya tidak dapat menjawab pertanyaan tersebut karena informasi tidak tersedia dalam dokumen yang saya miliki. Silakan tanyakan hal yang berkaitan dengan konten dokumen yang telah diunggah."
                st.markdown(response)
            
            st.session_state.messages.append(AIMessage(response))
        else:
            # Relevant documents found - proceed with answering
            # creating the VERY strict system prompt
            system_prompt = """Anda adalah asisten yang HANYA menjawab berdasarkan dokumen yang diberikan. Aturan KETAT:

1. HANYA gunakan informasi dari konteks di bawah ini untuk menjawab
2. Jika jawaban TIDAK ADA dalam konteks, Anda HARUS menjawab: "Maaf, informasi tersebut tidak tersedia dalam dokumen yang saya miliki."
3. JANGAN gunakan pengetahuan di luar konteks yang diberikan
4. JANGAN membuat asumsi atau kesimpulan di luar apa yang tertulis dalam konteks
5. JANGAN menjawab jika Anda tidak 100% yakin jawabannya ada dalam konteks

Konteks dari dokumen:
{context}

INGAT: Jika informasi tidak ada dalam konteks di atas, katakan Anda tidak bisa menjawab!"""

            # Populate the system prompt with the retrieved context
            system_prompt_fmt = system_prompt.format(context=docs_text)

            print("-- SYS PROMPT --")
            print(system_prompt_fmt)
            print("-- DOCS RETRIEVED --")
            print(f"Number of docs: {len(docs)}")
            print(f"Context length: {len(docs_text)}")

            # Create a temporary message list with the system prompt
            messages_with_context = [
                SystemMessage(system_prompt_fmt),
                HumanMessage(prompt)
            ]

            # invoking the llm with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in llm.stream(messages_with_context):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append(AIMessage(full_response))