"""
Documents - View source PDF documents
"""

import streamlit as st
import os

# Page config
st.set_page_config(
    page_title="Documents",
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
    
    .doc-card {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
    }
    
    .doc-title {
        color: #FAFAFA;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .doc-lang {
        color: #71717A;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style="text-align: center; padding: 32px 0 24px 0;">
    <h1 style="font-size: 2rem; font-weight: 600; color: #FAFAFA; margin-bottom: 4px; letter-spacing: -0.025em;">
        Documents
    </h1>
    <p style="color: #71717A; font-size: 0.875rem; font-weight: 400;">
        Source documents used for RAG benchmark
    </p>
</div>
""", unsafe_allow_html=True)

# Back button
if st.button("← Back to Benchmark"):
    st.switch_page("Benchmark.py")

st.markdown("---")

# Documents
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
    }
]

col1, col2 = st.columns(2)

for i, doc in enumerate(documents):
    with col1 if i == 0 else col2:
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-title">{doc['name']}</div>
            <div class="doc-lang">{doc['language']} • {doc['date']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        file_path = os.path.join("documents", doc['file'])
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"Download PDF",
                    data=f.read(),
                    file_name=doc['file'],
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.warning(f"File not found: {doc['file']}")
