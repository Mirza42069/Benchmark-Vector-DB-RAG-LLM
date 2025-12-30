"""
Document Processing Utilities for RAG System
Handles PDF loading, language detection, and text chunking
"""

import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Processes PDF documents for RAG system"""
    
    def __init__(self, documents_dir: str = "documents/"):
        self.documents_dir = documents_dir
        
    def load_pdfs(self) -> Tuple[List[Document], List[str]]:
        """Load all PDF files from documents directory"""
        all_documents = []
        pdf_files = []
        
        if not os.path.exists(self.documents_dir):
            raise FileNotFoundError(f"Directory '{self.documents_dir}' not found!")
        
        pdf_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{self.documents_dir}'")
        
        print(f"\n📚 Loading {len(pdf_files)} PDF file(s)...")
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.documents_dir, pdf_file)
            print(f"  📄 Loading: {pdf_file}")
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Add source file to metadata
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file
                
                all_documents.extend(docs)
                print(f"     ✓ Loaded {len(docs)} pages")
            except Exception as e:
                print(f"     ✗ Error loading {pdf_file}: {str(e)}")
        
        print(f"\n✅ Total pages loaded: {len(all_documents)}")
        return all_documents, pdf_files
    
    def detect_language(self, text: str) -> str:
        """Detect document language (Indonesian or English)"""
        text_lower = text.lower()
        text_sample = text_lower[:1000]
        
        indonesian_indicators = [
            'dan', 'dengan', 'untuk', 'yang', 'adalah', 'ini', 'akan', 'pada', 'di', 'ke',
            'atau', 'dari', 'tidak', 'sebagai', 'dalam', 'dapat', 'juga', 'oleh', 'telah',
            'mahasiswa', 'sistem', 'informasi', 'layanan', 'akun', 'aplikasi', 'kampus',
            'panduan', 'menggunakan', 'tersedia', 'melalui', 'myits', 'siakad', 'dptsi'
        ]
        
        english_indicators = [
            'the', 'and', 'for', 'are', 'with', 'this', 'will', 'on', 'in', 'to',
            'you', 'can', 'your', 'have', 'from', 'that', 'student', 'students',
            'international', 'guidebook', 'please', 'must', 'should', 'visa', 'passport'
        ]
        
        indonesian_count = sum(1 for word in indonesian_indicators if f' {word} ' in f' {text_sample} ')
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_sample} ')
        
        if indonesian_count > english_count * 1.5:
            return "id"
        elif english_count > indonesian_count:
            return "en"
        else:
            return "mixed"
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with unified settings for fair benchmarking"""
        print("\n✂️  Splitting documents into chunks...")
        
        # Detect language for each document
        for doc in documents:
            lang = self.detect_language(doc.page_content)
            doc.metadata['detected_language'] = lang
        
        # Unified splitter for all languages (fair benchmarking)
        # Using consistent chunk size ensures equal comparison across databases
        unified_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,      # Unified size for fair benchmarking
            chunk_overlap=180,   # Unified overlap
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Separate by language for statistics
        indonesian_docs = [doc for doc in documents if doc.metadata.get('detected_language') == 'id']
        english_docs = [doc for doc in documents if doc.metadata.get('detected_language') == 'en']
        mixed_docs = [doc for doc in documents if doc.metadata.get('detected_language') == 'mixed']
        
        print(f"  📊 Document distribution:")
        print(f"     🇮🇩 Indonesian: {len(indonesian_docs)} pages")
        print(f"     🇬🇧 English: {len(english_docs)} pages")
        print(f"     ❓ Mixed: {len(mixed_docs)} pages")
        
        # Split all documents with unified splitter
        chunks_indonesian = unified_splitter.split_documents(indonesian_docs) if indonesian_docs else []
        chunks_english = unified_splitter.split_documents(english_docs) if english_docs else []
        chunks_mixed = unified_splitter.split_documents(mixed_docs) if mixed_docs else []
        
        all_chunks = chunks_indonesian + chunks_english + chunks_mixed
        
        print(f"\n✅ Created {len(all_chunks)} total chunks:")
        print(f"   🇮🇩 Indonesian chunks: {len(chunks_indonesian)}")
        print(f"   🇬🇧 English chunks: {len(chunks_english)}")
        print(f"   ❓ Mixed chunks: {len(chunks_mixed)}")
        print(f"   📏 Chunk size: 900 chars | Overlap: 180 chars")
        
        return all_chunks
    
    def enhance_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add enhanced metadata to chunks"""
        print("\n🏷️  Enhancing metadata...")
        
        for i, doc in enumerate(chunks):
            chunk_lang = self.detect_language(doc.page_content[:500])
            doc.metadata['chunk_language'] = chunk_lang
            doc.metadata['chunk_id'] = i + 1
            doc.metadata['chunk_length'] = len(doc.page_content)
            
            if 'source' in doc.metadata and 'source_file' not in doc.metadata:
                doc.metadata['source_file'] = os.path.basename(doc.metadata['source'])
        
        return chunks
    
    def process_documents(self) -> List[Document]:
        """Complete document processing pipeline"""
        # Load PDFs
        documents, pdf_files = self.load_pdfs()
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Enhance metadata
        chunks = self.enhance_metadata(chunks)
        
        return chunks


# Sample test queries for benchmarking
# Indonesian queries: Based on "Panduan-Mahasiswa-Baru-DPTSI-2025" (IT services guide)
# English queries: Based on "General-Guidebook-for-International-Students_July-2024"
SAMPLE_QUERIES = [
    # ========== 60 RELEVANT QUERIES ==========
    
    # --- Indonesian (30 queries) - DPTSI IT Services Guide ---
    "Apa itu myITS Portal dan bagaimana cara mengaksesnya?",
    "Bagaimana cara mengubah password myITS Portal?",
    "Aplikasi apa saja yang bisa diakses melalui myITS Portal?",
    "Bagaimana cara mengaktivasi Multi-Factor Authentication (MFA)?",
    "Apa saja layanan internet yang tersedia di kampus ITS?",
    "Bagaimana cara mengakses Office 365 untuk mahasiswa ITS?",
    "Bagaimana cara menyambungkan perangkat ke WiFi ITS?",
    "Apa itu myITS SSO dan bagaimana cara menggunakannya?",
    "Bagaimana cara mengakses email ITS?",
    "Apa saja fitur yang tersedia di Microsoft Teams untuk mahasiswa?",
    "Bagaimana cara mengunduh Microsoft Office gratis untuk mahasiswa?",
    "Apa itu myITS StudentConnect?",
    "Bagaimana cara mengakses SIAKAD ITS?",
    "Apa saja layanan yang disediakan DPTSI untuk mahasiswa?",
    "Bagaimana cara reset password akun ITS?",
    "Apa itu ITS Single Sign-On?",
    "Bagaimana cara mengakses Google Workspace ITS?",
    "Apa saja aplikasi yang terintegrasi dengan myITS?",
    "Bagaimana cara mengaktifkan akun mahasiswa baru di ITS?",
    "Apa itu myITS dan apa bedanya dengan SIAKAD?",
    "Bagaimana cara mendapatkan lisensi software gratis dari ITS?",
    "Apa saja jaringan WiFi yang tersedia di kampus ITS?",
    "Bagaimana cara menghubungi helpdesk DPTSI?",
    "Apa itu VPN ITS dan bagaimana cara menggunakannya?",
    "Bagaimana cara mengakses e-learning ITS?",
    "Apa saja layanan cloud storage yang tersedia untuk mahasiswa?",
    "Bagaimana cara menggunakan OneDrive ITS?",
    "Apa itu ITS Repository dan bagaimana cara mengaksesnya?",
    "Bagaimana cara mengakses jurnal online dari perpustakaan ITS?",
    "Apa saja panduan keamanan akun yang disarankan DPTSI?",

    # --- English (30 queries) - International Student Guidebook ---
    "What documents do I need to bring when arriving in Surabaya?",
    "How many types of academic VISA are available for international students?",
    "What is the ITS administration fee for international students?",
    "Where is the ITS Global Engagement Office located?",
    "How do I register my phone's IMEI for a long stay in Indonesia?",
    "What are the internet package options for Tourist SIM Card?",
    "How much does accommodation cost per month near ITS?",
    "What is the emergency hotline number in Surabaya?",
    "Which banks are available inside ITS campus?",
    "What are the nearest hospitals to ITS?",
    "What is the fine for not having proof of residency within 14 days?",
    "Which transportation apps can I use in Surabaya?",
    "What faculties are available at ITS?",
    "How long does the admission evaluation process take?",
    "How do I open a bank account in Indonesia?",
    "What are the requirements for extending a study visa?",
    "Who should I contact in case of an emergency?",
    "Where can I find halal food near the campus?",
    "What sports facilities are available for students?",
    "What is the procedure for visa on arrival?",
    "How do I get a KITAS (temporary stay permit)?",
    "What are the living costs in Surabaya for students?",
    "Where can I exchange foreign currency in Surabaya?",
    "What is the process for airport pickup service?",
    "How do I get Indonesian phone number (SIM card)?",
    "What are the requirements for ITAS application?",
    "Where is the immigration office for visa extension?",
    "What public transportation is available in Surabaya?",
    "How do I register at the local immigration office?",
    "What health insurance options are available for international students?"
]
