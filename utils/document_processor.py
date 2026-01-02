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
    
    # --- Indonesian (30 queries) - Document 1: Panduan-Mahasiswa-Baru-DPTSI-2025 ---
    "Apa itu myITS Portal dan apa fungsinya?",
    "Bagaimana langkah-langkah login pertama kali ke myITS Portal?",
    "Apa format default email mahasiswa baru ITS?",
    "Bagaimana cara melakukan aktivasi Multi-Factor Authentication (MFA)?",
    "Apa syarat pembuatan password baru untuk akun myITS?",
    "Bagaimana cara verifikasi email alternatif dan nomor ponsel di myITS?",
    "Apa fungsi dari aplikasi SIAKAD (Sistem Informasi Akademik)?",
    "Bagaimana cara orang tua mahasiswa mengakses myITS Wali?",
    "Apa itu myITS Classroom dan bagaimana cara mengaksesnya?",
    "Bagaimana prosedur presensi kuliah menggunakan myITS Presensi?",
    "Apa saja layanan internet yang tersedia di lingkungan kampus ITS?",
    "Bagaimana cara menghubungkan perangkat ke jaringan Eduroam?",
    "Aplikasi apa yang wajib diunduh untuk keperluan MFA?",
    "Bagaimana cara menginstal Office 365 versi desktop untuk mahasiswa?",
    "Langkah-langkah aktivasi lisensi Office 365 menggunakan akun ITS.",
    "Bagaimana cara mengakses aplikasi Zoom berlisensi dari ITS?",
    "Apa kegunaan layanan VPN ITS bagi mahasiswa?",
    "Bagaimana cara setting OpenVPN Connect Client untuk VPN ITS?",
    "Apa saja fasilitas yang disediakan oleh Perpustakaan ITS?",
    "Bagaimana cara mahasiswa melakukan peminjaman buku secara online (OBOR)?",
    "Apa peran Direktorat Riset dan Pengabdian kepada Masyarakat (DRPM) bagi mahasiswa?",
    "Apa itu SIMPel ITS dan kegunaannya untuk penelitian?",
    "Di mana mahasiswa bisa mendapatkan informasi layanan untuk penyandang disabilitas?",
    "Apa saja provider seluler yang tidak bisa digunakan untuk verifikasi nomor ponsel di myITS?",
    "Ke mana mahasiswa harus menghubungi jika mengalami kendala layanan IT (Service Desk)?",
    "Bagaimana cara mengubah password melalui myITS Account?",
    "Apa password default yang digunakan saat login pertama kali?",
    "Bagaimana cara mengakses email ITS menggunakan Microsoft Outlook?",
    "Apa saja aplikasi yang tergabung dalam layanan Office 365 untuk mahasiswa?",
    "Bagaimana cara mengecek status verifikasi email alternatif?",

    # --- English (30 queries) - Document 2: General-Guidebook-for-International-Students ---
    "What items are on the arrival checklist for international students?",
    "How do I register my phone's IMEI in Indonesia?",
    "What is the tax calculation for registering a phone worth more than $500?",
    "What documents are required to apply for a Bachelor Degree (E30B) Visa?",
    "How much is the ITS administration fee for international students?",
    "What is the procedure for obtaining a Police Notification Report (STM)?",
    "How do I get a Proof of Residency (SKTT) and what is the deadline?",
    "What are the recommended mobile apps for finding accommodation in Surabaya?",
    "How to register for a local bank account at Mandiri?",
    "What are the emergency hotline numbers in Surabaya?",
    "What is the dress code for attending classes at ITS?",
    "Which hospitals are recommended near the ITS campus?",
    "What transport options are available from Juanda International Airport to ITS?",
    "How to use the Suroboyo Bus and what is the cost for students?",
    "What are the \"Dos and Don'ts\" regarding etiquette in Indonesia?",
    "Where is the Global Engagement Office located?",
    "What is the process for extending a tourist IMEI registration?",
    "List of faculties available under the SCIENTICS cluster.",
    "What is the \"O-Week\" and why is it mandatory?",
    "How to activate the \"Livin by Mandiri\" mobile banking app?",
    "What are the estimated living costs for accommodation near ITS?",
    "Where can students find traditional markets near the campus?",
    "What steps should be taken during an earthquake according to the disaster mitigation guide?",
    "What is the meaning of the Indonesian slang \"Mager\"?",
    "How can international students join Student Activity Units (UKM)?",
    "What are the requirements for an Internship Visa (C22A)?",
    "Locations of places of worship (mosques and churches) around ITS.",
    "What facilities are available at the ITS Medical Centre?",
    "How to use Gojek or Grab for local transportation?",
    "What is the penalty for failing to obtain Proof of Residency within 14 days?"
]
