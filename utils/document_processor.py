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
        """Split documents into chunks based on detected language"""
        print("\n✂️  Splitting documents into chunks...")
        
        # Detect language for each document
        for doc in documents:
            lang = self.detect_language(doc.page_content)
            doc.metadata['detected_language'] = lang
        
        # Configure splitters
        indonesian_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "。", " ", ""]
        )
        
        english_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Separate by language
        indonesian_docs = [doc for doc in documents if doc.metadata.get('detected_language') == 'id']
        english_docs = [doc for doc in documents if doc.metadata.get('detected_language') == 'en']
        mixed_docs = [doc for doc in documents if doc.metadata.get('detected_language') == 'mixed']
        
        print(f"  📊 Document distribution:")
        print(f"     🇮🇩 Indonesian: {len(indonesian_docs)} pages")
        print(f"     🇬🇧 English: {len(english_docs)} pages")
        print(f"     ❓ Mixed: {len(mixed_docs)} pages")
        
        # Split documents
        chunks_indonesian = indonesian_splitter.split_documents(indonesian_docs) if indonesian_docs else []
        chunks_english = english_splitter.split_documents(english_docs) if english_docs else []
        chunks_mixed = english_splitter.split_documents(mixed_docs) if mixed_docs else []
        
        all_chunks = chunks_indonesian + chunks_english + chunks_mixed
        
        print(f"\n✅ Created {len(all_chunks)} total chunks:")
        print(f"   🇮🇩 Indonesian chunks: {len(chunks_indonesian)}")
        print(f"   🇬🇧 English chunks: {len(chunks_english)}")
        print(f"   ❓ Mixed chunks: {len(chunks_mixed)}")
        
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
SAMPLE_QUERIES = [
    # ========== 30 RELEVANT QUERIES (From documents) ==========
    # Indonesian (15 queries)
    "Dokumen apa yang harus dibawa saat tiba di Surabaya?",
    "Berapa biaya administrasi ITS untuk mahasiswa internasional?",
    "Di mana lokasi ITS Global Engagement Office?",
    "Bagaimana cara mendaftar IMEI ponsel untuk tinggal lama di Indonesia?",
    "Berapa biaya akomodasi per bulan di dekat ITS?",
    "Apa itu myITS Portal dan bagaimana cara mengaksesnya?",
    "Bagaimana cara mengubah password myITS Portal?",
    "Aplikasi apa saja yang bisa diakses melalui myITS Portal?",
    "Bagaimana cara mengaktivasi Multi-Factor Authentication (MFA)?",
    "Apa saja layanan internet yang tersedia di kampus ITS?",
    "Bagaimana cara mengakses Office 365 untuk mahasiswa ITS?",
    "Fakultas apa saja yang ada di ITS?",
    "Berapa lama proses evaluasi penerimaan mahasiswa?",
    "Apa saja jenis visa akademik yang tersedia untuk mahasiswa internasional?",
    "Berapa denda jika tidak memiliki bukti tempat tinggal dalam 14 hari?",
    
    # English (15 queries)
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
    "How do I access the MyITS system?",
    "What is the fine for not having proof of residency within 14 days?",
    "Which transportation apps can I use in Surabaya?",
    "What faculties are available at ITS?",
    "How long does the admission evaluation process take?",
    
    # ========== 30 IRRELEVANT QUERIES (Not in documents) ==========
    # Indonesian (15 queries)
    "Berapa IPK minimum untuk program pascasarjana?",
    "Berapa biaya kuliah per semester untuk program sarjana?",
    "Berapa jumlah beasiswa yang tersedia untuk mahasiswa internasional?",
    "Bisakah Anda menjelaskan kurikulum program Teknik Mesin?",
    "Berapa tingkat penerimaan mahasiswa internasional di ITS?",
    "Bagaimana cara melamar magang di perusahaan di Surabaya?",
    "Apa prospek kerja setelah lulus dari ITS?",
    "Bagaimana peringkat ITS dibandingkan universitas Indonesia lainnya?",
    "Bagaimana cara mendapatkan izin kerja setelah lulus?",
    "Apa universitas terbaik di Jakarta?",
    "Bagaimana cuaca di Bali saat musim panas?",
    "Bagaimana cara mengkonversi kredit Indonesia ke kredit ECTS?",
    "Kapan upacara wisuda tahun ini?",
    "Siapa rektor ITS saat ini?",
    "Apa persyaratan untuk pindah jurusan?",
    
    # English (15 queries)
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
]