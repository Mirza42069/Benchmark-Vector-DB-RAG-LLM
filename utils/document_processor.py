"""
Document Processing Utilities for RAG System
Handles PDF loading, language detection, and text chunking
"""

import os
from typing import Any, List, Tuple
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
            chunk_size=500,      # Unified size for fair benchmarking (fits mxbai-embed-large context)
            chunk_overlap=100,   # Unified overlap
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
        print(f"   📏 Chunk size: 500 chars | Overlap: 100 chars")
        
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


BENCHMARK_DOCUMENTS = [
    {
        "name": "Panduan Riset Penugasan ITS SRG 2026",
        "file": "2026_Panduan-Riset-Penugasan-ITS-SRG-1.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Buku Panduan Corporate User",
        "file": "Buku-Panduan-untuk-Corporate-User.pdf",
        "language": "Indonesian",
    },
    {
        "name": "General Guidebook for International Students",
        "file": "General-Guidebook-for-International-Students_July-2024.pdf",
        "language": "English",
    },
    {
        "name": "Perjanjian Angkutan dengan Penumpang MRT Jakarta",
        "file": "MAN 01 - Perjanjian Angkutan dengan Penumpang 2021.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Jatim Melaju 2026",
        "file": "PANDUAN-RISET-JATIM-MELAJU-2026-PD-DRPM-ITS-022.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Penelitian Dana ITS 2026",
        "file": "PD-DRPM-ITS-001-Panduan-Penelitian-Dana-ITS-Tahun-2026.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Abmas ITS 2026",
        "file": "PD-DRPM-ITS-002-Panduan-Abmas-ITS-Tahun-2026.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Bantuan Biaya APC Dana ITS 2026",
        "file": "PD-DRPM-ITS-0024-Panduan-Bantuan-Biaya-APC-Dana-ITS-Tahun-2026.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Insentif Publikasi Terindeks Dana ITS 2026",
        "file": "PD-DRPM-ITS-005-Panduan-Insentif-Publikasi-Terindeks-Dana-ITS-Tahun-2026.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Penelitian dan Abmas Tahun 2026",
        "file": "PD-DRPM-ITS-023-Panduan-Penelitian-dan-Abmas-Tahun-2026.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Insentif Publikasi Top Tiers EQUITY WCU 2026",
        "file": "PD-DRPM-ITS-026-Panduan-Insentif-Publikasi-Top-Tiers-Dana-EQUITY-WCU-Tahun-2026.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Fast-D, Beasiswa Unggulan, dan GES 2026",
        "file": "PD_DRPM_ITS_021_Panduan-Fast-D_Beasiswa-Unggulan_GES.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Kopra by Mandiri Bill Payment",
        "file": "Panduan-Kopra-by-Mandiri-Bill-Payment-ID.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Kopra by Mandiri Reports Cek Saldo",
        "file": "Panduan-Kopra-by-Mandiri-Reports-Cek-Saldo.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Mahasiswa Baru DPTSI 2025",
        "file": "Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan PMKI ITS 2026",
        "file": "Panduan-PMKI_ITS_2026-1.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Penelitian Frontiers Dana HETI",
        "file": "Panduan-Penelitian-Front-Tiers-Dana-Heti-1.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan RKI ITS 2026",
        "file": "Panduan-RKI_ITS_2026-.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Post-Doctoral 2026",
        "file": "Post-Doctoral-2026-1-1.pdf",
        "language": "Indonesian",
    },
    {
        "name": "Panduan Bantuan Biaya APC EQUITY WCU 2026",
        "file": "PD-DRPM-ITS-0025-Panduan-Bantuan-Biaya-APC-Dana-EQUITY-WCU-Tahun-2026-1.pdf",
        "language": "Indonesian",
    },
]


def _append_discovered_benchmark_documents(max_documents: int = 100) -> None:
    known_files = {document["file"] for document in BENCHMARK_DOCUMENTS}
    documents_dir = "documents"
    if not os.path.isdir(documents_dir):
        return

    discovered_files = sorted(
        file_name
        for file_name in os.listdir(documents_dir)
        if file_name.lower().endswith(".pdf") and file_name not in known_files
    )
    for file_name in discovered_files:
        if len(BENCHMARK_DOCUMENTS) >= max_documents:
            break
        BENCHMARK_DOCUMENTS.append(
            {
                "name": os.path.splitext(file_name)[0].replace("-", " ").replace("_", " "),
                "file": file_name,
                "language": "Unknown",
            }
        )


_append_discovered_benchmark_documents()


# Each document contributes 5 answerable benchmark queries.
DOCUMENT_QUERY_SETS = [
    {
        "file": "2026_Panduan-Riset-Penugasan-ITS-SRG-1.pdf",
        "correct_queries": [
            {
                "query": "Berapa dana maksimal untuk skema SRG Tipe D pada panduan SRG 2026?",
                "keywords": ["SRG", "Tipe D", "Rp100.000.000"],
            },
            {
                "query": "Apa syarat H-index Scopus untuk pengusul SRG Tipe A?",
                "keywords": ["SRG", "Tipe A", "H-index", "4"],
            },
            {
                "query": "Apa target luaran minimum pada skema SRG Tipe B?",
                "keywords": ["SRG", "Tipe B", "Scopus", "Q2"],
            },
            {
                "query": "Kapan periode penerimaan proposal Riset Penugasan ITS SRG 2026?",
                "keywords": ["Penerimaan Proposal", "12", "25 Januari 2026", "SRG"],
            },
            {
                "query": "Apa rekomendasi co-authorship untuk SRG Tipe C?",
                "keywords": ["SRG Tipe C", "co-authorship", "mitra Industri"],
            },
        ],
    },
    {
        "file": "Buku-Panduan-untuk-Corporate-User.pdf",
        "correct_queries": [
            {
                "query": "Apa URL login Mandiri Cash Management 2.0 untuk corporate user?",
                "keywords": ["MCM 2.0", "login", "mcm2.bankmandiri.co.id"],
            },
            {
                "query": "Peran apa yang bertugas melepas transaksi yang sudah disetujui di MCM 2.0?",
                "keywords": ["Releaser", "transaksi", "approved"],
            },
            {
                "query": "Berapa limit maksimum transfer Online ke bank lain di MCM 2.0?",
                "keywords": ["Online", "bank lain", "Rp50 juta"],
            },
            {
                "query": "Apa tugas user Maker dalam Mandiri Cash Management 2.0?",
                "keywords": ["Maker", "penggagas", "transaksi"],
            },
            {
                "query": "Berapa periode maksimal histori saldo yang dapat dilihat pada menu balance history MCM 2.0?",
                "keywords": ["balance history", "maksimal 12 bulan", "periode inquiry 1 bulan"],
            },
        ],
    },
    {
        "file": "General-Guidebook-for-International-Students_July-2024.pdf",
        "correct_queries": [
            {
                "query": "How much is the ITS additional administration fee for international students?",
                "keywords": ["administration fee", "$170", "international students"],
            },
            {
                "query": "What is the emergency hotline number in Surabaya?",
                "keywords": ["emergency", "hotline", "112", "Surabaya"],
            },
            {
                "query": "How many student activity units (UKM) are listed in the guidebook?",
                "keywords": ["UKM", "38", "student activity"],
            },
            {
                "query": "What app must international students download to fill in the e-HAC form before departure?",
                "keywords": ["Satu Sehat", "e-HAC", "before departure"],
            },
            {
                "query": "Within how many days after arrival should students register their IMEI to avoid full device-cost tax?",
                "keywords": ["IMEI", "5 days", "full cost"],
            },
        ],
    },
    {
        "file": "MAN 01 - Perjanjian Angkutan dengan Penumpang 2021.pdf",
        "correct_queries": [
            {
                "query": "Jam berapa layanan kereta MRT Jakarta mulai beroperasi setiap hari?",
                "keywords": ["MRT Jakarta", "05.00", "operasi"],
            },
            {
                "query": "Berapa harga kartu Multi Trip Ticket (MTT)?",
                "keywords": ["MTT", "Rp25.000", "kartu"],
            },
            {
                "query": "Berapa denda maksimal untuk merokok di area MRT Jakarta?",
                "keywords": ["merokok", "denda", "Rp50.000.000"],
            },
            {
                "query": "Berapa tinggi badan anak-anak yang tidak dikenakan biaya jika bepergian dengan orang dewasa di MRT Jakarta?",
                "keywords": ["Anak-anak", "kurang dari 90 cm", "tidak dikenakan biaya"],
            },
            {
                "query": "Berapa saldo maksimum kartu MTT dan batas total isi ulang dalam satu bulan?",
                "keywords": ["MTT", "Rp 1.000.000", "Rp20.000.000"],
            },
        ],
    },
    {
        "file": "PANDUAN-RISET-JATIM-MELAJU-2026-PD-DRPM-ITS-022.pdf",
        "correct_queries": [
            {
                "query": "Siapa saja perguruan tinggi yang termasuk Cluster A pada skema Jatim Melaju 2026?",
                "keywords": ["Cluster A", "ITS", "UNAIR", "UB", "UNESA", "UM"],
            },
            {
                "query": "Berapa dana maksimal yang dapat diajukan peneliti ITS pada program Jatim Melaju 2026?",
                "keywords": ["Jatim Melaju", "Rp50.000.000", "ITS"],
            },
            {
                "query": "Luaran publikasi minimum apa yang wajib dipenuhi pada program Jatim Melaju 2026?",
                "keywords": ["luaran", "Scopus-Q2", "artikel jurnal internasional"],
            },
            {
                "query": "Kapan unggah full proposal Program Jatim Melaju 2026 dijadwalkan?",
                "keywords": ["Unggah Full Proposal", "28 Februari", "31 Maret 2026", "Jatim Melaju"],
            },
            {
                "query": "Apa kualifikasi H-index Scopus minimum untuk host Jatim Melaju pada kluster sains dan teknologi?",
                "keywords": ["Host", "h-index", "4", "Scopus"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-001-Panduan-Penelitian-Dana-ITS-Tahun-2026.pdf",
        "correct_queries": [
            {
                "query": "Apa topik penelitian Flagship ITS tahun 2026 pada panduan Penelitian Dana ITS?",
                "keywords": ["Flagship", "2026", "Regenerative Agrivoltaics"],
            },
            {
                "query": "Berapa dana maksimal untuk skema Penelitian Artikel Review?",
                "keywords": ["Artikel Review", "Rp10.000.000"],
            },
            {
                "query": "Skema penelitian apa saja yang termasuk Penelitian Pendukung Strategis ITS?",
                "keywords": ["Pendukung Strategis", "PRIMA", "NUSANTARA", "CENDEKIA"],
            },
            {
                "query": "Kapan penerimaan proposal Batch 2 Penelitian Kompetisi Dana ITS 2026?",
                "keywords": ["Batch 2", "Penerimaan Proposal", "20 April", "4 Mei 2026"],
            },
            {
                "query": "Berapa dana maksimum untuk skema Penelitian Flagship ITS?",
                "keywords": ["Penelitian Flagship ITS", "Rp. 300.000.000", "SATRIA"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-002-Panduan-Abmas-ITS-Tahun-2026.pdf",
        "correct_queries": [
            {
                "query": "Berapa dana maksimal untuk skema Abmas Prioritas ITS 2026?",
                "keywords": ["Abmas Prioritas", "Rp100.000.000"],
            },
            {
                "query": "Berapa dana maksimal untuk skema Abmas Berbasis Produk ITS 2026?",
                "keywords": ["Abmas Berbasis Produk", "Rp50.000.000"],
            },
            {
                "query": "Berapa dana maksimal untuk skema Abmas Internasional Tipe A ITS 2026?",
                "keywords": ["Abmas Internasional Tipe A", "Rp150.000.000"],
            },
            {
                "query": "Berapa durasi Abmas Berbasis Produk ITS 2026?",
                "keywords": ["Abmas Berbasis Produk", "6-8", "durasi"],
            },
            {
                "query": "Apa luaran utama yang ditekankan pada kegiatan Abmas ITS 2026?",
                "keywords": ["Jurnal Nasional", "Sewagati", "Sinta 4", "HKI"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-0024-Panduan-Bantuan-Biaya-APC-Dana-ITS-Tahun-2026.pdf",
        "correct_queries": [
            {
                "query": "Berapa nilai maksimal bantuan biaya APC Dana ITS tahun 2026?",
                "keywords": ["APC", "Rp40.000.000", "Dana ITS"],
            },
            {
                "query": "Apakah artikel harus sudah terbit atau cukup accepted untuk mengajukan bantuan APC Dana ITS?",
                "keywords": ["published", "terbit", "APC"],
            },
            {
                "query": "Berapa persentil minimal jurnal Scopus yang disyaratkan pada bantuan APC Dana ITS 2026?",
                "keywords": ["Scopus", "Q1", "90", "persentil"],
            },
            {
                "query": "Penerbit apa saja yang tidak diperbolehkan untuk artikel submission tahun 2026 pada bantuan APC Dana ITS?",
                "keywords": ["MDPI", "Frontiers", "Hindawi Publisher"],
            },
            {
                "query": "Apa tautan pendaftaran bantuan biaya APC Dana ITS 2026?",
                "keywords": ["https://its.id/BantuanAPC-DanaITS2026", "APC", "dokumen pendukung"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-005-Panduan-Insentif-Publikasi-Terindeks-Dana-ITS-Tahun-2026.pdf",
        "correct_queries": [
            {
                "query": "Berapa insentif untuk artikel jurnal Scopus Q2 pada panduan insentif publikasi terindeks 2026?",
                "keywords": ["insentif", "Scopus Q2", "Rp15.000.000"],
            },
            {
                "query": "Berapa insentif untuk artikel Q1 dengan persentil minimal 90 dan co-author luar negeri?",
                "keywords": ["Q1", "90", "co-author luar negeri", "Rp35.000.000"],
            },
            {
                "query": "Kapan jadwal pengajuan Batch I insentif publikasi terindeks tahun 2026?",
                "keywords": ["Batch I", "17 Maret 2026", "6 April 2026"],
            },
            {
                "query": "Kapan jadwal Batch II dan Batch III insentif publikasi terindeks tahun 2026?",
                "keywords": ["Batch II 2026", "Agustus 2026", "Batch III 2026", "Oktober 2026"],
            },
            {
                "query": "Berapa insentif untuk artikel jurnal Scopus Q1 tanpa persyaratan persentil 90 pada panduan insentif publikasi terindeks?",
                "keywords": ["Q1", "30.000.000", "insentif"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-023-Panduan-Penelitian-dan-Abmas-Tahun-2026.pdf",
        "correct_queries": [
            {
                "query": "Berapa dana maksimal untuk skema Abmas Tematik Tipe C pada panduan unit kerja 2026?",
                "keywords": ["Abmas Tematik Tipe C", "Rp50.000.000"],
            },
            {
                "query": "Apa luaran minimum untuk skema Penelitian Artikel Review pada panduan unit kerja 2026?",
                "keywords": ["Artikel Review", "Scopus-Q2", "luaran"],
            },
            {
                "query": "Berapa jumlah mahasiswa KKN yang wajib dilibatkan pada Abmas Tematik Tipe B?",
                "keywords": ["KKN", "Tipe B", "mahasiswa"],
            },
            {
                "query": "Berapa honorarium maksimal Post Doctoral Fellow pada skema Post Doctoral Dana Unit Kerja?",
                "keywords": ["Post Doctoral", "honorarium", "7.000.000", "minimal 6 bulan"],
            },
            {
                "query": "Berapa durasi video kegiatan yang disyaratkan pada standar luaran Abmas Dana Unit Kerja?",
                "keywords": ["Video", "durasi 3-5 menit", "testimoni dari mitra"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-026-Panduan-Insentif-Publikasi-Top-Tiers-Dana-EQUITY-WCU-Tahun-2026.pdf",
        "correct_queries": [
            {
                "query": "Berapa persentil Scopus yang digunakan untuk mendefinisikan publikasi Top Tier pada panduan EQUITY WCU 2026?",
                "keywords": ["Top Tier", "10%", "Scopus"],
            },
            {
                "query": "Berapa insentif untuk artikel Top Tier dengan co-author luar negeri?",
                "keywords": ["Top Tier", "co-author luar negeri", "Rp35.000.000"],
            },
            {
                "query": "Apa periode kontrak yang digunakan untuk kelayakan artikel pada insentif Top Tiers EQUITY WCU 2026?",
                "keywords": ["periode kontrak", "26 Agustus 2025", "26 Agustus 2026"],
            },
            {
                "query": "Berapa besaran insentif untuk artikel Q1 Percentile minimal 90% tanpa co-authorship LN pada program Top Tier EQUITY WCU?",
                "keywords": ["Q1 Percentile", "90", "33.000.000", "Top Tier"],
            },
            {
                "query": "Apa acknowledgement yang wajib dicantumkan untuk artikel yang didanai sepenuhnya oleh skema EQUITY?",
                "keywords": ["acknowledgement", "LPDP", "EQUITY Program", "Contract No"],
            },
        ],
    },
    {
        "file": "PD_DRPM_ITS_021_Panduan-Fast-D_Beasiswa-Unggulan_GES.pdf",
        "correct_queries": [
            {
                "query": "Berapa dana penelitian maksimal per proposal pada panduan Fast-D, Beasiswa Unggulan, dan GES 2026?",
                "keywords": ["Rp50.000.000", "Fast-D", "Beasiswa Unggulan", "GES"],
            },
            {
                "query": "Berapa jumlah publikasi Scopus minimum yang ditargetkan pada skema FAST-D?",
                "keywords": ["FAST-D", "4", "publikasi Scopus"],
            },
            {
                "query": "Paket luaran minimum apa yang ditetapkan untuk skema Beasiswa Unggulan?",
                "keywords": ["Beasiswa Unggulan", "3", "Q1", "Q2", "Q3"],
            },
            {
                "query": "Kapan jadwal penerimaan proposal program FAST-D, Beasiswa Unggulan, dan GES 2026?",
                "keywords": ["Penerimaan Proposal", "13", "20 Februari", "FAST-D", "GES"],
            },
            {
                "query": "Penerbit apa saja yang tidak termasuk sebagai publikasi internasional pada panduan FAST-D, Beasiswa Unggulan, dan GES?",
                "keywords": ["MDPI", "Frontiers", "Hindawi Publisher"],
            },
        ],
    },
    {
        "file": "Panduan-Kopra-by-Mandiri-Bill-Payment-ID.pdf",
        "correct_queries": [
            {
                "query": "Template file upload apa saja yang tersedia untuk bulk bill payment di Kopra by Mandiri?",
                "keywords": ["Consolidated", "Separated", "upload template"],
            },
            {
                "query": "Berapa jumlah sesi waktu transaksi manual pada fitur bill payment Kopra?",
                "keywords": ["bill payment", "7 sesi", "manual"],
            },
            {
                "query": "Setelah transaksi bill payment berhasil dikirim, ke menu mana transaksi tersebut masuk untuk proses persetujuan?",
                "keywords": ["pending task", "approval", "bill payment"],
            },
            {
                "query": "User apa yang digunakan untuk login pada fitur Bill Payment Kopra by Mandiri?",
                "keywords": ["User Maker", "koprabymandiri.com", "Bill Payment"],
            },
            {
                "query": "Field apa saja yang muncul pada overlay Manual Input Bill Payment?",
                "keywords": ["Source of Fund", "Pay Bill Again", "Recommended Choice"],
            },
        ],
    },
    {
        "file": "Panduan-Kopra-by-Mandiri-Reports-Cek-Saldo.pdf",
        "correct_queries": [
            {
                "query": "Opsi jadwal apa saja yang tersedia pada fitur Auto Report di Kopra by Mandiri?",
                "keywords": ["Auto Report", "One Time", "Daily", "Weekly", "Monthly"],
            },
            {
                "query": "Berapa jumlah sesi waktu report pada fitur Auto Report Kopra?",
                "keywords": ["Auto Report", "5 sesi", "report time"],
            },
            {
                "query": "Jenis account statement apa saja yang bisa dipilih pada laporan cek saldo Kopra?",
                "keywords": ["account statement", "Standard", "Advance"],
            },
            {
                "query": "Action apa saja yang tersedia pada laman Account Balance Preview di Kopra Reports?",
                "keywords": ["Copy Table", "Download", "Print", "Account Balance Preview"],
            },
            {
                "query": "Format file apa saja yang dapat dipilih saat mengunduh report di Kopra Reports?",
                "keywords": ["Download", "PDF", "XLS"],
            },
        ],
    },
    {
        "file": "Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf",
        "correct_queries": [
            {
                "query": "Apa format email default mahasiswa ITS menurut panduan mahasiswa baru DPTSI 2025?",
                "keywords": ["email", "nrp@student.its.ac.id", "ITS"],
            },
            {
                "query": "Berapa panjang minimum password akun myITS?",
                "keywords": ["password", "8 karakter", "myITS"],
            },
            {
                "query": "Jaringan internet apa saja yang disediakan ITS untuk mahasiswa di kampus?",
                "keywords": ["myITS-WiFi", "Eduroam", "internet"],
            },
            {
                "query": "Versi minimal OpenVPN Connect Client apa yang diperlukan untuk menggunakan myITS VPN?",
                "keywords": ["OpenVPN Connect Client", "minimal versi 3.4.2", "myITS VPN"],
            },
            {
                "query": "Apa tautan panduan aktivasi MFA untuk mahasiswa baru ITS?",
                "keywords": ["MFA", "its.id/PanduanAktivasiMFA", "Multi-Factor Authentication"],
            },
        ],
    },
    {
        "file": "Panduan-PMKI_ITS_2026-1.pdf",
        "correct_queries": [
            {
                "query": "Berapa jumlah PTNBH yang terlibat dalam PMKI 2026?",
                "keywords": ["PMKI", "24 PTNBH"],
            },
            {
                "query": "Berapa dana maksimal bagi ITS jika bertindak sebagai host pada PMKI 2026?",
                "keywords": ["ITS", "host", "Rp75.000.000"],
            },
            {
                "query": "Di universitas mana penandatanganan kontrak PMKI 2026 dilaksanakan?",
                "keywords": ["penandatanganan kontrak", "Universitas Sriwijaya"],
            },
            {
                "query": "Kapan periode penerimaan proposal PMKI 2026?",
                "keywords": ["Penerimaan Proposal", "13 Februari", "7 Maret 2026", "PMKI"],
            },
            {
                "query": "Berapa dana maksimal bagi ITS jika bertindak sebagai mitra pada PMKI 2026?",
                "keywords": ["Mitra", "Rp. 50.000.000", "PMKI"],
            },
        ],
    },
    {
        "file": "Panduan-Penelitian-Front-Tiers-Dana-Heti-1.pdf",
        "correct_queries": [
            {
                "query": "Skema hibah apa saja yang tersedia pada program Frontiers Dana HETI?",
                "keywords": ["Frontiers Profesor", "Frontiers Doktor Baru", "skema"],
            },
            {
                "query": "Berapa dana maksimal per proposal pada program Frontiers Dana HETI?",
                "keywords": ["Frontiers", "Rp200.000.000", "proposal"],
            },
            {
                "query": "Sampai kapan artikel wajib Frontiers harus sudah accepted?",
                "keywords": ["accepted", "Desember 2027", "Q1"],
            },
            {
                "query": "Berapa lama maksimal setelah graduation untuk pengusul Frontiers Doktor Baru?",
                "keywords": ["Frontiers Doktor Baru", "Maksimal 2 tahun", "graduation"],
            },
            {
                "query": "Kapan jadwal penerimaan proposal Program Frontiers ADB HETI 2026?",
                "keywords": ["Penerimaan Proposal", "06 Februari", "18 Februari 2026", "Frontiers ADB HETI"],
            },
        ],
    },
    {
        "file": "Panduan-RKI_ITS_2026-.pdf",
        "correct_queries": [
            {
                "query": "Apa perbedaan utama antara RKI Skema A dan Skema C?",
                "keywords": ["Skema A", "Skema C", "Top 100 QS WUR"],
            },
            {
                "query": "Berapa dana yang dapat diajukan host pada RKI Skema C?",
                "keywords": ["host", "Skema C", "Rp325.000.000"],
            },
            {
                "query": "Luaran publikasi minimum apa yang wajib dihasilkan host ITS pada program RKI?",
                "keywords": ["host", "Scopus Q1", "publikasi", "ITS"],
            },
            {
                "query": "Kapan jadwal penerimaan proposal RKI 2026?",
                "keywords": ["Penerimaan Proposal", "13 Februari", "7 Maret 2026", "RKI 2026"],
            },
            {
                "query": "Di universitas mana Monitoring dan Evaluasi Laporan ke-2 RKI 2026 dilaksanakan?",
                "keywords": ["Monitoring dan Evaluasi Laporan ke-2", "Universitas Negeri Semarang", "2", "4 Desember 2026"],
            },
        ],
    },
    {
        "file": "Post-Doctoral-2026-1-1.pdf",
        "correct_queries": [
            {
                "query": "Berapa lama maksimal sejak lulus S3 agar kandidat masih memenuhi syarat program Post-Doctoral 2026?",
                "keywords": ["S3", "5 tahun", "Post-Doctoral"],
            },
            {
                "query": "Dua luaran publikasi apa yang wajib dihasilkan peserta Post-Doctoral 2026?",
                "keywords": ["Scopus Q1 90%", "review Q2", "luaran"],
            },
            {
                "query": "Berapa jumlah proposal hibah yang wajib disiapkan peserta Post-Doctoral?",
                "keywords": ["2", "proposal hibah", "grant proposal"],
            },
            {
                "query": "Berapa bantuan akomodasi per bulan dan dana penelitian pada program Post Doctoral ADB HETI 2026?",
                "keywords": ["Bantuan Akomodasi", "Dana Penelitian", "Rp. 100.000.000"],
            },
            {
                "query": "Apa kriteria publikasi minimum kandidat Post Doctoral Fellow?",
                "keywords": ["minimal 1", "Scopus Q1", "penulis pertama", "corresponding author"],
            },
        ],
    },
    {
        "file": "PD-DRPM-ITS-0025-Panduan-Bantuan-Biaya-APC-Dana-EQUITY-WCU-Tahun-2026-1.pdf",
        "correct_queries": [
            {
                "query": "Berapa nilai maksimal bantuan biaya APC Dana EQUITY WCU tahun 2026?",
                "keywords": ["APC", "EQUITY WCU", "Rp40.000.000"],
            },
            {
                "query": "Status artikel apa yang disyaratkan untuk pengajuan bantuan APC EQUITY WCU?",
                "keywords": ["accepted", "status artikel", "EQUITY WCU"],
            },
            {
                "query": "Dokumen apa saja yang harus dilampirkan untuk klaim bantuan APC EQUITY WCU?",
                "keywords": ["acceptance email", "invoice", "payment proof", "open access link"],
            },
            {
                "query": "Apa tautan pengajuan bantuan APC Dana EQUITY WCU 2026?",
                "keywords": ["https://its.id/BantuanAPC", "DanaEQUITY2026", "dokumen pendukung"],
            },
            {
                "query": "Apa periode kontrak EQUITY untuk artikel bantuan APC high percentile journal?",
                "keywords": ["Periode Kontrak EQUITY", "26 Agustus 2025", "20 Juli 2026"],
            },
        ],
    },
]


ANSWERABLE_QUERIES: list[str] = []
QUERY_METADATA: dict[str, dict[str, Any]] = {}

for document_query_set in DOCUMENT_QUERY_SETS:
    for query_info in document_query_set["correct_queries"]:
        query_text = query_info["query"]
        ANSWERABLE_QUERIES.append(query_text)
        QUERY_METADATA[query_text] = {
            "query": query_text,
            "answerable": True,
            "query_type": "answerable",
            "source_file": document_query_set["file"],
            "keywords": query_info["keywords"],
        }
