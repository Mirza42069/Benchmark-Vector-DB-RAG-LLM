"""
Ground Truth Dataset for Retrieval Quality Evaluation
Maps queries to their expected relevant documents and keywords
"""

# Ground truth for Precision/Recall calculation
# Each query maps to:
# - relevant_sources: PDF files that should contain the answer
# - keywords: Key terms that must appear in relevant results
GROUND_TRUTH = {
    # ========== INDONESIAN QUERIES (DPTSI Guide) ==========
    "Apa itu myITS Portal dan bagaimana cara mengaksesnya?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["myITS", "portal", "akses", "login"]
    },
    "Bagaimana cara mengubah password myITS Portal?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["password", "ubah", "ganti", "myITS"]
    },
    "Aplikasi apa saja yang bisa diakses melalui myITS Portal?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["aplikasi", "myITS", "portal", "layanan"]
    },
    "Bagaimana cara mengaktivasi Multi-Factor Authentication (MFA)?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["MFA", "Multi-Factor", "Authentication", "autentikasi"]
    },
    "Apa saja layanan internet yang tersedia di kampus ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["internet", "WiFi", "jaringan", "kampus"]
    },
    "Bagaimana cara mengakses Office 365 untuk mahasiswa ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Office", "365", "Microsoft", "mahasiswa"]
    },
    "Bagaimana cara menyambungkan perangkat ke WiFi ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["WiFi", "ITS", "sambung", "koneksi"]
    },
    "Apa itu myITS SSO dan bagaimana cara menggunakannya?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["SSO", "Single Sign-On", "myITS", "login"]
    },
    "Bagaimana cara mengakses email ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["email", "ITS", "mail", "akses"]
    },
    "Apa saja fitur yang tersedia di Microsoft Teams untuk mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Teams", "Microsoft", "fitur", "mahasiswa"]
    },
    "Bagaimana cara mengunduh Microsoft Office gratis untuk mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Office", "Microsoft", "unduh", "gratis"]
    },
    "Apa itu myITS StudentConnect?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["StudentConnect", "myITS", "mahasiswa"]
    },
    "Bagaimana cara mengakses SIAKAD ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["SIAKAD", "akademik", "akses"]
    },
    "Apa saja layanan yang disediakan DPTSI untuk mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["DPTSI", "layanan", "mahasiswa"]
    },
    "Bagaimana cara reset password akun ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["reset", "password", "akun"]
    },
    "Apa itu ITS Single Sign-On?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Single Sign-On", "SSO", "ITS"]
    },
    "Bagaimana cara mengakses Google Workspace ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Google", "Workspace", "ITS"]
    },
    "Apa saja aplikasi yang terintegrasi dengan myITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["aplikasi", "myITS", "integrasi"]
    },
    "Bagaimana cara mengaktifkan akun mahasiswa baru di ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["akun", "mahasiswa", "baru", "aktivasi"]
    },
    "Apa itu myITS dan apa bedanya dengan SIAKAD?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["myITS", "SIAKAD", "beda", "perbedaan"]
    },
    "Bagaimana cara mendapatkan lisensi software gratis dari ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["lisensi", "software", "gratis"]
    },
    "Apa saja jaringan WiFi yang tersedia di kampus ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["WiFi", "jaringan", "kampus"]
    },
    "Bagaimana cara menghubungi helpdesk DPTSI?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["helpdesk", "DPTSI", "hubungi", "kontak"]
    },
    "Apa itu VPN ITS dan bagaimana cara menggunakannya?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["VPN", "ITS", "akses"]
    },
    "Bagaimana cara mengakses e-learning ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["e-learning", "learning", "ITS"]
    },
    "Apa saja layanan cloud storage yang tersedia untuk mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["cloud", "storage", "penyimpanan"]
    },
    "Bagaimana cara menggunakan OneDrive ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["OneDrive", "ITS", "storage"]
    },
    "Apa itu ITS Repository dan bagaimana cara mengaksesnya?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Repository", "ITS", "akses"]
    },
    "Bagaimana cara mengakses jurnal online dari perpustakaan ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["jurnal", "perpustakaan", "online"]
    },
    "Apa saja panduan keamanan akun yang disarankan DPTSI?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["keamanan", "akun", "DPTSI", "panduan"]
    },
    
    # ========== ENGLISH QUERIES (International Student Guide) ==========
    "What documents do I need to bring when arriving in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["documents", "arrival", "Surabaya", "bring"]
    },
    "How many types of academic VISA are available for international students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["VISA", "academic", "types", "international"]
    },
    "What is the ITS administration fee for international students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["fee", "administration", "international", "ITS"]
    },
    "Where is the ITS Global Engagement Office located?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Global Engagement", "office", "location", "ITS"]
    },
    "How do I register my phone's IMEI for a long stay in Indonesia?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["IMEI", "phone", "register", "Indonesia"]
    },
    "What are the internet package options for Tourist SIM Card?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["SIM Card", "internet", "package", "tourist"]
    },
    "How much does accommodation cost per month near ITS?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["accommodation", "cost", "month", "ITS"]
    },
    "What is the emergency hotline number in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["emergency", "hotline", "Surabaya", "number"]
    },
    "Which banks are available inside ITS campus?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["bank", "ITS", "campus"]
    },
    "What are the nearest hospitals to ITS?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["hospital", "nearest", "ITS"]
    },
    "What is the fine for not having proof of residency within 14 days?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["fine", "residency", "proof", "14 days"]
    },
    "Which transportation apps can I use in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["transportation", "app", "Surabaya"]
    },
    "What faculties are available at ITS?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["faculty", "faculties", "ITS"]
    },
    "How long does the admission evaluation process take?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["admission", "evaluation", "process"]
    },
    "How do I open a bank account in Indonesia?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["bank", "account", "open", "Indonesia"]
    },
    "What are the requirements for extending a study visa?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["visa", "extend", "study", "requirements"]
    },
    "Who should I contact in case of an emergency?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["emergency", "contact"]
    },
    "Where can I find halal food near the campus?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["halal", "food", "campus"]
    },
    "What sports facilities are available for students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["sports", "facilities", "students"]
    },
    "What is the procedure for visa on arrival?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["visa", "arrival", "procedure"]
    },
    "How do I get a KITAS (temporary stay permit)?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["KITAS", "temporary", "stay", "permit"]
    },
    "What are the living costs in Surabaya for students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["living", "cost", "Surabaya", "students"]
    },
    "Where can I exchange foreign currency in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["exchange", "currency", "Surabaya"]
    },
    "What is the process for airport pickup service?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["airport", "pickup", "service"]
    },
    "How do I get Indonesian phone number (SIM card)?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["phone", "SIM card", "Indonesian"]
    },
    "What are the requirements for ITAS application?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["ITAS", "requirements", "application"]
    },
    "Where is the immigration office for visa extension?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["immigration", "office", "visa", "extension"]
    },
    "What public transportation is available in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["public", "transportation", "Surabaya"]
    },
    "How do I register at the local immigration office?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["register", "immigration", "local"]
    },
    "What health insurance options are available for international students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["health", "insurance", "international", "students"]
    }
}


def calculate_retrieval_metrics(retrieved_docs, query, top_k=3):
    """
    Calculate Precision, Recall, and F1-Score for a query
    
    Args:
        retrieved_docs: List of retrieved documents from vector store
        query: The query text
        top_k: Number of documents retrieved
        
    Returns:
        dict with precision, recall, f1_score, and relevance details
    """
    if query not in GROUND_TRUTH:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'error': 'Query not in ground truth'
        }
    
    ground_truth = GROUND_TRUTH[query]
    expected_sources = ground_truth['relevant_sources']
    keywords = ground_truth['keywords']
    
    # Count relevant retrieved documents
    relevant_retrieved = 0
    relevance_details = []
    
    for doc in retrieved_docs:
        source_file = doc.metadata.get('source_file', '')
        content = doc.page_content.lower()
        
        # Check if source matches expected sources
        source_match = any(expected in source_file for expected in expected_sources)
        
        # Check if keywords are present (at least 1 keyword match)
        keyword_matches = sum(1 for kw in keywords if kw.lower() in content)
        keyword_match = keyword_matches >= 1
        
        # Document is relevant if source matches AND has keyword
        is_relevant = source_match and keyword_match
        
        if is_relevant:
            relevant_retrieved += 1
        
        relevance_details.append({
            'source': source_file,
            'source_match': source_match,
            'keyword_matches': keyword_matches,
            'is_relevant': is_relevant
        })
    
    # Calculate metrics
    total_retrieved = len(retrieved_docs)
    total_relevant = len(expected_sources)  # Expected to find at least 1 from each source
    
    # Precision: relevant_retrieved / total_retrieved
    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
    
    # Recall: did we find at least one relevant doc?
    # Since we have 1 relevant source per query, recall = 1 if we found it, 0 otherwise
    recall = 1.0 if relevant_retrieved > 0 else 0.0
    
    # F1-Score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'relevant_retrieved': relevant_retrieved,
        'total_retrieved': total_retrieved,
        'relevance_details': relevance_details
    }
