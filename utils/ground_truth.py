"""
Ground Truth Dataset for Retrieval Quality Evaluation
Maps queries to their expected relevant documents and keywords
"""

# Ground truth for Precision/Recall calculation
# Each query maps to:
# - relevant_sources: PDF files that should contain the answer
# - keywords: Key terms that must appear in relevant results
GROUND_TRUTH = {
    # ========== INDONESIAN QUERIES (Document 1: Panduan-Mahasiswa-Baru-DPTSI-2025) ==========
    "Apa itu myITS Portal dan apa fungsinya?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["myITS", "portal", "fungsi", "layanan"]
    },
    "Bagaimana langkah-langkah login pertama kali ke myITS Portal?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["login", "myITS", "portal", "pertama"]
    },
    "Apa format default email mahasiswa baru ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["email", "format", "mahasiswa", "its.ac.id"]
    },
    "Bagaimana cara melakukan aktivasi Multi-Factor Authentication (MFA)?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["MFA", "Multi-Factor", "Authentication", "aktivasi"]
    },
    "Apa syarat pembuatan password baru untuk akun myITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["password", "syarat", "myITS", "akun"]
    },
    "Bagaimana cara verifikasi email alternatif dan nomor ponsel di myITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["verifikasi", "email", "ponsel", "myITS"]
    },
    "Apa fungsi dari aplikasi SIAKAD (Sistem Informasi Akademik)?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["SIAKAD", "akademik", "sistem", "informasi"]
    },
    "Bagaimana cara orang tua mahasiswa mengakses myITS Wali?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["myITS Wali", "orang tua", "akses", "wali"]
    },
    "Apa itu myITS Classroom dan bagaimana cara mengaksesnya?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["myITS Classroom", "classroom", "akses", "kuliah"]
    },
    "Bagaimana prosedur presensi kuliah menggunakan myITS Presensi?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["presensi", "myITS Presensi", "kuliah", "absensi"]
    },
    "Apa saja layanan internet yang tersedia di lingkungan kampus ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["internet", "WiFi", "kampus", "jaringan"]
    },
    "Bagaimana cara menghubungkan perangkat ke jaringan Eduroam?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Eduroam", "jaringan", "WiFi", "koneksi"]
    },
    "Aplikasi apa yang wajib diunduh untuk keperluan MFA?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["MFA", "aplikasi", "Microsoft Authenticator", "unduh"]
    },
    "Bagaimana cara menginstal Office 365 versi desktop untuk mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Office 365", "instal", "desktop", "Microsoft"]
    },
    "Langkah-langkah aktivasi lisensi Office 365 menggunakan akun ITS.": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Office 365", "lisensi", "aktivasi", "ITS"]
    },
    "Bagaimana cara mengakses aplikasi Zoom berlisensi dari ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Zoom", "lisensi", "akses", "ITS"]
    },
    "Apa kegunaan layanan VPN ITS bagi mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["VPN", "ITS", "layanan", "akses"]
    },
    "Bagaimana cara setting OpenVPN Connect Client untuk VPN ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["OpenVPN", "VPN", "setting", "client"]
    },
    "Apa saja fasilitas yang disediakan oleh Perpustakaan ITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["perpustakaan", "fasilitas", "ITS", "layanan"]
    },
    "Bagaimana cara mahasiswa melakukan peminjaman buku secara online (OBOR)?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["OBOR", "peminjaman", "buku", "online"]
    },
    "Apa peran Direktorat Riset dan Pengabdian kepada Masyarakat (DRPM) bagi mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["DRPM", "riset", "penelitian", "pengabdian"]
    },
    "Apa itu SIMPel ITS dan kegunaannya untuk penelitian?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["SIMPel", "penelitian", "riset", "ITS"]
    },
    "Di mana mahasiswa bisa mendapatkan informasi layanan untuk penyandang disabilitas?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["disabilitas", "layanan", "informasi", "mahasiswa"]
    },
    "Apa saja provider seluler yang tidak bisa digunakan untuk verifikasi nomor ponsel di myITS?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["provider", "seluler", "verifikasi", "ponsel"]
    },
    "Ke mana mahasiswa harus menghubungi jika mengalami kendala layanan IT (Service Desk)?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Service Desk", "helpdesk", "kendala", "layanan"]
    },
    "Bagaimana cara mengubah password melalui myITS Account?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["password", "ubah", "myITS Account", "ganti"]
    },
    "Apa password default yang digunakan saat login pertama kali?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["password", "default", "login", "pertama"]
    },
    "Bagaimana cara mengakses email ITS menggunakan Microsoft Outlook?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["email", "Outlook", "Microsoft", "ITS"]
    },
    "Apa saja aplikasi yang tergabung dalam layanan Office 365 untuk mahasiswa?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["Office 365", "aplikasi", "Microsoft", "layanan"]
    },
    "Bagaimana cara mengecek status verifikasi email alternatif?": {
        "relevant_sources": ["Panduan-Mahasiswa-Baru-DPTSI-2025_revised-1.pdf"],
        "keywords": ["verifikasi", "email", "status", "alternatif"]
    },

    # ========== ENGLISH QUERIES (Document 2: General-Guidebook-for-International-Students) ==========
    "What items are on the arrival checklist for international students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["checklist", "arrival", "international", "students"]
    },
    "How do I register my phone's IMEI in Indonesia?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["IMEI", "register", "phone", "Indonesia"]
    },
    "What is the tax calculation for registering a phone worth more than $500?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["tax", "phone", "500", "IMEI"]
    },
    "What documents are required to apply for a Bachelor Degree (E30B) Visa?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["E30B", "visa", "Bachelor", "documents"]
    },
    "How much is the ITS administration fee for international students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["fee", "administration", "international", "ITS"]
    },
    "What is the procedure for obtaining a Police Notification Report (STM)?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["STM", "police", "notification", "report"]
    },
    "How do I get a Proof of Residency (SKTT) and what is the deadline?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["SKTT", "residency", "proof", "deadline"]
    },
    "What are the recommended mobile apps for finding accommodation in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["accommodation", "apps", "Surabaya", "mobile"]
    },
    "How to register for a local bank account at Mandiri?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["bank", "Mandiri", "account", "register"]
    },
    "What are the emergency hotline numbers in Surabaya?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["emergency", "hotline", "Surabaya", "number"]
    },
    "What is the dress code for attending classes at ITS?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["dress", "code", "class", "ITS"]
    },
    "Which hospitals are recommended near the ITS campus?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["hospital", "recommended", "ITS", "campus"]
    },
    "What transport options are available from Juanda International Airport to ITS?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Juanda", "airport", "transport", "ITS"]
    },
    "How to use the Suroboyo Bus and what is the cost for students?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Suroboyo", "bus", "cost", "students"]
    },
    "What are the \"Dos and Don'ts\" regarding etiquette in Indonesia?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["etiquette", "dos", "don'ts", "Indonesia"]
    },
    "Where is the Global Engagement Office located?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Global Engagement", "office", "location"]
    },
    "What is the process for extending a tourist IMEI registration?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["IMEI", "tourist", "extend", "registration"]
    },
    "List of faculties available under the SCIENTICS cluster.": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["SCIENTICS", "faculty", "faculties", "cluster"]
    },
    "What is the \"O-Week\" and why is it mandatory?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["O-Week", "orientation", "mandatory"]
    },
    "How to activate the \"Livin by Mandiri\" mobile banking app?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Livin", "Mandiri", "banking", "activate"]
    },
    "What are the estimated living costs for accommodation near ITS?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["living", "cost", "accommodation", "ITS"]
    },
    "Where can students find traditional markets near the campus?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["market", "traditional", "campus", "students"]
    },
    "What steps should be taken during an earthquake according to the disaster mitigation guide?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["earthquake", "disaster", "mitigation", "steps"]
    },
    "What is the meaning of the Indonesian slang \"Mager\"?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Mager", "slang", "Indonesian", "meaning"]
    },
    "How can international students join Student Activity Units (UKM)?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["UKM", "student", "activity", "international"]
    },
    "What are the requirements for an Internship Visa (C22A)?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["C22A", "internship", "visa", "requirements"]
    },
    "Locations of places of worship (mosques and churches) around ITS.": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["mosque", "church", "worship", "ITS"]
    },
    "What facilities are available at the ITS Medical Centre?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["medical", "centre", "facilities", "ITS"]
    },
    "How to use Gojek or Grab for local transportation?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["Gojek", "Grab", "transportation", "local"]
    },
    "What is the penalty for failing to obtain Proof of Residency within 14 days?": {
        "relevant_sources": ["General-Guidebook-for-International-Students_July-2024.pdf"],
        "keywords": ["penalty", "residency", "14 days", "fine"]
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
