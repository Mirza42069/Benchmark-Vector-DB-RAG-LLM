import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Connect to database
conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

print("=== TESTING RETRIEVAL ===\n")

test_queries = [
    "Explain how to register a phone in Indonesian Airport?",
    "What documents do I need to bring when arriving in Surabaya?",
    "What is the tuition fee for undergraduate programs?",  # Not in document
]

with psycopg.connect(conn_str) as conn:
    register_vector(conn)
    
    # Get collection ID
    with conn.cursor() as cur:
        cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (COLLECTION_NAME,))
        collection_id = cur.fetchone()[0]
        print(f"✅ Using collection: {COLLECTION_NAME} ({collection_id})\n")
    
    for query in test_queries:
        print(f"🔍 Query: {query}")
        print("-" * 70)
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search for similar documents
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    document, 
                    cmetadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM langchain_pg_embedding
                WHERE collection_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
            """, (query_embedding, str(collection_id), query_embedding))
            
            results = cur.fetchall()
            
            print(f"Results found: {len(results)}\n")
            
            if results:
                for i, (doc, metadata, similarity) in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  Similarity: {similarity:.4f}")
                    print(f"  Content: {doc[:250]}...")
                    print(f"  Metadata: {metadata}")
                    print()
            else:
                print("❌ No documents found\n")
        
        print("=" * 70 + "\n")