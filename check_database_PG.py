import os
from dotenv import load_dotenv
import psycopg
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")

# Connect directly to PostgreSQL
conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

print("=== DATABASE DIAGNOSTIC ===\n")

try:
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Check if pgvector extension exists
            print("1. Checking pgvector extension...")
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cur.fetchone()
            if result:
                print(f"   ✅ pgvector extension found: {result[0]}")
            else:
                print(f"   ❌ pgvector extension NOT found!")
            
            # Check collections
            print("\n2. Checking collections...")
            cur.execute("SELECT name, uuid, cmetadata FROM langchain_pg_collection;")
            collections = cur.fetchall()
            print(f"   Total collections: {len(collections)}")
            for coll in collections:
                print(f"   - {coll[0]} (UUID: {coll[1]})")
            
            # Check embeddings count
            print("\n3. Checking embeddings...")
            cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding;")
            count = cur.fetchone()[0]
            print(f"   Total embeddings: {count}")
            
            if count > 0:
                # Check sample embedding
                print("\n4. Sample embedding data...")
                cur.execute("""
                    SELECT 
                        e.document, 
                        e.cmetadata,
                        c.name as collection_name,
                        array_length(e.embedding, 1) as embedding_dim
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    LIMIT 3;
                """)
                samples = cur.fetchall()
                for i, sample in enumerate(samples, 1):
                    print(f"\n   Sample {i}:")
                    print(f"   Collection: {sample[2]}")
                    print(f"   Embedding dimension: {sample[3]}")
                    print(f"   Document preview: {sample[0][:200]}...")
                    print(f"   Metadata: {sample[1]}")
            else:
                print("   ❌ No embeddings found in database!")
            
            # Check for our specific collection
            print(f"\n5. Checking specific collection '{COLLECTION_NAME}'...")
            cur.execute("""
                SELECT COUNT(*) 
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s;
            """, (COLLECTION_NAME,))
            specific_count = cur.fetchone()[0]
            print(f"   Embeddings in '{COLLECTION_NAME}': {specific_count}")
            
except Exception as e:
    print(f"❌ Database connection error: {e}")

# Test embedding generation
print("\n6. Testing embedding generation...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
try:
    test_embedding = embeddings.embed_query("test")
    print(f"   ✅ Embedding generated successfully")
    print(f"   Dimension: {len(test_embedding)}")
    print(f"   First 5 values: {test_embedding[:5]}")
except Exception as e:
    print(f"   ❌ Embedding generation failed: {e}")