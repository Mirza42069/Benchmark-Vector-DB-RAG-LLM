import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import psycopg
from pgvector.psycopg import register_vector
import uuid

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "its_guidebook")

print("=== MANUAL INGESTION TO POSTGRESQL ===\n")

# Load documents
print("1. Loading PDF documents...")
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()
print(f"   ✅ Loaded {len(raw_documents)} documents")

# Split documents
print("\n2. Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(raw_documents)
print(f"   ✅ Created {len(documents)} chunks")
print(f"   Sample chunk: {documents[0].page_content[:200]}...")

# Initialize embeddings
print("\n3. Initializing embeddings...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Test embedding
print("\n4. Testing embedding generation...")
test_embedding = embeddings.embed_query("test query")
print(f"   ✅ Embedding dimension: {len(test_embedding)}")

# Connect to database
print("\n5. Connecting to PostgreSQL...")
conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

with psycopg.connect(conn_str) as conn:
    register_vector(conn)
    
    with conn.cursor() as cur:
        # Ensure uuid-ossp extension is enabled
        print("\n6. Enabling uuid-ossp extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        conn.commit()
        
        # Get or create collection
        print(f"\n7. Getting/creating collection '{COLLECTION_NAME}'...")
        cur.execute("""
            SELECT uuid FROM langchain_pg_collection WHERE name = %s;
        """, (COLLECTION_NAME,))
        result = cur.fetchone()
        
        if result:
            collection_id = result[0]
            print(f"   ✅ Found collection with ID: {collection_id}")
            
            # Clear existing data
            print("\n8. Clearing old embeddings...")
            cur.execute("""
                DELETE FROM langchain_pg_embedding WHERE collection_id = %s;
            """, (collection_id,))
            conn.commit()
            print(f"   ✅ Cleared old data")
        else:
            # Create collection with explicit UUID
            print(f"\n8. Creating new collection '{COLLECTION_NAME}'...")
            collection_id = uuid.uuid4()
            cur.execute("""
                INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
                VALUES (%s, %s, %s);
            """, (str(collection_id), COLLECTION_NAME, '{}'))
            conn.commit()
            print(f"   ✅ Collection created with ID: {collection_id}")
        
        # Drop and recreate embedding table with correct structure
        print("\n9. Recreating embedding table...")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        cur.execute(f"""
            CREATE TABLE langchain_pg_embedding (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                embedding VECTOR({len(test_embedding)}),
                document TEXT,
                cmetadata JSONB
            );
        """)
        
        # Create index
        cur.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_collection_idx 
            ON langchain_pg_embedding(collection_id);
        """)
        conn.commit()
        print("   ✅ Table recreated")
        
        # Insert documents with embeddings
        print(f"\n10. Inserting {len(documents)} documents...")
        
        success_count = 0
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"   Processing document {i+1}/{len(documents)}...")
            
            try:
                # Generate embedding
                embedding = embeddings.embed_query(doc.page_content)
                
                # Generate UUID for this embedding
                embedding_id = uuid.uuid4()
                
                # Insert with explicit ID
                cur.execute("""
                    INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata)
                    VALUES (%s, %s, %s, %s, %s);
                """, (
                    str(embedding_id),
                    str(collection_id),
                    embedding,
                    doc.page_content,
                    psycopg.types.json.Jsonb(doc.metadata if doc.metadata else {})
                ))
                
                success_count += 1
                
                # Commit every 20 documents
                if (i + 1) % 20 == 0:
                    conn.commit()
                    print(f"   ✅ Committed {i+1} documents")
                    
            except Exception as e:
                print(f"   ❌ Error on document {i}: {e}")
                continue
        
        # Final commit
        conn.commit()
        print(f"   ✅ Successfully inserted {success_count}/{len(documents)} documents!")
        
        # Verify
        print("\n11. Verifying insertion...")
        cur.execute("""
            SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s;
        """, (str(collection_id),))
        count = cur.fetchone()[0]
        print(f"   ✅ Verified: {count} embeddings in database")
        
        # Show sample
        if count > 0:
            print("\n12. Sample document check...")
            cur.execute("""
                SELECT document, cmetadata FROM langchain_pg_embedding 
                WHERE collection_id = %s LIMIT 1;
            """, (str(collection_id),))
            sample = cur.fetchone()
            print(f"   Sample text: {sample[0][:200]}...")
            print(f"   Sample metadata: {sample[1]}")
            
            # Test similarity search
            print("\n13. Testing similarity search...")
            test_query = "What documents do I need?"
            test_query_embedding = embeddings.embed_query(test_query)
            
            cur.execute("""
                SELECT document, 1 - (embedding <=> %s::vector) as similarity
                FROM langchain_pg_embedding
                WHERE collection_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT 3;
            """, (test_query_embedding, str(collection_id), test_query_embedding))
            
            results = cur.fetchall()
            print(f"   Found {len(results)} similar documents:")
            for idx, (doc, sim) in enumerate(results, 1):
                print(f"   {idx}. Similarity: {sim:.4f} - {doc[:100]}...")

print("\n🎉 Ingestion completed successfully!")