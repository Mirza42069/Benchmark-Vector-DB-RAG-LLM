import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")

def setup_database():
    """Initialize PostgreSQL database with pgvector extension"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("✅ Connected to PostgreSQL")
        
        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ Created pgvector extension")
        
        # Verify extension
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        if result:
            print(f"✅ pgvector extension verified: {result[0]}")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 Database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error setting up database: {str(e)}")
        raise

if __name__ == "__main__":
    setup_database()