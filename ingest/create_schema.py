import os
import sys
import time
from dotenv import load_dotenv
import psycopg


SCHEMA_SQL = """
-- 1. Sources Register
CREATE TABLE IF NOT EXISTS sources (
    id VARCHAR(50) PRIMARY KEY,
    area VARCHAR(100),
    page VARCHAR(255),
    url TEXT,
    owner VARCHAR(255),
    what_to_extract TEXT,
    format VARCHAR(100),
    tag VARCHAR(100),
    refresh VARCHAR(100),
    read_status VARCHAR(100),
    milestone VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Key Facts
CREATE TABLE IF NOT EXISTS key_facts (
    id VARCHAR(50) PRIMARY KEY,
    topic VARCHAR(255),
    tag VARCHAR(100),
    fact TEXT NOT NULL,
    valid_for VARCHAR(100),
    source_id VARCHAR(100),
    source_url TEXT,
    verify_note TEXT,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Flags and Gaps
CREATE TABLE IF NOT EXISTS flags_and_gaps (
    id SERIAL PRIMARY KEY,
    issue VARCHAR(255),
    what_i_found TEXT,
    what_to_do TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Documents (Ingested pages & articles)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(50) REFERENCES sources(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    url TEXT,
    content TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Chunks (For Vector Search & RAG)
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_key_facts_source_id ON key_facts(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in environment.", file=sys.stderr)
        sys.exit(1)

    # Retry connection
    conn = None
    for attempt in range(1, 3):
        try:
            conn = psycopg.connect(db_url)
            break
        except Exception as e:
            if attempt == 2:
                print(f"Failed to connect: {e}", file=sys.stderr)
                sys.exit(1)
            print("Retrying connection in 3 seconds...")
            time.sleep(3)

    try:
        with conn.cursor() as cur:
            print("Creating database schema...")
            cur.execute(SCHEMA_SQL)
            conn.commit()

            # Verify tables created
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            )
            tables = [row[0] for row in cur.fetchall()]
            print("SUCCESS: Tables defined in public schema:")
            for t in tables:
                print(f"  - {t}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

