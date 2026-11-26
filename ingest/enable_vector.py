import os
import sys
import time
from dotenv import load_dotenv
import psycopg


def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in environment.", file=sys.stderr)
        sys.exit(1)

    # Retry logic
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
            print("Enabling 'vector' extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            ext = cur.fetchone()
            if ext:
                print(f"SUCCESS: Extension '{ext[0]}' is active (version: {ext[1]}).")
            else:
                print("WARNING: Extension 'vector' was not found after creation.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

