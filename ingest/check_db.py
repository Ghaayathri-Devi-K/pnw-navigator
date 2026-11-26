import os
import sys
import time
from dotenv import load_dotenv
import psycopg


def connect_with_retry(conn_str: str, max_retries: int = 1, delay_seconds: int = 3):
    attempt = 0
    while True:
        try:
            return psycopg.connect(conn_str)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                # Do not expose the connection string in error messages
                print(
                    f"Connection failed after {max_retries + 1} attempts: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                f"Connection attempt {attempt} failed ({type(e).__name__}). "
                f"Retrying in {delay_seconds} seconds (Neon compute may be waking up)..."
            )
            time.sleep(delay_seconds)


def main():
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set in .env or the environment.", file=sys.stderr)
        sys.exit(1)

    conn = connect_with_retry(database_url, max_retries=1, delay_seconds=3)

    try:
        with conn.cursor() as cur:
            # 1. Postgres server version
            cur.execute("SELECT version();")
            version_row = cur.fetchone()
            version = version_row[0] if version_row else "Unknown"
            print(f"Postgres Server Version:\n  {version}\n")

            # 2. Vector extension check
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            vector_row = cur.fetchone()
            vector_installed = "Yes (Installed)" if vector_row else "No (Not Installed)"
            print(f"Vector Extension Installed: {vector_installed}\n")

            # 3. Tables in the public schema
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
            )
            tables = cur.fetchall()
            print("Tables in public schema:")
            if tables:
                for (table_name,) in tables:
                    print(f"  - {table_name}")
            else:
                print("  (None found / schema is empty)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

