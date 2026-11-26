import os
import sys
import time
from dotenv import load_dotenv
import psycopg


def get_connection(max_retries: int = 3, delay_seconds: int = 2) -> psycopg.Connection:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not set in the environment or .env file.")

    attempt = 0
    while True:
        try:
            return psycopg.connect(db_url)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                # Never print the db_url secret
                raise RuntimeError(f"Database connection failed after {max_retries + 1} attempts ({type(e).__name__}): {e}")
            print(f"Database connection attempt {attempt} failed ({type(e).__name__}). Retrying in {delay_seconds}s (Neon compute waking up)...")
            time.sleep(delay_seconds)

