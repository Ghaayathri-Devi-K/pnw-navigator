import argparse
import glob
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ingest.db import get_connection
except ImportError:
    from db import get_connection



def load_data_to_neon(
    parsed_dir: str = "data/parsed",
    embedded_chunks_file: str = "data/embedded_chunks.json",
    dry_run: bool = False,
    truncate: bool = False,
    limit_docs: int = None,
):
    if not os.path.exists(embedded_chunks_file):
        print(f"ERROR: '{embedded_chunks_file}' not found. Run stage 3 (ingest/embed.py) first.", file=sys.stderr)
        sys.exit(1)

    parsed_files = sorted(glob.glob(os.path.join(parsed_dir, "*.json")))
    if not parsed_files:
        print(f"ERROR: No parsed files found in '{parsed_dir}'.", file=sys.stderr)
        sys.exit(1)

    if limit_docs and limit_docs > 0:
        parsed_files = parsed_files[:limit_docs]
        allowed_source_ids = set()
        for pf in parsed_files:
            with open(pf, "r", encoding="utf-8") as f:
                d = json.load(f)
                allowed_source_ids.add(d["id"])
    else:
        allowed_source_ids = None

    with open(embedded_chunks_file, "r", encoding="utf-8") as f:
        all_embedded_chunks = json.load(f)

    if allowed_source_ids is not None:
        chunks_to_load = [c for c in all_embedded_chunks if c["source_id"] in allowed_source_ids]
    else:
        chunks_to_load = all_embedded_chunks

    print("=" * 60)
    mode_label = "DRY-RUN MODE (Transaction will be rolled back)" if dry_run else "LIVE MODE (Changes will be committed)"
    print(f"Starting Neon Database Load in {mode_label}")
    print(f"  Target Documents: {len(parsed_files)}")
    print(f"  Target Chunks:    {len(chunks_to_load)}")
    print("=" * 60)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if truncate and not dry_run:
                print("Clearing existing records from 'chunks' and 'documents'...")
                cur.execute("TRUNCATE TABLE chunks, documents RESTART IDENTITY CASCADE;")

            # 1. Insert Documents
            print(f"\nProcessing {len(parsed_files)} document(s)...")
            source_to_doc_id = {}
            doc_insert_count = 0

            for pf in parsed_files:
                with open(pf, "r", encoding="utf-8") as f:
                    pdoc = json.load(f)

                source_id = pdoc["id"]
                title = pdoc.get("title", "")
                url = pdoc.get("url", "")
                content = pdoc.get("text", "")
                doc_meta = {
                    "area": pdoc.get("area"),
                    "tag": pdoc.get("tag"),
                    "format": pdoc.get("format"),
                    "valid_for_term": pdoc.get("valid_for_term"),
                    "char_count": pdoc.get("char_count"),
                    "section_count": pdoc.get("section_count"),
                }

                cur.execute(
                    """
                    INSERT INTO documents (source_id, title, url, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (source_id, title, url, content, json.dumps(doc_meta)),
                )
                db_doc_id = cur.fetchone()[0]
                source_to_doc_id[source_id] = db_doc_id
                doc_insert_count += 1

            print(f"  -> Prepared {doc_insert_count} document inserts.")

            # 2. Insert Chunks with 384-d vector embeddings
            print(f"\nProcessing {len(chunks_to_load)} chunk(s)...")
            chunk_insert_count = 0

            for chunk in chunks_to_load:
                sid = chunk["source_id"]
                db_doc_id = source_to_doc_id.get(sid)
                if not db_doc_id:
                    continue

                chunk_index = chunk["chunk_index"]
                chunk_content = chunk["content"]
                embedding = chunk["embedding"]
                chunk_meta = {
                    "chunk_id": chunk.get("id"),
                    "source_id": sid,
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "area": chunk.get("area"),
                    "tag": chunk.get("tag"),
                    "valid_for_term": chunk.get("valid_for_term"),
                    "section_heading": chunk.get("section_heading"),
                    "char_count": chunk.get("char_count"),
                    "token_estimate": chunk.get("token_estimate"),
                }

                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s);
                    """,
                    (
                        db_doc_id,
                        chunk_index,
                        chunk_content,
                        json.dumps(embedding),
                        json.dumps(chunk_meta),
                    ),
                )
                chunk_insert_count += 1

            print(f"  -> Prepared {chunk_insert_count} chunk inserts with embeddings.")

            if dry_run:
                print("\n[DRY-RUN] Executing transaction rollback...")
                conn.rollback()
                print("SUCCESS: Dry-run passed with zero database mutations.")
                print(f"  Documents planned: {doc_insert_count}")
                print(f"  Chunks planned:    {chunk_insert_count}")
            else:
                print("\n[LIVE] Committing transaction to Neon Postgres...")
                conn.commit()
                print("SUCCESS: Live database load committed.")

            # Final verify counts in database
            cur.execute("SELECT COUNT(*) FROM documents;")
            actual_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks;")
            actual_chunks = cur.fetchone()[0]
            print(f"\nCurrent Database State: {actual_docs} documents, {actual_chunks} chunks.")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load documents and embedded chunks into Neon Postgres")
    parser.add_argument("--dry-run", action="store_true", help="Test execution without committing changes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to load")
    parser.add_argument("--truncate", action="store_true", help="Truncate documents and chunks before loading")
    parser.add_argument("--parsed-dir", type=str, default="data/parsed", help="Path to parsed JSON directory")
    parser.add_argument("--chunks-file", type=str, default="data/embedded_chunks.json", help="Path to embedded chunks JSON")
    args = parser.parse_args()

    load_data_to_neon(
        parsed_dir=args.parsed_dir,
        embedded_chunks_file=args.chunks_file,
        dry_run=args.dry_run,
        truncate=args.truncate,
        limit_docs=args.limit,
    )


if __name__ == "__main__":
    main()
