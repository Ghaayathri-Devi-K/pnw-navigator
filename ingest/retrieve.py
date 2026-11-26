"""
ingest/retrieve.py — Cosine-similarity retrieval from Neon Postgres (pgvector).

Usage:
    python ingest/retrieve.py --query "How do I apply for on-campus housing?" --top-k 5
    python ingest/retrieve.py --query "OPT application timeline" --top-k 3
    python ingest/retrieve.py --query "health insurance waiver" --top-k 5 --area Insurance
"""

import argparse
import json
import os
import sys
from typing import Optional

# Ensure project root is in sys.path so ingest.db resolves when run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ingest.db import get_connection
except ImportError:
    from db import get_connection


MODEL_NAME = "all-MiniLM-L6-v2"

_model_cache = {}


def get_embedding_model(model_name: str = MODEL_NAME):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_query(query: str, model_name: str = MODEL_NAME) -> list[float]:
    """Embed a single query string with L2 normalization (unit vector)."""
    import numpy as np

    model = get_embedding_model(model_name)
    vec = model.encode([query], normalize_embeddings=True)[0]
    # Confirm unit norm (should already be 1.0)
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > 1e-4:
        vec = vec / norm
    return vec.tolist()


def retrieve(
    query_vec: list[float],
    top_k: int = 5,
    area_filter: Optional[str] = None,
) -> list[dict]:
    """
    Run cosine similarity search against chunks.embedding using pgvector <=> operator.
    Similarity = 1 - cosine_distance  (range 0..1, higher = more similar).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            vec_str = json.dumps(query_vec)

            if area_filter:
                # Filter by metadata->>'area'
                sql = """
                    SELECT
                        c.id                          AS chunk_db_id,
                        c.content,
                        c.metadata,
                        c.chunk_index,
                        1 - (c.embedding <=> %s::vector) AS similarity
                    FROM chunks c
                    WHERE c.metadata->>'area' ILIKE %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(sql, (vec_str, area_filter, vec_str, top_k))
            else:
                sql = """
                    SELECT
                        c.id                          AS chunk_db_id,
                        c.content,
                        c.metadata,
                        c.chunk_index,
                        1 - (c.embedding <=> %s::vector) AS similarity
                    FROM chunks c
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(sql, (vec_str, vec_str, top_k))

            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        chunk_db_id, content, metadata, chunk_index, similarity = row
        meta = metadata if isinstance(metadata, dict) else {}
        results.append(
            {
                "chunk_db_id": chunk_db_id,
                "chunk_id": meta.get("chunk_id", f"chunk_{chunk_db_id}"),
                "source_id": meta.get("source_id", ""),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "area": meta.get("area", ""),
                "section_heading": meta.get("section_heading", ""),
                "similarity": round(float(similarity), 4),
                "content": content,
                "content_preview": content[:200].replace("\n", " "),
            }
        )
    return results


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval results."""
    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")
    if not results:
        print("  No results found.")
        return

    for rank, r in enumerate(results, start=1):
        print(f"\nRank {rank}  |  similarity={r['similarity']:.4f}")
        print(f"  source_id : {r['source_id']}")
        print(f"  chunk_id  : {r['chunk_id']}")
        print(f"  area      : {r['area']}")
        print(f"  heading   : {r['section_heading']}")
        print(f"  url       : {r['url']}")
        print(f"  preview   : {r['content_preview']}...")
    print(f"\n{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top-k chunks from Neon Postgres using cosine similarity."
    )
    parser.add_argument("--query", type=str, required=True, help="Natural-language query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--area", type=str, default=None, help="Optional area filter (e.g. Housing, Insurance)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Sentence-transformer model (default: {MODEL_NAME})")
    args = parser.parse_args()

    print(f"Embedding query with model: {args.model}")
    query_vec = embed_query(args.query, model_name=args.model)
    print(f"  -> Vector dim: {len(query_vec)}, norm: {sum(x**2 for x in query_vec)**0.5:.6f}")

    results = retrieve(query_vec, top_k=args.top_k, area_filter=args.area)
    print_results(args.query, results)


if __name__ == "__main__":
    main()

