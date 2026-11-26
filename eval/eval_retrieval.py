"""
eval/eval_retrieval.py - Recall@k evaluation for PNW Navigator retrieval pipeline.

Gold set comes from the `key_facts` table (46 facts, each with source_id and source_url).
Each fact's text is used as a surrogate query; the system must retrieve a chunk from
the expected source_id within the top-k results.

No LLM is involved - retrieval only.

Usage:
    python eval/eval_retrieval.py --top-k 5
    python eval/eval_retrieval.py --top-k 5 --limit 10   # test on first 10 facts
    python eval/eval_retrieval.py --top-k 5 --verbose     # print per-query results
    python eval/eval_retrieval.py --top-k 5 --output eval/results.json
"""

import argparse
import io
import json
import os
import sys

# Force UTF-8 stdout on Windows to avoid cp1252 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ingest.db import get_connection
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ingest")))
    from db import get_connection


MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Embedding (with model cache to avoid reloading per query)
# ---------------------------------------------------------------------------

_model_cache = {}


def get_model(model_name: str = MODEL_NAME):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_query(query: str, model=None) -> list:
    import numpy as np
    if model is None:
        model = get_model()
    vec = model.encode([query], normalize_embeddings=True)[0]
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > 1e-4:
        vec = vec / norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Retrieval (inline, avoids cross-module import complexity)
# ---------------------------------------------------------------------------

def retrieve(query_vec: list, top_k: int = 5) -> list:
    """
    Cosine similarity search against chunks.embedding (pgvector <=> operator).
    similarity = 1 - cosine_distance
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            vec_str = json.dumps(query_vec)
            sql = """
                SELECT
                    c.id,
                    c.content,
                    c.metadata,
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
        chunk_db_id, content, metadata, similarity = row
        meta = metadata if isinstance(metadata, dict) else {}
        results.append({
            "chunk_db_id": chunk_db_id,
            "source_id": meta.get("source_id", ""),
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "area": meta.get("area", ""),
            "section_heading": meta.get("section_heading", ""),
            "similarity": round(float(similarity), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Gold set helpers
# ---------------------------------------------------------------------------

def load_gold_set(limit=None) -> list:
    """Pull eval pairs from key_facts; exclude rows with no source_id."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT id, topic, tag, fact, source_id, source_url
                FROM key_facts
                WHERE source_id IS NOT NULL
                  AND fact IS NOT NULL
                ORDER BY id
            """
            if limit:
                sql += f" LIMIT {int(limit)}"
            sql += ";"
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()

    gold = []
    for row in rows:
        fact_id, topic, tag, fact, source_id, source_url = row
        gold.append({
            "fact_id": fact_id,
            "topic": topic or "",
            "tag": tag or "",
            "query": fact,
            "expected_source_id": source_id,
            "source_url": source_url or "",
        })
    return gold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_one(item: dict, top_k: int, model) -> dict:
    """Embed fact text, retrieve top-k, check if expected_source_id is in results."""
    query_vec = embed_query(item["query"], model=model)
    results = retrieve(query_vec, top_k=top_k)
    retrieved_source_ids = [r["source_id"] for r in results]
    expected = item["expected_source_id"]

    hit_at = {}
    for k in [1, 3, 5]:
        if k <= top_k:
            hit_at[k] = expected in retrieved_source_ids[:k]

    return {
        "fact_id": item["fact_id"],
        "topic": item["topic"],
        "expected_source_id": expected,
        "query_preview": item["query"][:80],
        "hit_at": hit_at,
        "top_result_source_id": retrieved_source_ids[0] if retrieved_source_ids else None,
        "top_similarity": results[0]["similarity"] if results else 0.0,
        "top_sources_returned": retrieved_source_ids,
    }


def compute_recall(eval_results: list, k: int) -> float:
    hits = sum(1 for r in eval_results if r["hit_at"].get(k, False))
    return hits / len(eval_results) if eval_results else 0.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(eval_results: list, top_k: int, verbose: bool) -> None:
    n = len(eval_results)

    print(f"\n{'='*65}")
    print(f"PNW NAVIGATOR - RETRIEVAL EVALUATION REPORT")
    print(f"  Gold set size : {n} facts")
    print(f"  Max top-k     : {top_k}")
    print(f"{'='*65}")

    if verbose:
        print(f"\n{'Fact ID':<10} {'Expected':>10} {'Top Hit':>10} {'Sim':>6}  {'@1':>4} {'@3':>4} {'@5':>4}  Query")
        print("-" * 80)
        for r in eval_results:
            ha = r["hit_at"]
            mark = lambda k: "Y" if ha.get(k, False) else "N"
            print(
                f"{r['fact_id']:<10} "
                f"{r['expected_source_id']:>10} "
                f"{(r['top_result_source_id'] or '?'):>10} "
                f"{r['top_similarity']:>6.3f}  "
                f"{mark(1):>4} {mark(3):>4} {mark(5):>4}  "
                f"{r['query_preview'][:45]}..."
            )

    print(f"\n{'k':<8} {'Hits':>8} {'Recall@k':>10}")
    print("-" * 32)
    for k in [1, 3, 5]:
        if k <= top_k:
            recall = compute_recall(eval_results, k)
            hits = sum(1 for r in eval_results if r["hit_at"].get(k, False))
            print(f"  @{k:<5}  {hits:>4}/{n:<4}  {recall:>8.1%}")

    print(f"\n{'='*65}\n")

    misses_at_5 = [r for r in eval_results if not r["hit_at"].get(5, True)]
    if misses_at_5:
        print(f"Missed at @5 ({len(misses_at_5)} facts):")
        for r in misses_at_5:
            print(
                f"  [{r['fact_id']}] expected={r['expected_source_id']}  "
                f"top_hit={r['top_result_source_id']}  "
                f"query: {r['query_preview'][:60]}..."
            )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PNW Navigator retrieval pipeline with Recall@k."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Max k to retrieve per query (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit gold set size (default: all facts)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Sentence-transformer model (default: {MODEL_NAME})")
    parser.add_argument("--verbose", action="store_true", help="Print per-fact result table")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to save results to")
    args = parser.parse_args()

    print(f"Loading gold set from key_facts table (limit={args.limit or 'all'})...")
    gold = load_gold_set(limit=args.limit)
    print(f"  -> {len(gold)} evaluation pairs loaded.")

    if not gold:
        print("ERROR: No gold set rows found. Ensure key_facts table has rows with source_id set.", file=sys.stderr)
        sys.exit(1)

    # Pre-load model once for all queries
    print(f"\nLoading model: {args.model}")
    model = get_model(args.model)
    print(f"  -> Model ready.")

    print(f"\nEmbedding {len(gold)} queries and retrieving top-{args.top_k} chunks each...\n")

    eval_results = []
    for i, item in enumerate(gold, start=1):
        result = eval_one(item, top_k=args.top_k, model=model)
        eval_results.append(result)
        if i % 5 == 0 or i == len(gold):
            hits = sum(1 for r in eval_results if r["hit_at"].get(5, False))
            print(f"  [{i:>3}/{len(gold)}]  Running Recall@5 so far: {hits}/{i}  ({hits/i:.1%})")

    print_summary(eval_results, top_k=args.top_k, verbose=args.verbose)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
