import argparse
import json
import math
import os
import sys
import time
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIM = 384


def compute_l2_norm(vec: list) -> float:
    return math.sqrt(sum(x * x for x in vec))


def embed_chunks(
    input_file: str = "data/chunks.json",
    output_file: str = "data/embedded_chunks.json",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    limit: int = None,
):
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading chunks from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if limit and limit > 0:
        chunks = chunks[:limit]
        print(f"  [Limit Applied]: Embedding {len(chunks)} chunk(s).")
    else:
        print(f"  Total chunks to embed: {len(chunks)}")

    if not chunks:
        print("No chunks to embed.")
        return []

    print(f"Loading embedding model '{model_name}'...")
    start_time = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.2f}s.")

    texts = [c["content"] for c in chunks]
    print(f"Generating normalized 384-d embeddings (batch_size={batch_size})...")
    embed_start = time.time()
    # normalize_embeddings=True applies L2 unit norm so dot product = cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embed_time = time.time() - embed_start
    print(f"  Embeddings generated in {embed_time:.2f}s ({len(chunks)/(embed_time or 0.001):.1f} chunks/sec).")

    # Verification checks
    norms = []
    embedded_chunks = []
    for chunk, emb in zip(chunks, embeddings):
        vec = emb.tolist()
        if len(vec) != EXPECTED_DIM:
            raise ValueError(f"Vector dim {len(vec)} does not match expected {EXPECTED_DIM}!")
        norm = compute_l2_norm(vec)
        norms.append(norm)

        chunk_copy = dict(chunk)
        chunk_copy["embedding"] = vec
        embedded_chunks.append(chunk_copy)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(embedded_chunks, f_out, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Embedding complete! Saved {len(embedded_chunks)} chunks to '{output_file}'.")
    avg_norm = sum(norms) / len(norms)
    print(f"Vector Shape: ({len(embedded_chunks)}, {EXPECTED_DIM})")
    print(f"L2 Norm Verification: Min={min(norms):.4f}, Max={max(norms):.4f}, Avg={avg_norm:.4f} (Target = 1.0)")
    return embedded_chunks


def main():
    parser = argparse.ArgumentParser(description="Generate normalized all-MiniLM-L6-v2 embeddings for chunks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to embed")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--input", type=str, default="data/chunks.json", help="Input chunks.json path")
    parser.add_argument("--output", type=str, default="data/embedded_chunks.json", help="Output embedded_chunks.json path")
    args = parser.parse_args()

    embed_chunks(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

