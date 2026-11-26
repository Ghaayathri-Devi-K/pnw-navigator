"""
rag/synthesizer.py — End-to-End RAG Answer Synthesizer for PNW Navigator.

Connects:
  Query -> Query Embedding -> Neon Postgres (pgvector) -> Top-K Chunks -> Prompt Engine -> LLM Generation

Usage:
  python rag/synthesizer.py --query "When can I apply for OPT?"
  python rag/synthesizer.py --query "How much is the housing deposit?" --provider gemini
  python rag/synthesizer.py --query "What is the health insurance waiver deadline?" --top-k 3
  python rag/synthesizer.py --interactive
"""

import argparse
import io
import os
import re
import sys
from typing import Optional

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set offline mode for local transformers caching
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

try:
    from ingest.retrieve import embed_query, retrieve
    from rag.client import get_llm_client, call_llm
    from rag.prompts import STRICT_SYSTEM_PROMPT, PERMISSIVE_SYSTEM_PROMPT, format_rag_prompt
except ImportError:
    from retrieve import embed_query, retrieve
    from client import get_llm_client, call_llm
    from prompts import STRICT_SYSTEM_PROMPT, PERMISSIVE_SYSTEM_PROMPT, format_rag_prompt


def answer_query(
    query: str,
    provider: str = "local_ollama",
    model: Optional[str] = None,
    top_k: int = 5,
    area_filter: Optional[str] = None,
    permissive_prompt: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Executes the end-to-end RAG pipeline for a given query.
    Returns:
    {
        "query": str,
        "answer": str,
        "citations": list[str],
        "retrieved_chunks": list[dict],
        "provider": str,
        "model": str,
        "token_usage": dict
    }
    """
    if verbose:
        print(f"[*] Embedding query: '{query}'...")

    query_vec = embed_query(query)

    if verbose:
        print(f"[*] Retrieving top-{top_k} chunks from Neon Postgres...")

    chunks = retrieve(query_vec, top_k=top_k, area_filter=area_filter)

    system_prompt = PERMISSIVE_SYSTEM_PROMPT if permissive_prompt else STRICT_SYSTEM_PROMPT
    user_prompt = format_rag_prompt(query, chunks)

    client, chosen_model = get_llm_client(provider=provider, model=model)

    if verbose:
        print(f"[*] Generating response with provider '{provider}' ({chosen_model})...")

    llm_result = call_llm(
        client=client,
        model=chosen_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )

    answer = llm_result["content"].strip()
    citations = sorted(list(set(re.findall(r"\[([A-Z]{2,4}-\d{2})\]", answer))))

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": chunks,
        "provider": provider,
        "model": chosen_model,
        "token_usage": {
            "input_tokens": llm_result["input_tokens"],
            "output_tokens": llm_result["output_tokens"],
            "total_tokens": llm_result["total_tokens"],
        },
    }


def print_qa_result(result: dict, verbose: bool = False) -> None:
    print(f"\n{'='*70}")
    print(f"QUESTION: {result['query']}")
    print(f"{'='*70}")

    if verbose:
        print("\n--- RETRIEVED CONTEXT CHUNKS ---")
        for i, c in enumerate(result["retrieved_chunks"], start=1):
            print(f"  [{i}] Source: {c.get('source_id')} | Similarity: {c.get('similarity')} | {c.get('title')}")
            if c.get("section_heading"):
                print(f"      Heading: {c.get('section_heading')}")
            print(f"      URL: {c.get('url')}")
        print("--------------------------------\n")

    print("\nANSWER:")
    print(result["answer"])

    print(f"\n{'-'*70}")
    print(f"Citations Detected : {', '.join(result['citations']) if result['citations'] else 'None'}")
    print(f"Engine             : {result['provider']} ({result['model']})")
    usage = result["token_usage"]
    print(f"Token Usage        : {usage['input_tokens']} in / {usage['output_tokens']} out ({usage['total_tokens']} total)")
    print(f"{'='*70}\n")


def interactive_session(provider: str, model: Optional[str], top_k: int) -> None:
    print("\n=======================================================")
    print("  PNW Navigator — Interactive Advising Session")
    print(f"  Provider: {provider} | Top-K: {top_k}")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=======================================================\n")

    while True:
        try:
            query = input("\nStudent Inquiry > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Ending session. Goodbye!")
                break

            result = answer_query(query=query, provider=provider, model=model, top_k=top_k)
            print_qa_result(result)
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"Error processing inquiry: {type(e).__name__}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="PNW Navigator RAG Answer Synthesizer")
    parser.add_argument("--query", type=str, default=None, help="Student inquiry question")
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "local_ollama", "local_vllm", "mock"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--area", type=str, default=None, help="Optional area filter (e.g. Housing, Insurance)")
    parser.add_argument("--permissive", action="store_true", help="Use permissive prompt (for ablation testing)")
    parser.add_argument("--verbose", action="store_true", help="Display retrieved chunk details and tokens")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat session")
    args = parser.parse_args()

    if args.interactive:
        interactive_session(provider=args.provider, model=args.model, top_k=args.top_k)
    elif args.query:
        result = answer_query(
            query=args.query,
            provider=args.provider,
            model=args.model,
            top_k=args.top_k,
            area_filter=args.area,
            permissive_prompt=args.permissive,
            verbose=args.verbose,
        )
        print_qa_result(result, verbose=args.verbose)
    else:
        print("Please provide a query via --query or use --interactive to start a session.")
        parser.print_help()


if __name__ == "__main__":
    main()
