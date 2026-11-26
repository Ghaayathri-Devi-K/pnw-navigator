"""
eval/eval_local_comparison.py — Head-to-Head Comparative Benchmark of Local LLMs on RTX 4050 GPU.

Evaluates local models (e.g. Qwen 2.5 7B, LLaMA 3.2 3B, LLaMA 3.1 8B) on:
  1. RAG Citation Precision: Strictly citing [SOURCE-ID] matching retrieved context.
  2. Groundedness Score: Semantic alignment with verified ground truth facts.
  3. Negative Constraint Adherence: Immediate refusal of unanswerable/out-of-scope queries with iss@pnw.edu referral.
  4. Latency & Throughput: End-to-end response time per query.
  5. Peak VRAM Footprint: Real-time GPU memory consumption on NVIDIA RTX 4050 (6GB).

Usage:
  python eval/eval_local_comparison.py --models qwen2.5:7b llama3.2:3b --limit 15
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.synthesizer import answer_query
from eval.eval_generation import evaluate_single_answer


def get_gpu_vram_mb() -> Optional[int]:
    """Queries nvidia-smi for current memory usage in MiB."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(res.stdout.strip().split("\n")[0])
    except Exception:
        return None


def benchmark_model(
    model_name: str,
    test_items: List[Dict[str, Any]],
    top_k: int = 5,
    permissive_prompt: bool = False,
    closed_book: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Runs the benchmark suite on a specific local Ollama model."""
    tag = model_name
    if closed_book:
        tag += " (Closed-Book / No Context)"
    elif permissive_prompt:
        tag += " (Permissive Prompt)"

    print(f"\n{'='*70}")
    print(f"  BENCHMARKING CONFIGURATION: {tag}")
    print(f"{'='*70}")

    initial_vram = get_gpu_vram_mb()
    print(f"[*] Initial GPU VRAM: {initial_vram} MiB")

    results = []
    latencies = []
    peak_vram = initial_vram or 0

    # Warmup query
    print(f"[*] Sending warmup query to initialize weights in VRAM...")
    try:
        answer_query("What is PNW?", provider="local_ollama", model=model_name, top_k=1)
    except Exception as e:
        print(f"[!] Warmup failed for {model_name}: {e}")

    loaded_vram = get_gpu_vram_mb()
    if loaded_vram and loaded_vram > peak_vram:
        peak_vram = loaded_vram
    print(f"[*] Active VRAM after model load: {loaded_vram} MiB (Delta: +{(loaded_vram or 0) - (initial_vram or 0)} MiB)")

    for idx, item in enumerate(test_items, 1):
        q = item["question"]
        q_type = item.get("type", "answerable")

        t0 = time.time()
        try:
            if closed_book:
                # Closed-book: retrieve 0 chunks, ask model without context
                rag_output = answer_query(
                    query=q,
                    provider="local_ollama",
                    model=model_name,
                    top_k=0,
                    permissive_prompt=permissive_prompt,
                )
            else:
                rag_output = answer_query(
                    query=q,
                    provider="local_ollama",
                    model=model_name,
                    top_k=top_k,
                    permissive_prompt=permissive_prompt,
                )
            elapsed = round(time.time() - t0, 2)
        except Exception as e:
            print(f"[!] Error on query '{q}': {e}")
            rag_output = {"answer": f"Error: {e}", "citations": []}
            elapsed = round(time.time() - t0, 2)

        latencies.append(elapsed)

        # Track VRAM
        curr_vram = get_gpu_vram_mb()
        if curr_vram and curr_vram > peak_vram:
            peak_vram = curr_vram

        eval_res = evaluate_single_answer(item, rag_output)
        eval_res["latency_sec"] = elapsed
        results.append(eval_res)

        status_mark = "PASS" if eval_res["pass"] else "FAIL"
        print(
            f"[{idx:>2}/{len(test_items)}] {item['id']:<7} ({q_type:<12}) -> "
            f"[{status_mark}] Groundedness: {eval_res['groundedness_score']:.2f} | Latency: {elapsed}s"
        )
        if verbose or not eval_res["pass"]:
            print(f"      Q: {q[:65]}...")
            print(f"      A: {eval_res['generated_answer'][:80]}...")
            if q_type == "answerable":
                print(f"      Expected: {eval_res['expected_source']} | Cited: {eval_res['citations_detected']}")
            else:
                print(f"      Refusal: {eval_res['refusal_detected']} | Contact: {eval_res['contact_provided']}")

    ans_results = [r for r in results if r["type"] == "answerable"]
    unans_results = [r for r in results if r["type"] == "unanswerable"]

    cite_pass = sum(1 for r in ans_results if r["citation_correct"])
    cite_acc = cite_pass / max(len(ans_results), 1)

    mean_groundedness = sum(r["groundedness_score"] for r in ans_results) / max(len(ans_results), 1)
    refusal_pass = sum(1 for r in unans_results if r["refusal_detected"] and r["contact_provided"])
    refusal_rate = refusal_pass / max(len(unans_results), 1)

    overall_pass = sum(1 for r in results if r["pass"])
    overall_acc = overall_pass / max(len(results), 1)
    avg_latency = sum(latencies) / max(len(latencies), 1)

    return {
        "model": tag,
        "base_model": model_name,
        "closed_book": closed_book,
        "permissive_prompt": permissive_prompt,
        "total_queries": len(results),
        "overall_accuracy": round(overall_acc, 3),
        "citation_precision": round(cite_acc, 3),
        "mean_groundedness": round(mean_groundedness, 3),
        "refusal_accuracy": round(refusal_rate, 3),
        "avg_latency_sec": round(avg_latency, 2),
        "initial_vram_mb": initial_vram,
        "peak_vram_mb": peak_vram,
        "vram_delta_mb": peak_vram - (initial_vram or 0),
        "detailed_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Local LLMs on RTX 4050 GPU")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b", "llama3.2:3b"], help="List of Ollama models")
    parser.add_argument("--benchmark", type=str, default="eval/benchmark_dataset.json", help="Benchmark path")
    parser.add_argument("--limit", type=int, default=15, help="Number of benchmark items to test per model")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth")
    parser.add_argument("--include-closed-book", action="store_true", help="Include Closed-Book (No RAG) baseline on primary model")
    parser.add_argument("--include-permissive-prompt", action="store_true", help="Include Permissive Prompt ablation on primary model")
    parser.add_argument("--output", type=str, default="eval/local_llm_comparison_results.json", help="Output JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-query debug output")
    args = parser.parse_args()

    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Balanced selection of answerable and unanswerable queries
    ans_items = [item for item in benchmark_data if item.get("type", "answerable") == "answerable"]
    unans_items = [item for item in benchmark_data if item.get("type", "answerable") == "unanswerable"]

    # Select representative subset
    ans_limit = int(args.limit * 0.7)
    unans_limit = args.limit - ans_limit
    test_items = ans_items[:ans_limit] + unans_items[:unans_limit]

    print(f"[*] Running Comparative Benchmark across models: {args.models}")
    print(f"[*] Test set: {len(test_items)} queries ({len(ans_items[:ans_limit])} answerable, {len(unans_items[:unans_limit])} adversarial/refusal)")

    comparison_results = []
    for model in args.models:
        metrics = benchmark_model(model, test_items, top_k=args.top_k, verbose=args.verbose)
        comparison_results.append(metrics)

    # Optional Ablation: Closed-Book LLM (No RAG)
    if args.include_closed_book and args.models:
        cb_model = args.models[0]
        print(f"\n[*] Running Ablation Exp 5: Closed-Book LLM on {cb_model}...")
        cb_metrics = benchmark_model(cb_model, test_items, closed_book=True, verbose=args.verbose)
        comparison_results.append(cb_metrics)

    # Optional Ablation: Permissive Prompt (No strict guardrails)
    if args.include_permissive_prompt and args.models:
        perm_model = args.models[0]
        print(f"\n[*] Running Ablation Exp 4: Permissive Prompt on {perm_model}...")
        perm_metrics = benchmark_model(perm_model, test_items, top_k=args.top_k, permissive_prompt=True, verbose=args.verbose)
        comparison_results.append(perm_metrics)

    # Save JSON summary
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Full comparison metrics written to: {args.output}")

    # Print Final Markdown Comparison Table
    print(f"\n{'='*78}")
    print("                      HEAD-TO-HEAD MODEL COMPARISON REPORT")
    print(f"{'='*78}")
    print(f"| Model               | Citation Prec. | Groundedness | Refusal Acc. | Latency/Q | Peak VRAM  |")
    print(f"|---------------------|----------------|--------------|--------------|-----------|------------|")
    for r in comparison_results:
        print(
            f"| {r['model']:<19} | "
            f"{r['citation_precision']:>13.1%} | "
            f"{r['mean_groundedness']:>11.1%} | "
            f"{r['refusal_accuracy']:>11.1%} | "
            f"{r['avg_latency_sec']:>7.2f}s | "
            f"{r['peak_vram_mb']:>7} MB |"
        )
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
