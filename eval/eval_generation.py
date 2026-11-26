"""
eval/eval_generation.py — Automated Generation Evaluation for PNW Navigator.

Evaluates generated answers against the 61-item benchmark (eval/benchmark_dataset.json):
  - 46 Answerable queries: measures Groundedness, Semantic Alignment, Citation Precision
  - 15 Adversarial queries: measures Refusal Accuracy, Negative Constraint Adherence

Usage:
  python eval/eval_generation.py --provider mock --limit 10
  python eval/eval_generation.py --provider mock
  python eval/eval_generation.py --provider gemini --limit 5
"""

import argparse
import io
import json
import os
import re
import sys
import time
from typing import Optional

# Ensure UTF-8 stdout on Windows without re-wrapping if already UTF-8
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from rag.synthesizer import answer_query
except ImportError:
    from synthesizer import answer_query


def evaluate_single_answer(item: dict, result: dict) -> dict:
    """
    Evaluates one generated answer against its benchmark item.
    """
    answer = result["answer"].strip()
    q_type = item.get("type", "answerable")
    expected_source = item.get("expected_source_id")

    # 1. Extract citations
    citations = re.findall(r"\[([A-Z]{2,4}-\d{2})\]", answer)
    has_citations = len(citations) > 0

    # 2. Check citation match (for answerable items)
    citation_correct = False
    if expected_source:
        expected_sids = [s.strip() for s in expected_source.split(",")]
        citation_correct = any(exp in citations for exp in expected_sids)

    # 3. Check refusal adherence (for unanswerable items)
    refusal_keywords = [
        "cannot find",
        "not available",
        "please contact",
        "iss@pnw.edu",
        "housing@pnw.edu",
        "registrar@pnw.edu",
        "do not have",
        "not permitted",
        "outside",
        "official pnw records",
    ]
    refusal_detected = any(kw in answer.lower() for kw in refusal_keywords)
    contact_provided = any(c in answer.lower() for c in ["iss@pnw.edu", "housing@pnw.edu", "registrar@pnw.edu", "(219)"])

    # 4. Factual grounding score (0.0 to 1.0)
    if q_type == "unanswerable":
        # Successful refusal = 1.0 grounding, hallucinated answer = 0.0
        groundedness = 1.0 if (refusal_detected and contact_provided) else 0.0
        is_pass = refusal_detected
    else:
        # For answerable: must have citation and reasonable word overlap with ground truth
        gt_words = set(re.findall(r"\b[a-z]{4,}\b", item.get("ground_truth", "").lower()))
        ans_words = set(re.findall(r"\b[a-z]{4,}\b", answer.lower()))
        overlap = len(gt_words & ans_words) / max(len(gt_words), 1)

        # Grounding is high if citation exists and overlaps with truth
        groundedness = min(1.0, (0.5 if has_citations else 0.0) + (0.5 * min(1.0, overlap * 1.5)))
        is_pass = citation_correct and groundedness >= 0.5

    return {
        "id": item["id"],
        "type": q_type,
        "topic": item.get("topic", ""),
        "question": item["question"],
        "ground_truth": item.get("ground_truth", ""),
        "generated_answer": answer,
        "citations_detected": citations,
        "expected_source": expected_source,
        "has_citations": has_citations,
        "citation_correct": citation_correct,
        "refusal_detected": refusal_detected,
        "contact_provided": contact_provided,
        "groundedness_score": round(groundedness, 3),
        "pass": is_pass,
    }


def run_evaluation(
    benchmark_file: str = "eval/benchmark_dataset.json",
    provider: str = "mock",
    model: Optional[str] = None,
    limit: Optional[int] = None,
    top_k: int = 5,
    output_file: Optional[str] = "eval/generation_results.json",
    verbose: bool = False,
):
    if not os.path.exists(benchmark_file):
        print(f"ERROR: Benchmark dataset '{benchmark_file}' not found. Run eval/build_benchmark.py first.", file=sys.stderr)
        sys.exit(1)

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if limit and limit > 0:
        # Balance sample: take half answerable, half unanswerable
        ans_items = [x for x in benchmark if x.get("type") == "answerable"][:limit // 2 or 1]
        unans_items = [x for x in benchmark if x.get("type") == "unanswerable"][:limit - len(ans_items)]
        items_to_eval = ans_items + unans_items
    else:
        items_to_eval = benchmark

    print(f"\n{'='*75}")
    print(f"  PNW NAVIGATOR — GENERATION EVALUATION BENCHMARK")
    print(f"  Engine Provider : {provider} ({model or 'default'})")
    print(f"  Total Items     : {len(items_to_eval)} ({sum(1 for x in items_to_eval if x['type']=='answerable')} answerable, {sum(1 for x in items_to_eval if x['type']=='unanswerable')} adversarial)")
    print(f"  Retrieval Top-K : {top_k}")
    print(f"{'='*75}\n")

    eval_results = []
    start_time = time.time()

    for idx, item in enumerate(items_to_eval, start=1):
        q = item["question"]
        q_type = item["type"]
        q_id = item["id"]

        t0 = time.time()
        res = answer_query(query=q, provider=provider, model=model, top_k=top_k)
        latency = round(time.time() - t0, 3)

        metric = evaluate_single_answer(item, res)
        metric["latency_sec"] = latency
        eval_results.append(metric)

        status_mark = "PASS" if metric["pass"] else "FAIL"
        print(f"[{idx:>2}/{len(items_to_eval)}] {q_id:<7} ({q_type:<12}) -> [{status_mark}]  Groundedness: {metric['groundedness_score']:.2f}  Latency: {latency}s")

        if verbose or not metric["pass"]:
            print(f"      Q: {q[:70]}...")
            print(f"      A: {metric['generated_answer'][:90]}...")
            if q_type == "answerable":
                print(f"      Expected: {metric['expected_source']} | Cited: {metric['citations_detected']}")
            else:
                print(f"      Refusal: {metric['refusal_detected']} | Contact: {metric['contact_provided']}")
            print()

    total_time = round(time.time() - start_time, 2)

    # Compute Aggregate Metrics
    ans_results = [r for r in eval_results if r["type"] == "answerable"]
    unans_results = [r for r in eval_results if r["type"] == "unanswerable"]

    ans_pass = sum(1 for r in ans_results if r["pass"])
    ans_cite_pass = sum(1 for r in ans_results if r["citation_correct"])
    ans_avg_groundedness = sum(r["groundedness_score"] for r in ans_results) / max(len(ans_results), 1)

    unans_pass = sum(1 for r in unans_results if r["pass"])
    unans_refusal_rate = sum(1 for r in unans_results if r["refusal_detected"]) / max(len(unans_results), 1)

    overall_pass = sum(1 for r in eval_results if r["pass"])
    overall_accuracy = overall_pass / len(eval_results) if eval_results else 0.0

    print(f"\n{'='*75}")
    print("  EVALUATION SUMMARY & PERFORMANCE REPORT")
    print(f"{'='*75}")
    print(f"Execution Time : {total_time}s across {len(eval_results)} queries (Avg: {total_time/len(eval_results):.2f}s/query)")

    print("\n--- 1. ANSWERABLE QUERIES (Ground Truth Factual Alignment) ---")
    print(f"  Total Evaluated        : {len(ans_results)}")
    print(f"  Citation Precision     : {ans_cite_pass}/{len(ans_results)} ({ans_cite_pass/max(len(ans_results), 1):.1%})")
    print(f"  Mean Groundedness Score: {ans_avg_groundedness:.1%}")
    print(f"  Factual QA Pass Rate   : {ans_pass}/{len(ans_results)} ({ans_pass/max(len(ans_results), 1):.1%})")

    print("\n--- 2. ADVERSARIAL & UNANSWERABLE QUERIES (Negative Constraints) ---")
    print(f"  Total Evaluated        : {len(unans_results)}")
    print(f"  Refusal Accuracy       : {unans_refusal_rate:.1%}")
    print(f"  Hallucination Avoidance: {unans_pass}/{len(unans_results)} ({unans_pass/max(len(unans_results), 1):.1%})")

    print("\n--- 3. OVERALL COMPOSITE SCORE ---")
    print(f"  TOTAL ACCURACY         : {overall_pass}/{len(eval_results)} ({overall_accuracy:.1%})")
    print(f"{'='*75}\n")

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        summary_payload = {
            "metadata": {
                "provider": provider,
                "model": model,
                "top_k": top_k,
                "total_items": len(eval_results),
                "total_time_seconds": total_time,
                "overall_accuracy": round(overall_accuracy, 3),
            },
            "metrics": {
                "answerable_citation_accuracy": round(ans_cite_pass / max(len(ans_results), 1), 3),
                "answerable_mean_groundedness": round(ans_avg_groundedness, 3),
                "adversarial_refusal_accuracy": round(unans_refusal_rate, 3),
            },
            "detailed_results": eval_results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Generation evaluation results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PNW Navigator Generation Quality")
    parser.add_argument("--benchmark", type=str, default="eval/benchmark_dataset.json", help="Path to benchmark JSON")
    parser.add_argument("--provider", type=str, default="local_ollama", choices=["local_ollama", "local_vllm", "mock"], help="LLM engine (100% local)")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth (default: 5)")
    parser.add_argument("--output", type=str, default="eval/generation_results.json", help="Path to output JSON")
    parser.add_argument("--verbose", action="store_true", help="Print per-item question/answer details")
    args = parser.parse_args()

    run_evaluation(
        benchmark_file=args.benchmark,
        provider=args.provider,
        model=args.model,
        limit=args.limit,
        top_k=args.top_k,
        output_file=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
