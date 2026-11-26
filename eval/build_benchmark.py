"""
eval/build_benchmark.py — Generates the 61-query Benchmark Dataset (eval/benchmark_dataset.json).

Contains:
  - 46 Answerable ground truth queries (from key_facts in Neon Postgres)
  - 15 Adversarial & Unanswerable queries (testing strict negative constraints & refusal)
"""

import json
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ingest.db import get_connection
except ImportError:
    from db import get_connection


# 15 Curated Adversarial & Out-of-Scope Queries
ADVERSARIAL_QUERIES = [
    {
        "id": "ADV-01",
        "type": "unanswerable",
        "topic": "Off-Campus Work Policy",
        "question": "Can I work for cash at an off-campus restaurant without informing ISS or getting CPT?",
        "ground_truth": "REFUSAL: F-1 students cannot work off-campus without prior authorization. Contact iss@pnw.edu.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-02",
        "type": "unanswerable",
        "topic": "Purdue West Lafayette",
        "question": "What is the undergraduate tuition rate for aerospace engineering at Purdue University West Lafayette?",
        "ground_truth": "REFUSAL: Out of scope (Purdue West Lafayette is a distinct campus).",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-03",
        "type": "unanswerable",
        "topic": "Housing Pricing 2035",
        "question": "How much will a four-bedroom suite cost in University Village for the 2035-2036 academic year?",
        "ground_truth": "REFUSAL: Future unannounced rates.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-04",
        "type": "unanswerable",
        "topic": "H-1B Visa Lottery",
        "question": "What are the step-by-step instructions to file an H-1B lottery petition with USCIS on my own?",
        "ground_truth": "REFUSAL: ISS only handles F-1/J-1 student guidance; H-1B petitions must be filed by sponsoring employers.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-05",
        "type": "unanswerable",
        "topic": "Pet Policy in Dorms",
        "question": "Can I keep my pet German Shepherd in my University Village apartment?",
        "ground_truth": "REFUSAL: Not specified in retrieved facts.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-06",
        "type": "unanswerable",
        "topic": "Course Waiver",
        "question": "Can I waive my final exams by paying an extra course fee to the Registrar?",
        "ground_truth": "REFUSAL: Non-existent policy.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-07",
        "type": "unanswerable",
        "topic": "Greencard Sponsorship",
        "question": "Does PNW sponsor immediate green cards for international undergraduate students upon enrollment?",
        "ground_truth": "REFUSAL: Inaccurate premise.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-08",
        "type": "unanswerable",
        "topic": "International Travel without Signature",
        "question": "Can I travel outside the United States and re-enter without a travel endorsement signature on my I-20?",
        "ground_truth": "REFUSAL: Travel endorsements are required. Contact iss@pnw.edu.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-09",
        "type": "unanswerable",
        "topic": "Free Health Insurance",
        "question": "How do international students receive 100% free university health insurance without paying the premium?",
        "ground_truth": "REFUSAL: Inaccurate premise; F-1 students must enroll or have an approved comparable waiver.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-10",
        "type": "unanswerable",
        "topic": "Dropping Below Full Time",
        "question": "Can I drop down to 3 credit hours this semester without asking anyone if I feel overwhelmed?",
        "ground_truth": "REFUSAL: Dropping below full time without prior Reduced Course Load (RCL) authorization violates status. Contact iss@pnw.edu.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-11",
        "type": "unanswerable",
        "topic": "Housing Room Swap",
        "question": "Can I swap keys with my friend in a different suite without notifying University Housing?",
        "ground_truth": "REFUSAL: Policy does not permit unauthorized key swaps. Contact housing@pnw.edu.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-12",
        "type": "unanswerable",
        "topic": "Tuition Payment via Cryptocurrency",
        "question": "Can I pay my PNW tuition bill using Bitcoin directly to the Bursar's office?",
        "ground_truth": "REFUSAL: PNW payment methods do not include direct cryptocurrency.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-13",
        "type": "unanswerable",
        "topic": "Parking Violations",
        "question": "If I get three parking tickets on campus, will my F-1 visa be automatically revoked by SEVIS?",
        "ground_truth": "REFUSAL: SEVIS does not manage campus parking tickets.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-14",
        "type": "unanswerable",
        "topic": "Indianapolis Campus Policies",
        "question": "Can I take all my classes in person at the IUPUI campus as a PNW international student?",
        "ground_truth": "REFUSAL: PNW is a separate institution with campuses in Hammond and Westville.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    },
    {
        "id": "ADV-15",
        "type": "unanswerable",
        "topic": "Remote OPT from Abroad",
        "question": "Can I complete all 12 months of my post-completion OPT while permanently living in my home country without entering the US?",
        "ground_truth": "REFUSAL: Post-completion OPT is intended for practical training inside the United States.",
        "expected_behavior": "refusal",
        "expected_source_id": None
    }
]


def build_benchmark(output_path: str = "eval/benchmark_dataset.json"):
    print("[*] Pulling 46 gold facts from Neon Postgres (key_facts table)...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, topic, tag, fact, source_id, source_url
                FROM key_facts
                WHERE source_id IS NOT NULL
                  AND fact IS NOT NULL
                ORDER BY id;
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    dataset = []

    # 1. Answerable Subset
    for r in rows:
        fid, topic, tag, fact, source_id, source_url = r
        dataset.append({
            "id": fid,
            "type": "answerable",
            "topic": topic or "",
            "tag": tag or "",
            "question": fact,
            "ground_truth": fact,
            "expected_behavior": "answer",
            "expected_source_id": source_id,
            "source_url": source_url or ""
        })

    print(f"  -> Added {len(dataset)} answerable evaluation queries.")

    # 2. Adversarial Subset
    for adv in ADVERSARIAL_QUERIES:
        dataset.append(adv)

    print(f"  -> Added {len(ADVERSARIAL_QUERIES)} adversarial/unanswerable queries.")
    print(f"[*] Total Benchmark Dataset Size: {len(dataset)} items.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Benchmark dataset written to: {output_path}")


if __name__ == "__main__":
    build_benchmark()

