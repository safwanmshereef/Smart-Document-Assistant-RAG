#!/usr/bin/env python3
"""
eval/run_eval.py — Smart Document Assistant RAG Evaluation Harness
===================================================================
Runs a curated set of Q&A pairs against the live API and scores answers
using keyword-based matching. No external judge or paid API required.

Prerequisites
-------------
1. Start the FastAPI backend:
       uvicorn app.main:app --host 127.0.0.1 --port 8000
   OR with Docker:
       docker-compose up

2. Upload the three sample documents via the Streamlit UI or the API:
       curl -X POST http://localhost:8000/documents/upload \
            -F "file=@sample_data/NEXUS ENTERPRISE SOLUTIONS.pdf"
       curl -X POST http://localhost:8000/documents/upload \
            -F "file=@sample_data/vertex_q2_financials.pdf"
       curl -X POST http://localhost:8000/documents/upload \
            -F "file=@sample_data/datacore_ops_manual.pdf"

3. Run this script:
       python eval/run_eval.py

Scoring
-------
Each question has a list of required keywords. An answer PASSES if it
contains ALL keywords (case-insensitive). This is a simple but honest
proxy for correctness — it avoids the cost and non-determinism of an
LLM-as-judge while still catching hallucinations and missing values.

Outputs
-------
- Per-question PASS / FAIL with the actual answer truncated for display
- Category-level accuracy breakdown
- Overall score printed to stdout
- Full results saved to eval/eval_results.json
"""

import json
import time
import uuid
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
PROVIDER = "ollama"          # "google" or "ollama" — must match your setup
MODEL_NAME = "llama3.2:3b"   # Use llama3.2:3b as local model default
REQUEST_DELAY_S = 2.0        # Seconds to wait between API calls (avoid rate limits)
OUTPUT_FILE = Path(__file__).parent / "eval_results.json"

# ---------------------------------------------------------------------------
# Eval Dataset — 15 questions across all 3 sample documents (5 per document)
# Each question has:
#   id        - unique identifier
#   document  - which PDF this question targets
#   question  - the natural language query sent to the agent
#   keywords  - ALL of these strings must appear in the answer (case-insensitive)
#   category  - "factual" | "calculation" | "guardrail" | "web_search"
# ---------------------------------------------------------------------------
EVAL_DATASET = [

    # ── Document 1: Nexus HR Policy ─────────────────────────────────────────
    {
        "id": "hr-01",
        "document": "NEXUS ENTERPRISE SOLUTIONS.pdf",
        "question": "What is the baseline annual leave allocation for permanent employees at Nexus?",
        "keywords": ["24"],
        "category": "factual",
    },
    {
        "id": "hr-02",
        "document": "NEXUS ENTERPRISE SOLUTIONS.pdf",
        "question": "How many weeks of fully paid maternity leave are female employees eligible for?",
        "keywords": ["26"],
        "category": "factual",
    },
    {
        "id": "hr-03",
        "document": "NEXUS ENTERPRISE SOLUTIONS.pdf",
        "question": (
            "If an Associate Engineer works at Nexus for a full year, "
            "what is their total annual Internet Allowance payout?"
        ),
        "keywords": ["30,000", "30000"],   # accept either formatting
        "category": "calculation",
    },
    {
        "id": "hr-04",
        "document": "NEXUS ENTERPRISE SOLUTIONS.pdf",
        "question": "What is the age limit for dependent children to be covered under the health insurance plan?",
        "keywords": ["21"],
        "category": "factual",
    },
    {
        "id": "hr-05",
        "document": "NEXUS ENTERPRISE SOLUTIONS.pdf",
        "question": (
            "Does the policy mention any monetary allowances for purchasing "
            "a remote work desk or ergonomic office chair?"
        ),
        "keywords": ["don't know", "do not contain", "not contain", "no information"],
        "category": "guardrail",
    },

    # ── Document 2: Vertex Q2 2026 Financials ────────────────────────────────
    {
        "id": "fin-01",
        "document": "vertex_q2_financials.pdf",
        "question": "Which operating segment generated the highest revenue in Q2 2026?",
        "keywords": ["cloud", "saas"],
        "category": "factual",
    },
    {
        "id": "fin-02",
        "document": "vertex_q2_financials.pdf",
        "question": "What percentage of total revenue was contributed by the Asia-Pacific (APAC) region?",
        "keywords": ["20"],
        "category": "factual",
    },
    {
        "id": "fin-03",
        "document": "vertex_q2_financials.pdf",
        "question": (
            "What is the exact net operating profit (Revenue minus Total OPEX) for Q2 2026?"
        ),
        "keywords": ["5,25,00,000", "52500000", "5.25"],
        "category": "calculation",
    },
    {
        "id": "fin-04",
        "document": "vertex_q2_financials.pdf",
        "question": "What was the year-on-year revenue increase percentage compared to Q2 2025?",
        "keywords": ["11"],
        "category": "factual",
    },
    {
        "id": "fin-05",
        "document": "vertex_q2_financials.pdf",
        "question": "What is the projected net profit margin or revenue forecast for Q3 2026?",
        "keywords": ["don't know", "do not contain", "not contain", "no information"],
        "category": "guardrail",
    },

    # ── Document 3: DataCore Ops Manual ─────────────────────────────────────
    {
        "id": "ops-01",
        "document": "datacore_ops_manual.pdf",
        "question": "What is the maximum allowed Time-to-Resolve (TTR) deadline for a Severity 1 incident?",
        "keywords": ["45"],
        "category": "factual",
    },
    {
        "id": "ops-02",
        "document": "datacore_ops_manual.pdf",
        "question": "How long are high-frequency trace logs maintained at full verbosity before compression?",
        "keywords": ["7"],
        "category": "factual",
    },
    {
        "id": "ops-03",
        "document": "datacore_ops_manual.pdf",
        "question": (
            "During a disaster recovery simulation, what is the maximum absolute number "
            "of CPUs that can be utilised including the elastic burst allowance?"
        ),
        "keywords": ["540"],
        "category": "calculation",
    },
    {
        "id": "ops-04",
        "document": "datacore_ops_manual.pdf",
        "question": "What algorithm and compression level are applied to the trace logs after 7 days?",
        "keywords": ["zstandard", "3"],
        "category": "factual",
    },
    {
        "id": "ops-05",
        "document": "datacore_ops_manual.pdf",
        "question": (
            "Which specific public cloud provider (AWS, Azure, or GCP) "
            "does DataCore use to host its deep glacier cold vaults?"
        ),
        "keywords": ["don't know", "do not contain", "not contain", "no information"],
        "category": "guardrail",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    """Returns True if the FastAPI backend is reachable."""
    try:
        resp = requests.get(f"{API_BASE}/", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def score_answer(answer: str, keywords: list[str]) -> bool:
    """
    PASS if the answer contains at least one of the provided keywords
    (case-insensitive). For guardrail questions the keywords are all
    refusal-indicating phrases — any one of them is sufficient.
    """
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def call_chat(session_id: str, question: str) -> dict:
    """Sends a single message to POST /chat and returns the JSON response."""
    payload = {
        "session_id": session_id,
        "message": question,
        "provider": PROVIDER,
        "model_name": MODEL_NAME,
    }
    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def truncate(text: str, max_len: int = 120) -> str:
    """Truncates a string for display purposes."""
    return text if len(text) <= max_len else text[:max_len] + "…"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    print("=" * 70)
    print("  Smart Document Assistant — RAG Evaluation Harness")
    print("=" * 70)

    # Health check
    if not check_api_health():
        print("\n❌  Cannot reach the API at:", API_BASE)
        print("    Start it first:  uvicorn app.main:app --port 8000\n")
        sys.exit(1)
    print(f"\n✅  API reachable at {API_BASE}")
    print(f"    Provider : {PROVIDER}")
    print(f"    Questions: {len(EVAL_DATASET)}\n")

    results = []
    passed = 0
    by_category: dict[str, dict] = {}

    for idx, item in enumerate(EVAL_DATASET, start=1):
        session_id = str(uuid.uuid4())   # fresh session per question
        q_id = item["id"]
        question = item["question"]
        keywords = item["keywords"]
        category = item["category"]

        print(f"[{idx:02d}/{len(EVAL_DATASET)}] {q_id}  ({category})")
        print(f"         Q: {truncate(question, 80)}")

        try:
            response = call_chat(session_id, question)
            answer = response.get("output", "")
            trace = response.get("reasoning_trace", [])
            tools_used = [t["tool"] for t in trace]
        except requests.exceptions.HTTPError as exc:
            answer = f"HTTP ERROR: {exc}"
            tools_used = []
        except Exception as exc:
            answer = f"ERROR: {exc}"
            tools_used = []

        passed_flag = score_answer(answer, keywords)
        passed += int(passed_flag)
        status_str = "✅ PASS" if passed_flag else "❌ FAIL"

        print(f"         A: {truncate(answer, 100)}")
        print(f"         Tools used : {tools_used if tools_used else 'none'}")
        print(f"         Score      : {status_str}\n")

        # Category tracking
        if category not in by_category:
            by_category[category] = {"passed": 0, "total": 0}
        by_category[category]["total"] += 1
        by_category[category]["passed"] += int(passed_flag)

        results.append({
            "id": q_id,
            "document": item["document"],
            "category": category,
            "question": question,
            "keywords": keywords,
            "answer": answer,
            "tools_used": tools_used,
            "passed": passed_flag,
        })

        if idx < len(EVAL_DATASET):
            time.sleep(REQUEST_DELAY_S)

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(EVAL_DATASET)
    pct = (passed / total) * 100 if total else 0

    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Overall : {passed}/{total}  ({pct:.1f}%)\n")
    print("  By category:")
    for cat, counts in sorted(by_category.items()):
        cat_pct = (counts["passed"] / counts["total"]) * 100
        bar = "█" * counts["passed"] + "░" * (counts["total"] - counts["passed"])
        print(f"    {cat:<14}  {bar}  {counts['passed']}/{counts['total']}  ({cat_pct:.0f}%)")

    if pct >= 80:
        grade = "🟢  GOOD  — system is performing well"
    elif pct >= 60:
        grade = "🟡  FAIR  — some retrieval or reasoning gaps"
    else:
        grade = "🔴  NEEDS WORK — check document upload and LLM config"

    print(f"\n  Grade: {grade}")
    print("=" * 70)

    # ── Save full results ────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "passed": passed,
                    "total": total,
                    "accuracy_pct": round(pct, 2),
                    "by_category": by_category,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Full results saved → {OUTPUT_FILE}\n")


if __name__ == "__main__":
    run_evaluation()
