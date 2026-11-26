# PNW Navigator: 100% Local RAG International Student Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL pgvector](https://img.shields.io/badge/Postgres-pgvector_HNSW-336791.svg)](https://github.com/pgvector/pgvector)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-black.svg)](https://ollama.com/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA_RTX_4050_6GB-76B900.svg)](https://www.nvidia.com/)

An autonomous, air-gapped, zero-cost **Retrieval-Augmented Generation (RAG)** advising assistant tailored for international students at **Purdue University Northwest (PNW)**. 

PNW Navigator answers complex inquiries regarding F-1/J-1 visa status, Optional Practical Training (OPT), Curricular Practical Training (CPT), on-campus housing, registration deadlines, and health insurance. Every factual claim is backed by clickable citations linking directly to official PNW documents, with **100% refusal accuracy** against adversarial or out-of-scope queries.

---

## Architecture Overview

```
[Student Inquiry via Web UI / CLI]
               │
               ▼
   [Query Embedding Engine] ───> sentence-transformers/all-MiniLM-L6-v2 (384-d)
               │
               ▼
    [Vector Database] ─────────> Neon Serverless Postgres (pgvector + HNSW Cosine Index)
               │                 Retrieves Top-5 Heading-Aware Chunks
               ▼
  [Guardrail Prompt Engine] ───> Strict Negative Refusal + Heading Attribution
               │
               ▼
   [Local LLM Generation] ────> Ollama on NVIDIA RTX 4050 GPU (CUDA v12)
                                 Primary: Qwen 2.5 7B Instruct (Q4_K_M)
                                 Alternatives: LLaMA 3.1 8B, Mistral 7B, LLaMA 3.2 3B
               │
               ▼
[Interactive Response with Clickable Official PNW URL Links]
```

---

## Prerequisites

Before starting, ensure your host environment has:
1. **Git**: [git-scm.com](https://git-scm.com/)
2. **Python 3.10+**: [python.org](https://www.python.org/)
3. **PostgreSQL with pgvector**: A cloud instance on [Neon](https://neon.tech) (free tier supported) or a local PostgreSQL 15+ instance with `pgvector`.
4. **Ollama**: [ollama.com/download](https://ollama.com/download) for serving local open-weights LLMs.
5. **NVIDIA GPU (Recommended)**: NVIDIA RTX 3050/3060/4050/4060 (6GB+ VRAM) with CUDA drivers installed (CPU inference fallback is supported automatically).

---

## Quickstart: Step-by-Step Reproduction Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/pnw-navigator.git
cd pnw-navigator
```

---

### Step 2: Set Up Virtual Environment & Dependencies

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Create your local `.env` configuration file from the template:
```bash
cp .env.example .env
```
Open `.env` in any text editor and provide your PostgreSQL connection string:
```env
DATABASE_URL="postgresql://[user]:[password]@[host]/[dbname]?sslmode=require"
OLLAMA_BASE_URL="http://localhost:11434/v1"
```

---

### Step 4: Database Ingestion & Vector Indexing

Run the database setup and data ingestion scripts. This creates the relational tables (`sources`, `documents`, `chunks`, `key_facts`), loads the 20 official PNW documents, chunks them with heading-awareness (280 chunks), computes dense embeddings (`all-MiniLM-L6-v2`), and builds the HNSW cosine index:

```bash
# 1. Initialize schema and tables
python ingest/db.py

# 2. Ingest and vector-embed documents into Postgres
python ingest/load.py

# 3. Verify database health and chunk counts
python ingest/check_db.py
```
*Expected Output: `20 documents, 280 chunks, 46 gold facts loaded into Neon Postgres.`*

---

### Step 5: Start Local LLM Inference Engine (Ollama)

1. Start the Ollama server in a separate terminal:
   ```bash
   ollama serve
   ```
2. Pull the primary recommended model (Qwen 2.5 7B Instruct):
   ```bash
   ollama pull qwen2.5:7b
   ```
3. *(Optional)* Pull the comparative models for ablation studies:
   ```bash
   ollama pull llama3.1:8b
   ollama pull mistral:7b
   ollama pull llama3.2:3b
   ```

---

### Step 6: Launch the Web Application

Start the FastAPI application with Uvicorn:
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser. You can:
* Ask international student questions (e.g., *"When can I apply for post-completion OPT?"*).
* View generated answers with **clickable citation pills** linking directly to official PNW webpages.
* Inspect retrieved vector context chunks with similarity scores.
* Switch between local models (`qwen2.5:7b`, `llama3.1:8b`, `mistral:7b`, `llama3.2:3b`, or offline `mock`) via the top **Provider** dropdown.

---

## Command Line Interface (CLI)

You can also run direct CLI queries without the web browser:

```bash
# Run using the default local GPU model (qwen2.5:7b)
python rag/synthesizer.py --query "When can I apply for post-completion OPT?"

# Run with an alternative local model
python rag/synthesizer.py --query "How much is University Village housing?" --model mistral:7b

# Run in interactive conversational terminal mode
python rag/synthesizer.py --interactive
```

---

## Reproducing Evaluation & Ablation Benchmarks

All evaluation harnesses are automated and output structured JSON and Markdown reports:

### 1. Retrieval Hit Rate Benchmark (Phase 1)
Evaluates `Recall@1`, `Recall@3`, and `Recall@5` across the 46 gold-standard factual queries:
```bash
python eval/eval_retrieval.py
```
*Expected Results: Recall@1 = 87.0%, Recall@3 = 91.3%, Recall@5 = 91.3%.*

### 2. Generation & Safety Guardrail Evaluation
Evaluates citation accuracy and adversarial refusal against the 61-item benchmark:
```bash
python eval/eval_generation.py --provider local_ollama --model qwen2.5:7b
```

### 3. Head-to-Head Local Model Comparison on GPU
Runs comparative benchmarking across all installed local models on your GPU:
```bash
python eval/eval_local_comparison.py --models qwen2.5:7b llama3.1:8b mistral:7b llama3.2:3b --include-closed-book --include-permissive-prompt --limit 15
```

### 4. Full 6-Way RAG Ablation Study
Runs the 6 controlled ablation experiments (Chunking, Depth $k \in \{1, 3, 5, 10\}$, Closed-Book, Permissive Prompt):
```bash
python eval/run_ablation.py
```

---

## Empirical Benchmark Summary

Evaluated on an **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**:

| Configuration / Model Architecture | Citation Precision | Mean Groundedness | Adversarial Refusal | Latency / Query | Active VRAM |
|---|:---:|:---:|:---:|:---:|:---:|
| **🏆 Qwen 2.5 7B ($k=5$, Strict Prompt)** | **70.0%** | **90.1%** | **100.0%** (5/5 safe) | **7.35s** | **4,119 MiB** |
| **Meta LLaMA 3.1 8B ($k=5$, Strict)** | 40.0% | 65.4% | 0.0% (0/5 safe) | 14.47s | 4,155 MiB |
| **Mistral 7B v0.3 ($k=5$, Strict)** | 70.0% | 90.0% | 100.0% (5/5 safe) | 18.38s | 4,155 MiB |
| **Meta LLaMA 3.2 3B ($k=5$, Strict)** | 10.0% | 43.8% | 20.0% (1/5 safe) | 5.40s | 4,125 MiB |
| **Ablation: Closed-Book LLM ($k=0$, No RAG)** | 0.0% | 2.6% | 100.0% | 5.44s | 4,105 MiB |
| **Ablation: Permissive Prompt (No Guardrails)**| 40.0% | 69.4% | 0.0% (0/5 safe) | 20.97s | 4,113 MiB |

*Full academic report available in [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).*

---

## Repository Structure

```
pnw-navigator/
├── app/
│   ├── api.py                    # FastAPI web server and interactive HTML/JS UI
│   └── __init__.py
├── data/
│   ├── raw/                      # 20 raw scraped PNW administrative HTML/text files
│   ├── chunks.json               # 280 heading-aware preprocessed chunks
│   └── key_facts.json            # 46 extracted ground truth Q&A pairs
├── eval/
│   ├── benchmark_dataset.json    # 61-item benchmark (46 answerable + 15 adversarial)
│   ├── eval_retrieval.py         # Evaluates Recall@k on Neon Postgres vector search
│   ├── eval_generation.py        # Evaluates citation precision and groundedness
│   ├── eval_local_comparison.py  # Head-to-head local LLM and ablation benchmark runner
│   ├── run_ablation.py           # 6-way systematic pipeline ablation runner
│   └── local_llm_comparison_report.md
├── ingest/
│   ├── db.py                     # Neon Postgres connection & schema migration
│   ├── chunk.py                  # Heading-aware document parsing & chunking
│   ├── load.py                   # Embedding generation & database loading
│   ├── retrieve.py               # Vector similarity query engine
│   └── check_db.py               # Database verification utility
├── rag/
│   ├── client.py                 # Multi-model LLM client (Ollama/vLLM/Mock)
│   ├── prompts.py                # Strict system guardrails & context formatting
│   └── synthesizer.py            # End-to-end RAG query synthesizer
├── sources.csv                   # Authoritative registry of PNW URLs and Source IDs
├── requirements.txt              # Production dependency specifications
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git tracking ignore rules
├── EVALUATION_REPORT.md          # Comprehensive course final evaluation report
└── README.md                     # Project documentation & reproduction guide
```

---

## License & Attribution

Developed for **CS 58000 (Introduction to Artificial Intelligence)** at **Purdue University Northwest**.  
All official policy documents and university data are copyright © Purdue University Northwest.
