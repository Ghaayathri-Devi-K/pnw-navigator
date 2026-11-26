# Comprehensive Empirical Evaluation of Local LLMs on NVIDIA RTX 4050 GPU (6GB VRAM)
**Project**: PNW Navigator — Autonomous RAG Assistant for International Students  
**Environment**: 100% Local On-Campus Deployment (No Cloud API Keys Required)  
**Host Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), CUDA 12.6/13.3, Driver 610.60  
**Serving Engine**: Ollama v0.40.18 (`http://localhost:11434/v1`) with CUDA v12 acceleration  
**Benchmark Suite**: `eval/eval_local_comparison.py` testing against gold-standard items from `eval/benchmark_dataset.json`  

---

## 1. Executive Summary

To assess feasibility for an **air-gapped, zero-cost, privacy-preserving campus deployment** at Purdue University Northwest, we performed an extensive ablation and comparative study across open-weights models running directly on a student/lab laptop equipped with an NVIDIA RTX 4050 GPU (6GB VRAM).

We evaluated:
1. **Four Local Model Architectures**:
   - `qwen2.5:7b` (Alibaba / Qwen Team)
   - `llama3.1:8b` (Meta AI — recommended Tier 3 in `LLM_SELECTION_AND_ABLATION_STUDY.md`)
   - `mistral:7b` (Mistral AI — recommended Tier 3 in `LLM_SELECTION_AND_ABLATION_STUDY.md`)
   - `llama3.2:3b` (Meta AI — lightweight edge model)
2. **Two Critical Ablation Experiments** (from `LLM_SELECTION_AND_ABLATION_STUDY.md`):
   - **Experiment 5: Closed-Book vs. RAG Pipeline Validation**
   - **Experiment 4: Guardrail Prompt Ablation (Strict vs. Permissive System Prompt)**

---

## 2. Head-to-Head Comparative & Ablation Matrix

Full results are recorded in `eval/local_llm_comparison_results.json`:

| Configuration / Model Architecture | Citation Precision | Mean Groundedness | Refusal Accuracy (Adversarial) | Mean Latency / Query | Peak GPU VRAM (MiB) |
|---|:---:|:---:|:---:|:---:|:---:|
| **🏆 Qwen 2.5 (7B Instruct)** *(Recommended)* | **70.0%** | **90.1%** | **100.0%** (5/5 safe) | **7.35s** | **4,119 MiB** |
| **Meta LLaMA 3.1 (8B Instruct)** | 40.0% | 65.4% | 0.0% (0/5 safe) | 14.47s | 4,155 MiB |
| **Mistral (7B Instruct v0.3)** | 70.0% | 90.0% | 100.0% (5/5 safe) | 18.38s | 4,155 MiB |
| **Meta LLaMA 3.2 (3B Instruct)** | 10.0% | 43.8% | 20.0% (1/5 safe) | 5.40s | 4,125 MiB |
| **Ablation: Qwen 2.5 (Closed-Book / No RAG)** | 0.0% | 2.6% | 100.0% | 5.44s | 4,105 MiB |
| **Ablation: Qwen 2.5 (Permissive Prompt)** | 40.0% | 69.4% | 0.0% (0/5 safe) | 20.97s | 4,113 MiB |

---

## 3. Key Empirical Findings & Insights

### Insight 1: Why Qwen 2.5 7B is the Definitive Winner for Local Campus Serving
* **Speed & Throughput**: Qwen 2.5 7B completed queries in **7.35 seconds** at **24.2 tokens/sec**, which is **2.5x faster than Mistral 7B (18.38s)** and **2x faster than LLaMA 3.1 8B (14.47s)**.
* **VRAM Efficiency**: Fits within **4.1 GB of VRAM**, leaving **2.0 GB of VRAM headroom** on the 6GB RTX 4050.
* **Safety & Compliance**: Achieved a flawless **100.0% Refusal Accuracy** on unanswerable/adversarial queries (e.g. unauthorized cash employment, H-1B filings, out-of-scope policies), immediately directing students to `iss@pnw.edu`.

### Insight 2: LLaMA 3.1 8B Bottlenecks on 6GB VRAM
* **Memory Offloading Overhead**: Because LLaMA 3.1 has 8.03 billion parameters, Ollama offloaded 26 layers into CUDA VRAM (3,506 MiB) and kept 7 layers in host system RAM (1,178 MiB). The PCI-e bus memory transfer cut generation speed in half to ~11.5 tokens/sec.
* **Negative Constraint Failure**: LLaMA 3.1 scored **0.0% on Refusal Accuracy** because it attempted to answer adversarial questions with general caveats instead of strictly emitting the official referral block (`iss@pnw.edu`, `(219) 989-2082`).

### Insight 3: Mistral 7B is High-Quality but Latency-Constrained
* Mistral 7B matched Qwen 2.5 in Citation Precision (70.0%), Groundedness (90.0%), and Refusal Accuracy (100.0%).
* However, its average response latency of **18.38s** is too sluggish for responsive student advising.

### Insight 4: Validation of RAG Pipeline (Closed-Book Ablation — Exp 5)
* Stripping retrieved context drops factual groundedness from **90.1% down to 2.6%**, and citation precision to **0.0%**.
* This proves that the local LLM cannot answer PNW-specific policies from parametric memory; the retrieval pipeline (`all-MiniLM-L6-v2` + Neon Postgres pgvector) provides **+87.5% factual accuracy**.

### Insight 5: Impact of Guardrails (Prompt Ablation — Exp 4)
* Removing the strict refusal guardrail causes refusal accuracy to collapse from **100.0% to 0.0%**.
* Permissive prompting causes the model to engage with illegal employment inquiries and bloats token output (inflating latency from 7.35s to 20.97s).

---

## 4. Web UI & Provider Integration

The PNW Navigator web interface (`app/api.py`) and backend API (`rag/client.py`) have been updated with native support for all evaluated local models:

### Provider Dropdown Options:
- **`local_ollama:qwen2.5:7b`**: Local Ollama: Qwen 2.5 7B *(Recommended — High Accuracy & Safety)*
- **`local_ollama:llama3.1:8b`**: Local Ollama: LLaMA 3.1 8B *(Meta Open-Weights)*
- **`local_ollama:mistral:7b`**: Local Ollama: Mistral 7B *(Mistral AI)*
- **`local_ollama:llama3.2:3b`**: Local Ollama: LLaMA 3.2 3B *(Fast Edge)*
- **`mock`**: Deterministic Grounded Extractor *(Offline CI)*
- **`gemini`**: Google Gemini 1.5 Flash *(Cloud API)*
- **`openai`**: OpenAI GPT-4o-mini *(Cloud API)*

### Running the App:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) and select any model from the dropdown.
