"""
app/api.py — FastAPI Application and Interactive Web Interface for PNW Navigator.

Features:
  - GET  /           : Modern responsive student web chat interface with clickable citations
  - POST /api/chat   : RAG query endpoint (answers with clickable citations & context)
  - GET  /api/health : Neon Postgres health check & counts
  - GET  /api/sources: List of indexed official university sources

Usage:
  python app/api.py
  # Open in browser: http://localhost:8000
"""

import csv
import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from rag.synthesizer import answer_query
    from ingest.db import get_connection
except ImportError:
    from synthesizer import answer_query
    from db import get_connection


# Load sources.csv to resolve official URLs for all source IDs
SOURCES_MAP = {}
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sources.csv"))
if os.path.exists(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("id")
            url = row.get("url")
            if sid and url:
                SOURCES_MAP[sid.strip()] = url.strip()


app = FastAPI(
    title="PNW Navigator API",
    description="RAG Assistant for International Students at Purdue University Northwest",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    query: str
    provider: Optional[str] = "local_ollama:qwen2.5:7b"
    model: Optional[str] = None
    top_k: Optional[int] = 5
    area: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[str]
    source_urls: dict[str, str] = {}
    retrieved_chunks: list[dict]
    provider: str
    model: str
    token_usage: dict


@app.get("/api/health")
def health_check():
    """Verify Neon Postgres connection and record counts."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks;")
            chunk_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM key_facts;")
            facts_count = cur.fetchone()[0]
            return {
                "status": "healthy",
                "database": "Neon Postgres (pgvector)",
                "documents": doc_count,
                "chunks": chunk_count,
                "gold_facts": facts_count,
            }
    finally:
        conn.close()


@app.get("/api/sources")
def list_sources():
    """Returns official registered PNW data sources."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, area, page, url, tag FROM sources ORDER BY id;")
            rows = cur.fetchall()
            return [
                {"id": r[0], "area": r[1], "title": r[2], "url": r[3], "tag": r[4]}
                for r in rows
            ]
    finally:
        conn.close()


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Answer a student question using vector retrieval and grounded synthesis."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    provider = (req.provider or "mock").strip()
    model = req.model
    if ":" in provider and not provider.startswith("http"):
        parts = provider.split(":", 1)
        provider = parts[0]
        model = model or parts[1]

    try:
        result = answer_query(
            query=req.query.strip(),
            provider=provider,
            model=model,
            top_k=req.top_k or 5,
            area_filter=req.area,
        )

        # Resolve URLs for all citations
        source_urls = {}
        for sid in result.get("citations", []):
            url = SOURCES_MAP.get(sid)
            if not url:
                for c in result.get("retrieved_chunks", []):
                    if c.get("source_id") == sid and c.get("url"):
                        url = c.get("url")
                        break
            if url:
                source_urls[sid] = url

        result["source_urls"] = source_urls
        return result
    except Exception as e:
        err_msg = str(e)
        suggestion = "Switch the Provider dropdown at the top to 'Mock Extractor (Offline / Fast)' to test immediately without external API keys."
        
        if "API_KEY" in err_msg or "API key" in err_msg or "400" in err_msg:
            friendly_answer = f"⚠️ **API Key / Provider Error ({req.provider})**\n\nThe selected provider requires a valid API key. (Details: {err_msg})\n\n💡 **Quick Fix**: {suggestion}"
        elif "Connection" in err_msg or "connection" in err_msg:
            friendly_answer = f"⚠️ **Connection Error ({req.provider})**\n\nCould not reach the local model server. Ensure your local engine (Ollama or vLLM) is running, or:\n\n💡 **Quick Fix**: {suggestion}"
        else:
            friendly_answer = f"⚠️ **Service Notice ({type(e).__name__})**\n\n{err_msg}\n\n💡 **Quick Fix**: {suggestion}"

        return ChatResponse(
            query=req.query.strip(),
            answer=friendly_answer,
            citations=[],
            source_urls={},
            retrieved_chunks=[],
            provider=req.provider or "mock",
            model="notice-handler",
            token_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


# Embedded Interactive HTML/JS Web Interface
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PNW Navigator — International Student Advising</title>
  <style>
    :root {
      --pnw-gold: #C28E00;
      --pnw-dark-gold: #9E7400;
      --pnw-black: #1A1A1A;
      --bg: #F8F9FA;
      --card-bg: #FFFFFF;
      --text: #212529;
      --muted: #6C757D;
      --border: #DEE2E6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); display: flex; flex-direction: column; min-height: 100vh; }
    header { background-color: var(--pnw-black); color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 4px solid var(--pnw-gold); }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-badge { background: var(--pnw-gold); color: black; font-weight: 800; font-size: 1.1rem; padding: 4px 10px; border-radius: 4px; }
    .brand h1 { font-size: 1.3rem; font-weight: 600; }
    .badge-sub { font-size: 0.85rem; color: #CCC; }
    .container { max-width: 960px; margin: 1.5rem auto; width: 95%; flex: 1; display: flex; flex-direction: column; gap: 1.5rem; }
    .config-bar { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1.2rem; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; font-size: 0.9rem; }
    .config-bar select, .config-bar input { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.85rem; }
    .chat-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; min-height: 380px; max-height: 520px; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; }
    .msg { display: flex; flex-direction: column; max-width: 85%; line-height: 1.5; font-size: 0.95rem; }
    .msg.user { align-self: flex-end; background: #0056B3; color: white; padding: 10px 16px; border-radius: 14px 14px 2px 14px; }
    .msg.bot { align-self: flex-start; background: #F1F3F5; color: var(--text); padding: 14px 18px; border-radius: 14px 14px 14px 2px; border: 1px solid #E9ECEF; width: 100%; max-width: 100%; }
    
    /* Clickable Citation Pill Styling */
    .msg.bot a.citation-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: var(--pnw-gold);
      color: #1A1A1A;
      font-weight: 700;
      font-size: 0.8rem;
      padding: 2px 9px;
      border-radius: 12px;
      margin: 0 3px;
      text-decoration: none;
      transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .msg.bot a.citation-pill:hover {
      background: #E5A800;
      color: #000;
      text-decoration: underline;
      transform: translateY(-1px);
      box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .msg.bot span.citation-pill {
      display: inline-block;
      background: #E0E0E0;
      color: #444;
      font-weight: 600;
      font-size: 0.8rem;
      padding: 2px 8px;
      border-radius: 12px;
      margin: 0 3px;
    }

    .context-drawer { margin-top: 12px; font-size: 0.84rem; background: #FFF; border: 1px solid #E0E0E0; border-radius: 6px; padding: 8px 12px; }
    .context-drawer summary { cursor: pointer; font-weight: 600; color: var(--muted); }
    .chunk-item { margin-bottom: 8px; border-bottom: 1px solid #F0F0F0; padding-bottom: 6px; }
    .chunk-item:last-child { border-bottom: none; }
    .chunk-link { color: #0056B3; text-decoration: none; font-weight: 600; }
    .chunk-link:hover { text-decoration: underline; }

    .input-section { display: flex; gap: 10px; }
    .input-section input { flex: 1; padding: 14px 16px; font-size: 1rem; border: 2px solid var(--border); border-radius: 8px; outline: none; }
    .input-section input:focus { border-color: var(--pnw-gold); }
    .input-section button { background: var(--pnw-gold); color: black; font-weight: 700; border: none; padding: 0 24px; border-radius: 8px; cursor: pointer; transition: 0.2s; }
    .input-section button:hover { background: var(--pnw-dark-gold); color: white; }
    .suggestions { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { background: white; border: 1px solid #D0D0D0; padding: 6px 12px; border-radius: 16px; font-size: 0.85rem; cursor: pointer; transition: 0.15s; }
    .chip:hover { background: #EEE; border-color: #999; }
    footer { text-align: center; padding: 1rem; color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--border); background: white; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">PNW</div>
      <div>
        <h1>PNW Navigator</h1>
        <div class="badge-sub">International Student Advising Assistant — Purdue University Northwest</div>
      </div>
    </div>
    <div style="font-size:0.85rem; color:#DDD;" id="db-status">Connecting to Neon...</div>
  </header>

  <div class="container">
    <div class="config-bar">
      <label><strong>Provider:</strong>
        <select id="providerSelect">
          <option value="local_ollama:qwen2.5:7b" selected>Local Ollama: Qwen 2.5 7B (Recommended)</option>
          <option value="local_ollama:llama3.1:8b">Local Ollama: LLaMA 3.1 8B (Meta)</option>
          <option value="local_ollama:mistral:7b">Local Ollama: Mistral 7B (Mistral AI)</option>
          <option value="local_ollama:llama3.2:3b">Local Ollama: LLaMA 3.2 3B (Fast Edge)</option>
          <option value="mock">Mock Extractor (Offline / Fast)</option>
          <option value="local_vllm">Local vLLM (localhost:8000)</option>
        </select>
      </label>
      <label><strong>Retrieval Top-K:</strong>
        <select id="topKSelect">
          <option value="3">k = 3</option>
          <option value="5" selected>k = 5 (Optimal)</option>
          <option value="10">k = 10</option>
        </select>
      </label>
      <div style="margin-left:auto; color:var(--muted); font-size:0.8rem;">
        ⚡ Click any citation badge to view the official source page
      </div>
    </div>

    <div class="suggestions">
      <div class="chip" onclick="askChip('When can I apply for post-completion OPT?')">📅 OPT Application Timeline</div>
      <div class="chip" onclick="askChip('How much is a 4-bedroom suite in University Village?')">🏠 Housing Rates</div>
      <div class="chip" onclick="askChip('How do I submit a health insurance waiver?')">🏥 Health Insurance Waiver</div>
      <div class="chip" onclick="askChip('Can I drop below 12 credit hours as an F-1 student?')">⚠️ Dropping Below Full-Time</div>
    </div>

    <div class="chat-box" id="chatBox">
      <div class="msg bot">
        <strong>Welcome to PNW Navigator!</strong><br>
        I provide official guidance for international students regarding F-1/J-1 status, on-campus housing, registration deadlines, and health insurance. Every factual statement is backed by clickable citations linking directly to official PNW documents.
      </div>
    </div>

    <div class="input-section">
      <input type="text" id="queryInput" placeholder="Ask a question (e.g. When is the earliest I can apply for OPT?)" onkeydown="if(event.key==='Enter') sendQuery()">
      <button onclick="sendQuery()" id="sendBtn">Ask</button>
    </div>
  </div>

  <footer>
    PNW Navigator &bull; Built with Neon Serverless Postgres, pgvector & all-MiniLM-L6-v2 &bull; Purdue University Northwest
  </footer>

  <script>
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        document.getElementById('db-status').innerText = `✅ Neon: ${data.documents} docs / ${data.chunks} chunks active`;
      } catch (e) {
        document.getElementById('db-status').innerText = `⚠️ Neon: Disconnected`;
      }
    }
    checkHealth();

    function askChip(text) {
      document.getElementById('queryInput').value = text;
      sendQuery();
    }

    async function sendQuery() {
      const input = document.getElementById('queryInput');
      const query = input.value.trim();
      if (!query) return;

      const provider = document.getElementById('providerSelect').value;
      const top_k = parseInt(document.getElementById('topKSelect').value);
      const chatBox = document.getElementById('chatBox');
      const btn = document.getElementById('sendBtn');

      // Append User message
      chatBox.innerHTML += `<div class="msg user">${escapeHtml(query)}</div>`;
      input.value = '';
      btn.disabled = true;
      btn.innerText = 'Searching...';
      chatBox.scrollTop = chatBox.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, provider, top_k })
        });
        const data = await res.json();

        // Convert [SID] citations into clickable HTML links
        let answerFormatted = escapeHtml(data.answer);
        const sourceUrls = data.source_urls || {};

        (data.citations || []).forEach(sid => {
          const url = sourceUrls[sid];
          const re = new RegExp('\\\\[' + sid + '\\\\]', 'g');
          if (url) {
            answerFormatted = answerFormatted.replace(
              re,
              `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="citation-pill" title="Open official PNW page for ${sid}">[${sid}] ↗</a>`
            );
          } else {
            answerFormatted = answerFormatted.replace(
              re,
              `<span class="citation-pill">[${sid}]</span>`
            );
          }
        });

        // Context details with active hyperlinks
        let contextHtml = '<details class="context-drawer"><summary>View Retrieved Context Chunks (' + data.retrieved_chunks.length + ')</summary><div style="margin-top:8px;">';
        data.retrieved_chunks.forEach((c, idx) => {
          const chunkUrl = c.url || sourceUrls[c.source_id] || '#';
          contextHtml += `<div class="chunk-item">
            <strong>[${idx+1}] <a href="${escapeHtml(chunkUrl)}" target="_blank" class="chunk-link">${escapeHtml(c.source_id)}</a></strong>
            (Sim: ${c.similarity}) — <em>${escapeHtml(c.title || 'Official Document')}</em>
            ${c.section_heading ? `&bull; <strong>${escapeHtml(c.section_heading)}</strong>` : ''}<br>
            <span style="color:#555;">${escapeHtml(c.content_preview)}...</span>
          </div>`;
        });
        contextHtml += '</div></details>';

        chatBox.innerHTML += `<div class="msg bot">
          <div>${answerFormatted.replace(/\\n/g, '<br>')}</div>
          ${contextHtml}
          <div style="font-size:0.75rem; color:#888; margin-top:8px;">
            Engine: ${escapeHtml(data.provider)} (${escapeHtml(data.model)}) &bull; Tokens: ${data.token_usage.total_tokens}
          </div>
        </div>`;
      } catch (err) {
        chatBox.innerHTML += `<div class="msg bot" style="color:red;">Error retrieving answer. Please check connection.</div>`;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Ask';
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.innerText = text;
      return div.innerHTML;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_home():
    """Serves the interactive PNW Navigator student web interface."""
    return HTML_TEMPLATE


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
