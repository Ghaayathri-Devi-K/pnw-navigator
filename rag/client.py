"""
rag/client.py — Unified Multi-Provider LLM Client for PNW Navigator.

Supports:
  - gemini       : Google Gemini via OpenAI-compatible endpoint (uses GOOGLE_API_KEY or GEMINI_API_KEY)
  - openai       : OpenAI via official API (uses OPENAI_API_KEY)
  - local_ollama : Ollama local server (http://localhost:11434/v1)
  - local_vllm   : vLLM local production server (http://localhost:8000/v1)
  - mock         : Deterministic grounded simulator for offline testing & CI

Never prints or logs API keys/secrets.
"""

import os
import re
import sys
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI


# Default models per provider (100% local, on-campus deployment)
DEFAULT_MODELS = {
    "local_ollama": "qwen2.5:7b",
    "local_vllm": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mock": "mock-grounded-extractor",
}

DEFAULT_BASE_URLS = {
    "local_ollama": "http://localhost:11434/v1",
    "local_vllm": "http://localhost:8000/v1",
    "mock": None,
}


def _simulate_grounded_answer(user_prompt: str) -> str:
    """
    Offline simulator that inspects the prompt's Context block and extracts
    the most relevant sentences with their [Source-ID] tags, or refuses if no match.
    """
    # Extract Context and Question from user_prompt
    context_match = re.search(r"### OFFICIAL PNW CONTEXT:\s*(.*?)\s*### STUDENT INQUIRY:", user_prompt, re.DOTALL)
    inquiry_match = re.search(r"### STUDENT INQUIRY:\s*(.*?)\s*### YOUR ANSWER", user_prompt, re.DOTALL)

    context = context_match.group(1) if context_match else ""
    inquiry = inquiry_match.group(1).strip().lower() if inquiry_match else ""

    # Parse individual document blocks: --- Document [SID] ... ---
    doc_blocks = re.findall(r"---\s*Document\s*\[([A-Z]{2,4}-\d{2})\][^\n]*---\n(.*?)(?=(?:---\s*Document|\Z))", context, re.DOTALL)

    inquiry_words = set(re.findall(r"\b[a-z]{3,}\b", inquiry))
    stopwords = {"when", "what", "where", "which", "how", "can", "the", "for", "and", "pnw", "purdue", "university", "northwest"}
    query_keywords = inquiry_words - stopwords

    scored_sentences = []
    for sid, text in doc_blocks:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 20:
                continue
            s_words = set(re.findall(r"\b[a-z]{3,}\b", s_clean.lower()))
            overlap = len(query_keywords & s_words)
            if overlap >= 2:
                scored_sentences.append((overlap, sid, s_clean))

    if not scored_sentences:
        return (
            "I cannot find this information in the official PNW records. "
            "Please contact the International Students and Scholars (ISS) office directly at "
            "iss@pnw.edu or (219) 989-2082."
        )

    # Pick top 2 most relevant sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_sentences[:2]

    answer_parts = []
    for _, sid, s in top_matches:
        # Strip heading tag if already present
        clean_text = re.sub(r"^\[.*?\]\s*", "", s)
        answer_parts.append(f"{clean_text} [{sid}]")

    return " ".join(answer_parts)


def get_llm_client(
    provider: str = "local_ollama",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[Optional[OpenAI], str]:
    """
    Returns an initialized (OpenAI client, model_name) tuple for local serving.
    Never exposes API secrets in exceptions or logs.
    """
    load_dotenv()
    provider = provider.lower().strip()

    # Handle provider:model alias (e.g., local_ollama:mistral:7b)
    if ":" in provider and not provider.startswith("http"):
        parts = provider.split(":", 1)
        provider = parts[0]
        model = model or parts[1]

    if provider in ("gemini", "openai"):
        raise ValueError(
            f"Cloud API provider '{provider}' has been removed. "
            "PNW Navigator operates 100% locally on-campus via 'local_ollama' "
            "(Qwen 2.5 7B, LLaMA 3.1 8B, Mistral 7B, LLaMA 3.2 3B) or 'mock'."
        )

    if provider == "mock":
        return None, "mock-grounded-extractor"

    if provider not in DEFAULT_MODELS and not base_url:
        supported = ", ".join(DEFAULT_MODELS.keys())
        raise ValueError(f"Unknown provider '{provider}'. Supported providers: {supported}")

    chosen_model = model or DEFAULT_MODELS.get(provider, "qwen2.5:7b")
    chosen_base_url = base_url or DEFAULT_BASE_URLS.get(provider)

    resolved_key = api_key or "local-no-key-required"

    client_kwargs = {"api_key": resolved_key}
    if chosen_base_url:
        client_kwargs["base_url"] = chosen_base_url

    client = OpenAI(**client_kwargs)
    return client, chosen_model


def call_llm(
    client: Optional[OpenAI],
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> dict:
    """
    Executes a chat completion call and returns a structured dictionary:
    {
        "content": str,
        "model": str,
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int
    }
    """
    if client is None or model == "mock-grounded-extractor":
        simulated_content = _simulate_grounded_answer(user_prompt)
        words_in = len(user_prompt.split())
        words_out = len(simulated_content.split())
        return {
            "content": simulated_content,
            "model": "mock-grounded-extractor",
            "input_tokens": int(words_in * 1.3),
            "output_tokens": int(words_out * 1.3),
            "total_tokens": int((words_in + words_out) * 1.3),
        }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    usage = response.usage
    return {
        "content": response.choices[0].message.content or "",
        "model": model,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
