import argparse
import glob
import json
import os
import re
import sys

# Rule: strictly fits within all-MiniLM-L6-v2 limit (256 tokens ~ 700-800 chars)
DEFAULT_MAX_CHARS = 700
DEFAULT_OVERLAP = 80


def estimate_tokens(text: str) -> int:
    # 1 word ~ 1.3 tokens, plus punctuation
    words = len(text.split())
    char_est = len(text) // 4
    return max(int(words * 1.3), char_est)


def split_text_by_words(text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP):
    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for w in words:
        w_len = len(w) + 1
        if current_len + w_len > max_chars and current_words:
            chunk_str = " ".join(current_words).strip()
            if chunk_str:
                chunks.append(chunk_str)
            # Find overlap in words
            overlap_words = []
            overlap_len = 0
            for ow in reversed(current_words):
                if overlap_len + len(ow) + 1 <= overlap:
                    overlap_words.insert(0, ow)
                    overlap_len += len(ow) + 1
                else:
                    break
            current_words = overlap_words + [w]
            current_len = sum(len(x) + 1 for x in current_words)
        else:
            current_words.append(w)
            current_len += w_len

    if current_words:
        chunk_str = " ".join(current_words).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks


def split_section_into_chunks(content: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [content.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_chars:
            # First flush what we have
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

            # Sub-split long paragraph by sentences or words
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_chunk = []
            sent_len = 0
            for s in sentences:
                if len(s) > max_chars:
                    if sent_chunk:
                        chunks.append(" ".join(sent_chunk).strip())
                        sent_chunk = []
                        sent_len = 0
                    # Split giant single sentence by words
                    for word_chunk in split_text_by_words(s, max_chars=max_chars, overlap=overlap):
                        chunks.append(word_chunk)
                elif sent_len + len(s) + 1 > max_chars and sent_chunk:
                    chunks.append(" ".join(sent_chunk).strip())
                    sent_chunk = [s]
                    sent_len = len(s)
                else:
                    sent_chunk.append(s)
                    sent_len += len(s) + 1
            if sent_chunk:
                chunks.append(" ".join(sent_chunk).strip())

        else:
            if current_length + para_len + 2 > max_chars and current_chunk:
                chunk_text = "\n\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # Overlap
                overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = sum(len(x) + 2 for x in current_chunk)
            else:
                current_chunk.append(para)
                current_length += para_len + 2

    if current_chunk:
        chunk_text = "\n\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_document(doc_data: dict, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP):
    doc_id = doc_data["id"]
    title = doc_data.get("title", "")
    url = doc_data.get("url", "")
    area = doc_data.get("area", "")
    tag = doc_data.get("tag", "")
    valid_for = doc_data.get("valid_for_term", "")
    sections = doc_data.get("sections", [])

    chunks = []
    chunk_index = 0

    if sections:
        for sec in sections:
            heading = sec.get("heading", title).strip()
            sec_content = sec.get("content", "").strip()
            if not sec_content:
                continue

            sec_chunks = split_section_into_chunks(sec_content, max_chars=max_chars, overlap=overlap)
            for raw_chunk in sec_chunks:
                clean_chunk = raw_chunk.strip()
                if not clean_chunk:
                    continue

                # Add contextual heading prefix if helpful
                chunk_text = clean_chunk
                if heading and not chunk_text.lower().startswith(heading.lower()):
                    chunk_text = f"[{heading}] {chunk_text}"

                chunk_index += 1
                chunks.append(
                    {
                        "id": f"{doc_id}_c{chunk_index:03d}",
                        "source_id": doc_id,
                        "chunk_index": chunk_index,
                        "title": title,
                        "url": url,
                        "area": area,
                        "tag": tag,
                        "valid_for_term": valid_for,
                        "section_heading": heading,
                        "content": chunk_text,
                        "char_count": len(chunk_text),
                        "token_estimate": estimate_tokens(chunk_text),
                    }
                )
    else:
        full_text = doc_data.get("text", "").strip()
        raw_chunks = split_section_into_chunks(full_text, max_chars=max_chars, overlap=overlap)
        for raw_chunk in raw_chunks:
            chunk_index += 1
            chunks.append(
                {
                    "id": f"{doc_id}_c{chunk_index:03d}",
                    "source_id": doc_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "url": url,
                    "area": area,
                    "tag": tag,
                    "valid_for_term": valid_for,
                    "section_heading": title,
                    "content": raw_chunk,
                    "char_count": len(raw_chunk),
                    "token_estimate": estimate_tokens(raw_chunk),
                }
            )

    return chunks


def process_chunks(input_dir: str = "data/parsed", output_file: str = "data/chunks.json", limit: int = None, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP):
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        print(f"No parsed JSON files found in {input_dir}.", file=sys.stderr)
        return []

    if limit and limit > 0:
        files = files[:limit]

    print(f"Processing heading-aware chunks for {len(files)} document(s)...")
    all_chunks = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        doc_chunks = chunk_document(doc_data, max_chars=max_chars, overlap=overlap)
        all_chunks.extend(doc_chunks)
        print(f"  - {doc_data['id']}: {len(doc_chunks)} chunk(s)")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(all_chunks, f_out, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Generated {len(all_chunks)} total chunks saved to {output_file}.")
    if all_chunks:
        avg_chars = sum(c["char_count"] for c in all_chunks) / len(all_chunks)
        max_c = max(c["char_count"] for c in all_chunks)
        min_c = min(c["char_count"] for c in all_chunks)
        max_tok = max(c["token_estimate"] for c in all_chunks)
        print(f"Character Stats: Min={min_c}, Max={max_c}, Avg={avg_chars:.1f}")
        print(f"Max token estimate: {max_tok} (Model limit is 256 tokens)")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Heading-aware chunker for PNW Navigator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to chunk")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum characters per chunk")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Character overlap between chunks")
    parser.add_argument("--input-dir", type=str, default="data/parsed", help="Path to parsed documents directory")
    parser.add_argument("--output", type=str, default="data/chunks.json", help="Path to output chunks.json")
    args = parser.parse_args()

    process_chunks(
        input_dir=args.input_dir,
        output_file=args.output,
        limit=args.limit,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
