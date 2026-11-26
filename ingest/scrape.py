import argparse
import csv
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pypdf
import requests

# Allowed domains per .agents/rules/pnw-navigator.md
ALLOWED_DOMAINS = ["pnw.edu", "pnw.myahpcare.com"]

DEFAULT_HEADERS = {
    "User-Agent": "PNWNavigatorBot/1.0 (+https://www.pnw.edu/international-student-services/)"
}


def is_allowed_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        # Remove port if present
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        for allowed in ALLOWED_DOMAINS:
            if netloc == allowed or netloc.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode whitespace & multiple spaces/newlines
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_html_content(html_content: str, fallback_title: str = ""):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for el in soup(
        ["script", "style", "nav", "footer", "header", "noscript", "svg", "aside", "form"]
    ):
        el.decompose()

    # Determine page title
    title = fallback_title
    if soup.title and soup.title.string:
        clean_title = soup.title.string.strip()
        if clean_title:
            title = clean_title
    elif soup.find("h1"):
        h1_text = soup.find("h1").get_text(strip=True)
        if h1_text:
            title = h1_text

    # Extract sections based on headings
    sections = []
    current_heading = title
    current_paragraphs = []

    # Target main content area if available
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main", re.I))
        or soup.body
        or soup
    )

    for el in main_content.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "tr", "table", "blockquote"]
    ):
        tag_name = el.name.lower()
        if tag_name in ["h1", "h2", "h3", "h4"]:
            heading_text = el.get_text(strip=True)
            if heading_text:
                if current_paragraphs:
                    section_text = clean_text("\n".join(current_paragraphs))
                    if section_text:
                        sections.append(
                            {"heading": current_heading, "content": section_text}
                        )
                    current_paragraphs = []
                current_heading = heading_text
        elif tag_name == "table":
            table_text = clean_text(el.get_text(separator=" | ", strip=True))
            if table_text:
                current_paragraphs.append(table_text)
        else:
            text = el.get_text(strip=True)
            if text:
                current_paragraphs.append(text)

    if current_paragraphs:
        section_text = clean_text("\n".join(current_paragraphs))
        if section_text:
            sections.append({"heading": current_heading, "content": section_text})

    full_text = clean_text(main_content.get_text(separator="\n", strip=True))
    return title, full_text, sections


def parse_pdf_file(pdf_path: str, fallback_title: str = ""):
    reader = pypdf.PdfReader(pdf_path)
    sections = []
    full_text_parts = []

    for idx, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if page_text:
            full_text_parts.append(page_text)
            sections.append({"heading": f"Page {idx}", "content": page_text})

    full_text = clean_text("\n\n".join(full_text_parts))
    return fallback_title, full_text, sections


def fetch_with_retry(
    url: str, is_binary: bool = False, max_retries: int = 3, delay: float = 1.0
):
    time.sleep(delay)  # Polite crawl delay
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.content if is_binary else resp.text
            elif resp.status_code in [429, 500, 502, 503, 504]:
                wait_time = attempt * 2
                print(
                    f"    HTTP {resp.status_code} received. Retrying in {wait_time}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                print(
                    f"    HTTP error {resp.status_code} fetching {url}",
                    file=sys.stderr,
                )
                return None
        except requests.RequestException as e:
            if attempt == max_retries:
                print(f"    Request failed after {max_retries} attempts: {e}", file=sys.stderr)
                return None
            wait_time = attempt * 2
            print(
                f"    Network error ({type(e).__name__}). Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
    return None


def scrape_sources(csv_path: str = "sources.csv", limit: int = None, target_id: str = None, force: bool = False):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/parsed", exist_ok=True)

    sources = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row)

    if target_id:
        sources = [s for s in sources if s["id"] == target_id]

    if limit and limit > 0:
        sources = sources[:limit]

    print(f"Starting scraping for {len(sources)} source(s)...")
    scraped_count = 0
    skipped_count = 0

    for idx, src in enumerate(sources, start=1):
        sid = src["id"]
        url = src["url"]
        fmt = src["format"].lower()
        title = src["title"]
        raw_ext = "pdf" if fmt == "pdf" else "html"
        raw_path = os.path.join("data", "raw", f"{sid}.{raw_ext}")
        parsed_path = os.path.join("data", "parsed", f"{sid}.json")

        print(f"\n[{idx}/{len(sources)}] Processing {sid}: {title}")
        print(f"  URL: {url}")

        # Check domain allowlist
        if not is_allowed_domain(url):
            print(f"  [SKIPPED - BLOCKED DOMAIN] URL {url} does not match allowed domains!")
            skipped_count += 1
            continue

        # Check existing cache unless force
        if not force and os.path.exists(raw_path) and os.path.exists(parsed_path):
            print(f"  [CACHED] Raw & parsed files already exist. Loading cached parsed data.")
            scraped_count += 1
            continue

        is_binary = (fmt == "pdf")
        content = fetch_with_retry(url, is_binary=is_binary)
        if content is None:
            print(f"  [FAILED] Could not retrieve content for {sid}.")
            skipped_count += 1
            continue

        # Save raw content
        if is_binary:
            with open(raw_path, "wb") as f_raw:
                f_raw.write(content)
        else:
            with open(raw_path, "w", encoding="utf-8", errors="replace") as f_raw:
                f_raw.write(content)

        # Parse content
        if is_binary:
            parsed_title, full_text, sections = parse_pdf_file(raw_path, fallback_title=title)
        else:
            parsed_title, full_text, sections = parse_html_content(content, fallback_title=title)

        parsed_data = {
            "id": sid,
            "area": src["area"],
            "title": parsed_title or title,
            "url": url,
            "tag": src["tag"],
            "format": fmt,
            "valid_for_term": src["valid_for_term"],
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "raw_file": raw_path,
            "char_count": len(full_text),
            "section_count": len(sections),
            "text": full_text,
            "sections": sections,
        }

        with open(parsed_path, "w", encoding="utf-8") as f_out:
            json.dump(parsed_data, f_out, indent=2, ensure_ascii=False)

        print(f"  [SAVED] {raw_path} & {parsed_path}")
        print(f"  Extracted {len(full_text)} characters across {len(sections)} sections.")
        scraped_count += 1

    print("\n" + "=" * 50)
    print(f"Scraping complete: {scraped_count} succeeded, {skipped_count} skipped/failed.")
    return scraped_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description="Polite scraper for PNW Navigator data sources")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sources to scrape")
    parser.add_argument("--source-id", type=str, default=None, help="Scrape a specific source ID (e.g. ISS-01)")
    parser.add_argument("--csv", type=str, default="sources.csv", help="Path to sources.csv")
    parser.add_argument("--force", action="store_true", help="Force re-scrape even if cache exists")
    args = parser.parse_args()

    scrape_sources(csv_path=args.csv, limit=args.limit, target_id=args.source_id, force=args.force)


if __name__ == "__main__":
    main()

