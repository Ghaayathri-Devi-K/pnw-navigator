---
description: Domain, scraping, chunking, and retrieval rules for PNW Navigator (RAG assistant for international students at Purdue University Northwest)
---

# PNW Navigator Project Rules

## 1. Domain & Scope Boundaries
- **Strict Allow-List**: Ingest only official pages from `*.pnw.edu` and official student insurance portal `*.pnw.myahpcare.com`.
- **Prohibited Sources**: Do NOT scrape or index pages from Purdue University West Lafayette (`purdue.edu`), PNWCC, PNWU, or external unofficial housing/immigration sites. Discrepancies between campuses can lead to severe visa/status violations for international students.
- **Milestone 1 Scope**: Focus on International Student Services (ISS), University Housing, Course Registration, and Student Health Insurance.

## 2. Polite Scraping Standards
- **Pacing**: Enforce a minimum delay of 1.0 second between consecutive HTTP requests.
- **Identification**: Use a polite and identifiable User-Agent: `PNWNavigatorBot/1.0 (+https://www.pnw.edu/international-student-services/)`.
- **Resilience**: Implement exponential backoff for transient network issues or rate limits.
- **Data Persistence**: Store raw downloads under `data/raw/` and parsed structured content under `data/parsed/`.

## 3. Heading-Aware Chunking
- **Boundary Respect**: Chunks must respect document structure (`h1`, `h2`, `h3`, markdown section markers). Never split sentences or logical bullet lists across arbitrary boundaries when avoidable.
- **Model Token Limit**: Chunks must fit within the context length of `all-MiniLM-L6-v2` (maximum 256 tokens / ~800–1000 characters), with ~100 characters overlap.
- **Metadata Preservation**: Every chunk must carry: `source_id`, `chunk_index`, `title`, `url`, `section_heading`, and `content`.

## 4. Embedding & Database Standards
- **Model**: `all-MiniLM-L6-v2` generating 384-dimensional vectors.
- **Vector Normalization**: All embeddings must be L2-normalized ($\|v\|_2 = 1.0$) prior to persistence and search.
- **Distance Metric**: Use cosine distance (`vector_cosine_ops` in pgvector: `<=>` operator).
- **Idempotency**: Data ingestion into `documents` and `chunks` must support `--dry-run` and safe upserts.

## 5. Security & Accuracy
- **Zero Secret Exposure**: Never print, log, or hardcode connection strings, database credentials, or API keys.
- **Strict Grounding**: Retrieval must support citation verification back to specific document sources and URLs.

