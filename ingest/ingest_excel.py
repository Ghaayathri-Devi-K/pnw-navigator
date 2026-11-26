import os
import re
import sys
import time
from dotenv import load_dotenv
import openpyxl
import psycopg


def clean_str(val):
    if val is None:
        return None
    val_str = str(val).strip()
    return val_str if val_str else None


def ingest_sources(cur, ws):
    print("Ingesting 'Source Register' into 'sources' table...")
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    count = 0
    for r in rows:
        if not r or not r[0]:
            continue
        source_id = clean_str(r[0])
        # Skip notes / footer rows
        if not source_id or not re.match(r"^[A-Z]{2,4}-\d+", source_id):
            print(f"  [Skipping footer/legend row in Source Register]: {source_id[:60]}...")
            continue

        area = clean_str(r[1]) if len(r) > 1 else None
        page = clean_str(r[2]) if len(r) > 2 else None
        url = clean_str(r[3]) if len(r) > 3 else None
        owner = clean_str(r[4]) if len(r) > 4 else None
        what_to_extract = clean_str(r[5]) if len(r) > 5 else None
        fmt = clean_str(r[6]) if len(r) > 6 else None
        tag = clean_str(r[7]) if len(r) > 7 else None
        refresh = clean_str(r[8]) if len(r) > 8 else None
        read_status = clean_str(r[9]) if len(r) > 9 else None
        milestone = clean_str(r[10]) if len(r) > 10 else None

        cur.execute(
            """
            INSERT INTO sources (
                id, area, page, url, owner, what_to_extract, format, tag, refresh, read_status, milestone
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                area = EXCLUDED.area,
                page = EXCLUDED.page,
                url = EXCLUDED.url,
                owner = EXCLUDED.owner,
                what_to_extract = EXCLUDED.what_to_extract,
                format = EXCLUDED.format,
                tag = EXCLUDED.tag,
                refresh = EXCLUDED.refresh,
                read_status = EXCLUDED.read_status,
                milestone = EXCLUDED.milestone;
            """,
            (
                source_id,
                area,
                page,
                url,
                owner,
                what_to_extract,
                fmt,
                tag,
                refresh,
                read_status,
                milestone,
            ),
        )
        count += 1
    print(f"  -> Ingested {count} sources.")
    return count


def ingest_key_facts(cur, ws):
    print("Ingesting 'Key Facts' into 'key_facts' table...")
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    count = 0
    for r in rows:
        if not r or not r[0]:
            continue
        fact_id = clean_str(r[0])
        # Skip notes / footer rows
        if not fact_id or not re.match(r"^F-\d+", fact_id):
            print(f"  [Skipping footer/legend row in Key Facts]: {fact_id[:60]}...")
            continue

        topic = clean_str(r[1]) if len(r) > 1 else None
        tag = clean_str(r[2]) if len(r) > 2 else None
        fact = clean_str(r[3]) if len(r) > 3 else None
        if not fact:
            continue
        valid_for = clean_str(r[4]) if len(r) > 4 else None
        source_id = clean_str(r[5]) if len(r) > 5 else None
        source_url = clean_str(r[6]) if len(r) > 6 else None
        verify_note = clean_str(r[7]) if len(r) > 7 else None

        cur.execute(
            """
            INSERT INTO key_facts (
                id, topic, tag, fact, valid_for, source_id, source_url, verify_note
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                topic = EXCLUDED.topic,
                tag = EXCLUDED.tag,
                fact = EXCLUDED.fact,
                valid_for = EXCLUDED.valid_for,
                source_id = EXCLUDED.source_id,
                source_url = EXCLUDED.source_url,
                verify_note = EXCLUDED.verify_note;
            """,
            (
                fact_id,
                topic,
                tag,
                fact,
                valid_for,
                source_id,
                source_url,
                verify_note,
            ),
        )
        count += 1
    print(f"  -> Ingested {count} key facts.")
    return count


def ingest_flags_and_gaps(cur, ws):
    print("Ingesting 'Flags & Gaps' into 'flags_and_gaps' table...")
    cur.execute("TRUNCATE TABLE flags_and_gaps RESTART IDENTITY;")
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    count = 0
    for r in rows:
        if not r or not any(r):
            continue
        issue = clean_str(r[0]) if len(r) > 0 else None
        what_i_found = clean_str(r[1]) if len(r) > 1 else None
        what_to_do = clean_str(r[2]) if len(r) > 2 else None
        if not (issue or what_i_found or what_to_do):
            continue

        cur.execute(
            """
            INSERT INTO flags_and_gaps (issue, what_i_found, what_to_do)
            VALUES (%s, %s, %s);
            """,
            (issue, what_i_found, what_to_do),
        )
        count += 1
    print(f"  -> Ingested {count} flags & gaps.")
    return count


def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in environment.", file=sys.stderr)
        sys.exit(1)

    excel_file = "PNW_Navigator_Data_Sources.xlsx"
    if not os.path.exists(excel_file):
        print(f"ERROR: Excel file '{excel_file}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading '{excel_file}'...")
    wb = openpyxl.load_workbook(excel_file, data_only=True)

    # Retry connection
    conn = None
    for attempt in range(1, 3):
        try:
            conn = psycopg.connect(db_url)
            break
        except Exception as e:
            if attempt == 2:
                print(f"Failed to connect: {e}", file=sys.stderr)
                sys.exit(1)
            print("Retrying connection in 3 seconds...")
            time.sleep(3)

    try:
        with conn.cursor() as cur:
            if "Source Register" in wb.sheetnames:
                ingest_sources(cur, wb["Source Register"])

            if "Key Facts" in wb.sheetnames:
                ingest_key_facts(cur, wb["Key Facts"])

            if "Flags & Gaps" in wb.sheetnames:
                ingest_flags_and_gaps(cur, wb["Flags & Gaps"])

            conn.commit()

            print("\nDatabase Row Counts Summary:")
            for tbl in ["sources", "key_facts", "flags_and_gaps", "documents", "chunks"]:
                cur.execute(f"SELECT COUNT(*) FROM {tbl};")
                cnt = cur.fetchone()[0]
                print(f"  - {tbl}: {cnt} rows")

            print("\nData ingestion completed successfully!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
