"""Build a character-bigram inverted index of Japanese Wikipedia article titles."""
import argparse
from pathlib import Path
import bz2
import sqlite3
import tempfile
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
import os


def bigrams(text):
    text = unicodedata.normalize("NFC", text)
    return {text[i:i+2] for i in range(len(text)-1)}


def pages(source):
    # Remove processed page elements so the XML tree does not grow with the dump.
    events = ET.iterparse(source, events=("start", "end"))
    _, root = next(events)
    ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
    for event, elem in events:
        if event != "end" or elem.tag != ns + "page":
            continue
        if elem.findtext(ns + "ns") == "0" and elem.find(ns + "redirect") is None:
            yield int(elem.findtext(ns + "id")), elem.findtext(ns + "title") or ""
        root.remove(elem)


def build(input_path, output_path, limit=0, batch_size=1000):
    input_path, output_path = Path(input_path), Path(output_path)
    if limit < 0 or batch_size < 1:
        raise ValueError("invalid limit or batch size")
    if output_path.exists():
        raise FileExistsError(output_path)
    total_bytes = input_path.stat().st_size
    started = last_report = time.monotonic()
    count = 0
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        db = Path(tmp) / "index.sqlite"
        conn = sqlite3.connect(db)
        try:
            conn.executescript("""
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL
                );
                CREATE TABLE postings (
                    gram TEXT NOT NULL,
                    doc_id INTEGER NOT NULL REFERENCES documents(id),
                    PRIMARY KEY (gram, doc_id)
                ) WITHOUT ROWID;
            """)
            docs, postings = [], []

            def flush():
                with conn:
                    conn.executemany("INSERT INTO documents VALUES (?, ?, ?)", docs)
                    conn.executemany("INSERT INTO postings VALUES (?, ?)", postings)
                docs.clear()
                postings.clear()

            print("Building title bigram inverted index...", flush=True)
            with input_path.open("rb") as raw:
                source = bz2.BZ2File(raw) if input_path.suffix == ".bz2" else raw
                try:
                    for doc_id, title in pages(source):
                        url = "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(
                            title.replace(" ", "_"), safe="")
                        title = unicodedata.normalize("NFC", title)
                        docs.append((doc_id, title, url))
                        postings.extend((g, doc_id) for g in sorted(bigrams(title)))
                        count += 1
                        if len(docs) >= batch_size:
                            flush()
                        now = time.monotonic()
                        if now - last_report >= 2:
                            fraction = min(1, raw.tell() / max(1, total_bytes))
                            if limit:
                                fraction = max(fraction, count / limit)
                            elapsed = max(now - started, 0.001)
                            eta = elapsed * (1 - fraction) / fraction if fraction else 0
                            print(f"{fraction:.1%} | {count:,} articles | "
                                  f"{count/elapsed:,.0f} articles/s | "
                                  f"ETA {eta/60:.1f} min | "
                                  f"DB {db.stat().st_size/1048576:.1f} MiB", flush=True)
                            last_report = now
                        if limit and count >= limit:
                            break
                finally:
                    if source is not raw:
                        source.close()
            flush()
            print("Finalizing index...", flush=True)
            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()
        os.link(db, output_path)
    print(f"Done: {count:,} articles, {output_path} "
          f"({output_path.stat().st_size/1048576:.1f} MiB)", flush=True)


def search(conn, query, limit=20):
    query = unicodedata.normalize("NFC", query)
    if not query:
        return []
    grams = sorted(bigrams(query))
    if not grams:
        return conn.execute(
            "SELECT id,title,url FROM documents WHERE instr(title,?)>0 ORDER BY id LIMIT ?",
            (query, limit)).fetchall()
    # Intersect posting lists, then verify the exact substring.
    placeholders = ",".join("?" for _ in grams)
    return conn.execute(f"""
        SELECT d.id,d.title,d.url
        FROM documents d JOIN (
            SELECT doc_id FROM postings WHERE gram IN ({placeholders})
            GROUP BY doc_id HAVING COUNT(*) = ?
        ) p ON p.doc_id=d.id
        WHERE instr(d.title,?)>0 ORDER BY d.id LIMIT ?
    """, (*grams, len(grams), query, limit)).fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("jawiki-latest-pages-articles.xml"))
    parser.add_argument("--output", type=Path, default=Path("bigram_index.sqlite"))
    parser.add_argument("--limit", type=int, default=0, help="0: all articles")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--query", help="search an existing --output database instead")
    args = parser.parse_args()
    if args.query is not None:
        with sqlite3.connect(args.output.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            for doc_id, title, url in search(conn, args.query):
                print(f"{doc_id}\t{title}\t{url}")
    else:
        build(args.input, args.output, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
