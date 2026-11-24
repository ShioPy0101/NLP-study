"""Build a character-bigram inverted index of Japanese Wikipedia article wikitext without storing bodies."""
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
import re
from collections import defaultdict, Counter
from itertools import islice


CS_CATEGORY = re.compile(r'\[\[\s*(?:Category|カテゴリ)\s*:\s*([^|\]\n]+)', re.IGNORECASE)
CS_TERMS = ('情報科学', '計算機科学', 'コンピュータ', 'プログラミング',
            'ソフトウェア', 'アルゴリズム', 'データ構造', 'データベース',
            '人工知能', '機械学習', '自然言語処理', '情報工学',
            '情報セキュリティ', 'オペレーティングシステム', '計算理論',
            'コンパイラ', '計算複雑性', '分散コンピューティング', '情報理論')


def is_cs_article(title, body):
    return any(term in value for value in (title, *CS_CATEGORY.findall(body))
               for term in CS_TERMS)


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
            yield (int(elem.findtext(ns + "id")), elem.findtext(ns + "title") or "",
                   elem.findtext(ns + "revision/" + ns + "text") or "")
        root.remove(elem)


def append_varint(out, value):
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def decode_ids(blob):
    value = shift = previous = 0
    for byte in blob:
        value |= (byte & 127) << shift
        if byte & 128:
            shift += 7
            if shift >= 64:
                raise ValueError("invalid varint")
        else:
            previous += value
            yield previous
            value = shift = 0
    if shift:
        raise ValueError("truncated varint")


def intersect(a, b):
    a, b = iter(a), iter(b)
    x, y = next(a, None), next(b, None)
    while x is not None and y is not None:
        if x == y:
            yield x
            x, y = next(a, None), next(b, None)
        elif x < y:
            x = next(a, None)
        else:
            y = next(b, None)


def build(input_path, output_path, limit=0, batch_size=1000):
    input_path, output_path = Path(input_path), Path(output_path)
    if limit < 0 or batch_size < 1:
        raise ValueError("invalid limit or batch size")
    if output_path.exists():
        raise FileExistsError(output_path)
    total_bytes = input_path.stat().st_size
    started = last_report = time.monotonic()
    count = scanned = 0
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        db = Path(tmp) / "index.sqlite"
        conn = sqlite3.connect(db)
        try:
            conn.executescript("""
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY,
                    wiki_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL
                );
                CREATE TABLE postings (
                    gram TEXT PRIMARY KEY,
                    doc_count INTEGER NOT NULL,
                    doc_ids BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE chunks (
                    gram TEXT NOT NULL,
                    batch INTEGER NOT NULL,
                    doc_count INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    PRIMARY KEY (gram, batch)
                ) WITHOUT ROWID;
            """)
            docs = []
            postings = defaultdict(bytearray)
            previous = {}
            frequencies = Counter()
            batch = pending = 0

            def flush():
                nonlocal batch, pending
                with conn:
                    conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", docs)
                    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                                     ((g, batch, frequencies[g], blob)
                                      for g, blob in postings.items()))
                docs.clear()
                postings.clear()
                frequencies.clear()
                pending = 0
                batch += 1

            print("Building article body bigram inverted index...", flush=True)
            with input_path.open("rb") as raw:
                source = bz2.BZ2File(raw) if input_path.suffix == ".bz2" else raw
                try:
                    for doc_id, title, body in pages(source):
                        scanned += 1
                        now = time.monotonic()
                        if now - last_report >= 2:
                            fraction = min(1, raw.tell() / max(1, total_bytes))
                            elapsed = max(now - started, 0.001)
                            eta = elapsed * (1 - fraction) / fraction if fraction else 0
                            print(f"{fraction:.1%} | scanned {scanned:,} | indexed {count:,} | "
                                  f"ETA {eta/60:.1f} min | DB {db.stat().st_size/1048576:.1f} MiB", flush=True)
                            last_report = now
                        if not is_cs_article(title, body):
                            continue
                        url = "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(
                            title.replace(" ", "_"), safe="")
                        title = unicodedata.normalize("NFC", title)
                        dense_id = count
                        docs.append((dense_id, doc_id, title, url))
                        for g in bigrams(body):
                            append_varint(postings[g], dense_id - previous.get(g, 0))
                            previous[g] = dense_id
                            frequencies[g] += 1
                            pending += 1
                        count += 1
                        if len(docs) >= batch_size or pending >= 200000:
                            flush()
                        if limit and count >= limit:
                            break
                finally:
                    if source is not raw:
                        source.close()
            flush()
            print("Merging compressed posting chunks...", flush=True)
            # Chunks already encode deltas across batch boundaries, so concatenate
            # in batch order without decoding or storing uncompressed pairs.
            from itertools import groupby
            cursor = conn.execute("SELECT gram,doc_count,data FROM chunks ORDER BY gram,batch")
            merged = 0
            for gram, rows in groupby(cursor, key=lambda row: row[0]):
                blob = bytearray()
                frequency = 0
                for _, n, data in rows:
                    blob.extend(data)
                    frequency += n
                conn.execute("INSERT INTO postings VALUES (?,?,?)",
                             (gram, frequency, blob))
                merged += 1
                if merged % 10000 == 0:
                    print(f"Merged {merged:,} grams", flush=True)
            cursor.close()
            conn.execute("DROP TABLE chunks")
            conn.commit()
            print("Compacting database...", flush=True)
            conn.execute("VACUUM")
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
        raise ValueError("query must contain at least two characters")
    streams = []
    for gram in grams:
        row = conn.execute("SELECT doc_count,doc_ids FROM postings WHERE gram=?", (gram,)).fetchone()
        if row is None:
            return []
        streams.append(row)
    streams.sort(key=lambda row: row[0])
    ids = decode_ids(streams[0][1])
    for _, blob in streams[1:]:
        ids = intersect(ids, decode_ids(blob))
    return [conn.execute("SELECT id,title,url FROM documents WHERE id=?", (doc_id,)).fetchone()
            for doc_id in islice(ids, limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("jawiki-latest-pages-articles.xml"))
    parser.add_argument("--output", type=Path, default=Path("bigram_cs_compressed.sqlite"))
    parser.add_argument("--limit", type=int, default=0, help="0: all matching computer science articles")
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
