"""Build a compact character n-gram frequency database for search scoring."""
import argparse
from collections import Counter
from pathlib import Path
import sqlite3
import tempfile
import time
import unicodedata


def build(corpus, output, max_lines=0, min_count=2, batch_lines=10000, n_values=(1, 2, 3)):
    if min_count < 1 or batch_lines < 1 or max_lines < 0:
        raise ValueError("invalid limits")
    corpus, output = Path(corpus), Path(output)
    if corpus.resolve() == output.resolve():
        raise ValueError("input and output must differ")
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        db = Path(tmp) / "build.sqlite"
        counts = Counter()
        totals = Counter()
        conn = sqlite3.connect(db)
        try:
            conn.executescript("""
                CREATE TABLE ngrams (
                    n INTEGER NOT NULL,
                    gram TEXT NOT NULL,
                    cnt INTEGER NOT NULL,
                    PRIMARY KEY(n, gram)
                ) WITHOUT ROWID;
                CREATE TABLE totals (
                    n INTEGER PRIMARY KEY,
                    cnt INTEGER NOT NULL
                );
            """)
            def flush():
                with conn:
                    conn.executemany("""
                        INSERT INTO ngrams VALUES (?, ?, ?)
                        ON CONFLICT(n, gram) DO UPDATE SET cnt=cnt+excluded.cnt
                    """, ((n, g, c) for (n, g), c in counts.items()))
                counts.clear()

            total_bytes = corpus.stat().st_size
            processed_bytes = processed_lines = 0
            started = time.monotonic()
            last_report = started

            def report():
                elapsed = max(time.monotonic() - started, 0.001)
                fraction = (processed_bytes / total_bytes if total_bytes else 1.0)
                if max_lines:
                    fraction = max(fraction, processed_lines / max_lines)
                fraction = min(fraction, 1.0)
                eta = elapsed * (1 - fraction) / fraction if fraction else 0
                print(f"Collecting: {fraction:6.2%} | {processed_lines:,} lines | "
                      f"{processed_lines / elapsed:,.0f} lines/s | "
                      f"ETA {eta / 60:.1f} min | DB {db.stat().st_size / 1048576:.1f} MiB",
                      flush=True)

            print("Collecting n-grams...", flush=True)
            with corpus.open("rb") as source:
                for line_no, raw in enumerate(source, 1):
                    if max_lines and line_no > max_lines:
                        break
                    processed_lines = line_no
                    processed_bytes += len(raw)
                    line = unicodedata.normalize("NFC", raw.decode("utf-8").strip())
                    # Do not count across whitespace or line boundaries.
                    for word in line.split():
                        for n in n_values:
                            totals[n] += max(0, len(word) - n + 1)
                            counts.update((n, word[i:i+n])
                                          for i in range(len(word)-n+1))
                    if line_no % batch_lines == 0 or len(counts) >= 500000:
                        flush()
                    if time.monotonic() - last_report >= 2:
                        report()
                        last_report = time.monotonic()
            flush()
            report()
            print("Collection complete. Filtering low-frequency n-grams...", flush=True)
            with conn:
                conn.executemany("INSERT INTO totals VALUES (?, ?)",
                                 ((n, totals[n]) for n in n_values))
                conn.execute("DELETE FROM ngrams WHERE cnt < ?", (min_count,))
                conn.execute("""
                    CREATE VIEW probabilities AS
                    SELECT g.n, g.gram, g.cnt,
                           CAST(g.cnt AS REAL)/t.cnt AS probability
                    FROM ngrams g JOIN totals t USING(n)
                """)
            print("Compacting SQLite (VACUUM)...", flush=True)
            conn.execute("VACUUM")
            print("Writing final database...", flush=True)
        finally:
            conn.close()
        # Exclusive creation: never overwrite an existing output.
        import os
        os.link(db, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("wiki.txt"))
    parser.add_argument("--output", type=Path, default=Path("search_ngrams.sqlite"))
    parser.add_argument("--max-lines", type=int, default=0,
                        help="0 means the entire corpus")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--batch-lines", type=int, default=10000)
    parser.add_argument("--n", type=int, choices=(1, 2, 3), nargs="+",
                        default=[1, 2, 3], help="n-gram sizes to collect")
    args = parser.parse_args()
    build(args.input, args.output, args.max_lines, args.min_count, args.batch_lines,
          tuple(sorted(set(args.n))))
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
