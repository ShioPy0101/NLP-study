"""Build a compact character n-gram frequency database for search scoring."""
import argparse
from collections import Counter
from pathlib import Path
import sqlite3
import tempfile
import unicodedata


def build(corpus, output, max_lines=0, min_count=2, batch_lines=10000):
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

            with corpus.open(encoding="utf-8") as source:
                for line_no, line in enumerate(source, 1):
                    if max_lines and line_no > max_lines:
                        break
                    line = unicodedata.normalize("NFC", line.strip())
                    # Do not count across whitespace or line boundaries.
                    for word in line.split():
                        for n in (1, 2, 3):
                            totals[n] += max(0, len(word) - n + 1)
                            counts.update((n, word[i:i+n])
                                          for i in range(len(word)-n+1))
                    if line_no % batch_lines == 0 or len(counts) >= 500000:
                        flush()
            flush()
            with conn:
                conn.executemany("INSERT INTO totals VALUES (?, ?)",
                                 ((n, totals[n]) for n in (1, 2, 3)))
                conn.execute("DELETE FROM ngrams WHERE cnt < ?", (min_count,))
                conn.execute("""
                    CREATE VIEW probabilities AS
                    SELECT g.n, g.gram, g.cnt,
                           CAST(g.cnt AS REAL)/t.cnt AS probability
                    FROM ngrams g JOIN totals t USING(n)
                """)
            conn.execute("VACUUM")
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
    args = parser.parse_args()
    build(args.input, args.output, args.max_lines, args.min_count, args.batch_lines)
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
