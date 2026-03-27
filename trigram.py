import pickle
import sqlite3
from collections import defaultdict

import sentencepiece as spm


INPUT_FILE = "wiki.txt"
MODEL_FILE = "sentencepiece.model"
DB_FILE = "trigram_counts.sqlite"
OUTPUT_FILE = "trigram_model.pkl"
BATCH_SIZE = 5000


sp = spm.SentencePieceProcessor()
sp.Load(MODEL_FILE)


def normalize_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    line = line.replace("。", "").replace("、", "")
    line = line.replace("「", "").replace("」", "")
    line = line.replace("『", "").replace("』", "")
    return line.strip()


def create_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trigram_counts (
            w1 TEXT NOT NULL,
            w2 TEXT NOT NULL,
            w3 TEXT NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY (w1, w2, w3)
        )
        """
    )


def flush_batch(cur: sqlite3.Cursor, conn: sqlite3.Connection, batch: list[tuple[str, str, str, int]]) -> None:
    if not batch:
        return

    cur.executemany(
        """
        INSERT INTO trigram_counts (w1, w2, w3, cnt)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(w1, w2, w3)
        DO UPDATE SET cnt = cnt + excluded.cnt
        """,
        batch,
    )
    conn.commit()
    batch.clear()


def build_sqlite_counts() -> None:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    create_table(cur)
    conn.commit()

    batch: list[tuple[str, str, str, int]] = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = normalize_line(line)
            if not line:
                continue

            tokens = sp.encode(line, out_type=str)
            tokens = ["<BOS>", "<BOS>"] + tokens + ["<EOS>"]

            for i in range(len(tokens) - 2):
                batch.append((tokens[i], tokens[i + 1], tokens[i + 2], 1))

            if line_no % 1000 == 0:
                print(f"Processing line {line_no}")

            if len(batch) >= BATCH_SIZE:
                flush_batch(cur, conn, batch)

    flush_batch(cur, conn, batch)
    conn.close()
    print(f"sqlite counts saved to {DB_FILE}")


def export_pickle(output_file: str = OUTPUT_FILE) -> None:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    model = defaultdict(dict)

    for w1, w2, w3, cnt in cur.execute(
        """
        SELECT w1, w2, w3, cnt
        FROM trigram_counts
        """
    ):
        model[(w1, w2)][w3] = cnt

    conn.close()

    with open(output_file, "wb") as f:
        pickle.dump(dict(model), f)

    print(f"saved: {output_file}")


if __name__ == "__main__":
    build_sqlite_counts()
    export_pickle()
