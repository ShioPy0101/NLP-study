import argparse
import os
import pickle
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sentencepiece as spm


# このスクリプトは、wiki.txt から trigram モデルを学習する。
# 学習は map-reduce 型で行い、
# 1. 入力をチャンク単位に分割
# 2. 各チャンクごとに部分 trigram を集計して pickle 保存
# 3. 最後に全 partial を単一プロセスでマージ
# という流れで進める。

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "wiki.txt"
MODEL_FILE = BASE_DIR / "sentencepiece.model"
PART_DIR = BASE_DIR / "trigram_parts"
OUTPUT_FILE = BASE_DIR / "trigram_model.pkl"
SQLITE_FILE = BASE_DIR / "trigram_counts.sqlite"

# SentencePiece の通常 ID と衝突しない予約値を文頭・文末に使う。
BOS_ID = -1
EOS_ID = -2
DEFAULT_LINES_PER_CHUNK = 100000
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 1)
SQLITE_BATCH_SIZE = 5000
SCAN_LOG_INTERVAL = 10000
CHUNK_LOG_INTERVAL = 20000


def normalize_line(line: str) -> str:
    # 空白や不要な記号を落として、学習対象の文字列を整える。
    line = line.strip()
    if not line:
        return ""

    # 記号を軽く落としてから SentencePiece に渡す。
    line = line.replace("。", "").replace("、", "")
    line = line.replace("「", "").replace("」", "")
    line = line.replace("『", "").replace("』", "")
    return line.strip()


def model_to_dict(model) -> dict[tuple[int, int], dict[int, int]]:
    # defaultdict のままだと保存後の扱いがやや不安定なので、
    # pickle 前に通常の dict へ変換しておく。
    return {
        context: dict(next_tokens)
        for context, next_tokens in model.items()
    }


def format_progress(current: int, total: int) -> str:
    # ログ表示用に "現在/総数 (割合)" の形へ整形する。
    if total <= 0:
        return f"{current}"
    return f"{current}/{total} ({current / total:.1%})"


def process_chunk(chunk_index: int, lines: list[str], model_file: str, part_dir: str) -> str:
    # 各 worker 内で SentencePiece をロードする。
    # プロセス間で Processor を共有しないほうが扱いやすい。
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)

    # 各 worker は独立に部分 trigram を数えて pickle に保存する。
    model = defaultdict(lambda: defaultdict(int))
    total_lines = len(lines)

    for line_index, line in enumerate(lines, start=1):
        # 文字列ではなく整数 ID で学習することで、
        # メモリ使用量とキー比較コストを抑える。
        token_ids = sp.encode(line, out_type=int)
        token_ids = [BOS_ID, BOS_ID] + token_ids + [EOS_ID]

        # (前々語, 前語) -> 次語 の頻度を加算する。
        for i in range(len(token_ids) - 2):
            context = (token_ids[i], token_ids[i + 1])
            next_token = token_ids[i + 2]
            model[context][next_token] += 1

        if line_index % CHUNK_LOG_INTERVAL == 0 or line_index == total_lines:
            print(f"chunk {chunk_index:04d}: {format_progress(line_index, total_lines)}")

    part_path = Path(part_dir) / f"part_{chunk_index:04d}.pkl"
    # partial ごとに保存しておくことで、
    # 途中で落ちてもチャンク単位で再利用しやすい。
    with open(part_path, "wb") as f:
        pickle.dump(model_to_dict(model), f)

    return str(part_path)


def chunked_lines(input_file: Path, lines_per_chunk: int):
    # 入力テキストを一定行数ごとに区切って worker へ渡す。
    chunk: list[str] = []
    chunk_index = 0
    valid_line_count = 0
    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))

    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = normalize_line(raw_line)
            if not line:
                continue

            chunk.append(line)
            valid_line_count += 1

            if line_no % SCAN_LOG_INTERVAL == 0:
                print(
                    f"scan: {format_progress(line_no, total_lines)}, valid lines {valid_line_count}"
                )

            if len(chunk) >= lines_per_chunk:
                print(f"prepared chunk {chunk_index:04d}: {len(chunk)} normalized lines")
                yield chunk_index, chunk
                chunk_index += 1
                chunk = []

    if chunk:
        print(f"prepared chunk {chunk_index:04d}: {len(chunk)} normalized lines")
        yield chunk_index, chunk


def build_partial_models(
    input_file: Path = INPUT_FILE,
    model_file: Path = MODEL_FILE,
    part_dir: Path = PART_DIR,
    lines_per_chunk: int = DEFAULT_LINES_PER_CHUNK,
    workers: int = DEFAULT_WORKERS,
) -> list[Path]:
    # partial pickle の出力先ディレクトリを用意する。
    part_dir.mkdir(exist_ok=True)

    # 先に chunk 一覧を作ることで、全体件数ベースの進捗を出せるようにする。
    chunk_jobs = list(chunked_lines(input_file, lines_per_chunk))
    if not chunk_jobs:
        return []

    print(f"building {len(chunk_jobs)} chunks with workers={workers}")
    part_paths: list[Path] = []

    if workers <= 1:
        # worker=1 のときはシンプルに逐次処理する。
        for job_index, (chunk_index, lines) in enumerate(chunk_jobs, start=1):
            print(f"starting chunk {format_progress(job_index, len(chunk_jobs))}: part_{chunk_index:04d}")
            part_path = process_chunk(chunk_index, lines, str(model_file), str(part_dir))
            print(f"saved partial {format_progress(job_index, len(chunk_jobs))}: {part_path}")
            part_paths.append(Path(part_path))
        return part_paths

    # 複数 worker のときは各 chunk を独立ジョブとして投げる。
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_chunk, chunk_index, lines, str(model_file), str(part_dir))
            for chunk_index, lines in chunk_jobs
        ]
        for completed_index, future in enumerate(futures, start=1):
            part_path = future.result()
            print(f"saved partial {format_progress(completed_index, len(chunk_jobs))}: {part_path}")
            part_paths.append(Path(part_path))

    return sorted(part_paths)


def merge_partial_models(
    part_dir: Path = PART_DIR,
    output_file: Path = OUTPUT_FILE,
) -> Path:
    # 最後の merge だけは単一プロセスで行い、結果を安定させる。
    # trigram の集計は単なる加算なので、partial を順に足せば元の結果と一致する。
    merged = defaultdict(lambda: defaultdict(int))
    part_files = sorted(part_dir.glob("part_*.pkl"))
    print(f"merging {len(part_files)} partial files")

    for index, part_path in enumerate(part_files, start=1):
        print(f"merging {format_progress(index, len(part_files))}: {part_path.name}")
        with open(part_path, "rb") as f:
            partial = pickle.load(f)

        # 各 partial の頻度を最終モデルへ足し込む。
        for context, next_tokens in partial.items():
            for token_id, count in next_tokens.items():
                merged[context][token_id] += count

    with open(output_file, "wb") as f:
        pickle.dump(model_to_dict(merged), f)

    print(f"merged model saved to {output_file}")
    return output_file


def configure_sqlite(conn: sqlite3.Connection) -> None:
    # SQLite は最終保存先として使う前提で軽量な設定にする。
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")


def export_pickle_to_sqlite(
    pickle_file: Path = OUTPUT_FILE,
    sqlite_file: Path = SQLITE_FILE,
    batch_size: int = SQLITE_BATCH_SIZE,
) -> Path:
    # 必要なら最終 pickle を SQLite へ移し替える。
    # 学習中に DB を共有更新するのではなく、最後の保存先としてだけ使う。
    with open(pickle_file, "rb") as f:
        model = pickle.load(f)

    total_contexts = len(model)
    print(f"exporting {total_contexts} contexts to sqlite")

    conn = sqlite3.connect(sqlite_file, timeout=30.0)
    configure_sqlite(conn)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trigram_counts (
            w1 INTEGER NOT NULL,
            w2 INTEGER NOT NULL,
            w3 INTEGER NOT NULL,
            cnt INTEGER NOT NULL,
            PRIMARY KEY (w1, w2, w3)
        )
        """
    )
    # 再出力時は既存内容を入れ替える。
    cur.execute("DELETE FROM trigram_counts")

    batch: list[tuple[int, int, int, int]] = []
    for context_index, ((w1, w2), next_tokens) in enumerate(model.items(), start=1):
        for w3, cnt in next_tokens.items():
            batch.append((w1, w2, w3, cnt))
            if len(batch) >= batch_size:
                # 小さすぎる commit を避けるため、ある程度まとめて書き込む。
                cur.executemany(
                    """
                    INSERT INTO trigram_counts (w1, w2, w3, cnt)
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()
                print(f"sqlite export: {format_progress(context_index, total_contexts)}")

    if batch:
        cur.executemany(
            """
            INSERT INTO trigram_counts (w1, w2, w3, cnt)
            VALUES (?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()
        print(f"sqlite export: {format_progress(total_contexts, total_contexts)}")

    conn.close()
    print(f"sqlite model saved to {sqlite_file}")
    return sqlite_file


def build_arg_parser() -> argparse.ArgumentParser:
    # 通常実行は build + merge。
    # 必要なら merge だけ、SQLite 出力付き、のように切り替えられる。
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines-per-chunk", type=int, default=DEFAULT_LINES_PER_CHUNK)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--export-sqlite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # 部分モデルの作成。
    if not args.skip_build:
        build_partial_models(
            lines_per_chunk=args.lines_per_chunk,
            workers=args.workers,
        )

    # partial のマージ。
    if not args.skip_merge:
        merge_partial_models()

    # 必要なときだけ SQLite へ変換。
    if args.export_sqlite:
        export_pickle_to_sqlite()


if __name__ == "__main__":
    main()
