import argparse
import pickle
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PART_DIR = BASE_DIR / "trigram_parts"
SHARD_CHUNK_DIR = BASE_DIR / "trigram_shard_chunks"
SHARD_OUTPUT_DIR = BASE_DIR / "trigram_shards"

DEFAULT_NUM_SHARDS = 64
DEFAULT_WORKERS = 8
SHARD_LOG_INTERVAL = 50000


# このスクリプトは、既存の part_*.pkl をできるだけ速く合算するためのもの。
# SQLite を介さず、context をシャードに振り分けてから各シャードを reduce する。
# 単一の巨大 dict を最後に作らないので、元の merge より落ちにくい。


def format_progress(current: int, total: int) -> str:
    if total <= 0:
        return f"{current}"
    return f"{current}/{total} ({current / total:.1%})"


def shard_index_for_context(context: tuple[int, int], num_shards: int) -> int:
    w1, w2 = context
    return ((w1 * 1000003) ^ w2) % num_shards


def merge_next_tokens(dst: dict[int, int], src: dict[int, int]) -> None:
    for token_id, count in src.items():
        dst[token_id] = dst.get(token_id, 0) + count


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def partition_part_file(
    part_path: Path,
    shard_chunk_dir: Path,
    num_shards: int,
) -> list[Path]:
    # 各 part を読み込み、context ごとに shard へ振り分けて保存する。
    # 同じファイルへ複数 worker で追記すると pickle が壊れるので、
    # part ごと・shard ごとに一意なファイル名へ書く。
    with open(part_path, "rb") as f:
        partial = pickle.load(f)

    shard_buckets: dict[int, dict[tuple[int, int], dict[int, int]]] = {}
    context_total = len(partial)

    for context_index, (context, next_tokens) in enumerate(partial.items(), start=1):
        shard_index = shard_index_for_context(context, num_shards)
        bucket = shard_buckets.setdefault(shard_index, {})
        existing = bucket.get(context)
        if existing is None:
            bucket[context] = dict(next_tokens)
        else:
            merge_next_tokens(existing, next_tokens)

        if context_index % SHARD_LOG_INTERVAL == 0 or context_index == context_total:
            print(
                f"partition {part_path.name}: {format_progress(context_index, context_total)}"
            )

    written_paths: list[Path] = []
    for shard_index, bucket in shard_buckets.items():
        shard_chunk_path = shard_chunk_dir / f"{part_path.stem}.shard_{shard_index:03d}.pkl"
        with open(shard_chunk_path, "wb") as f:
            pickle.dump(bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
        written_paths.append(shard_chunk_path)

    return written_paths


def partition_all_parts(
    part_dir: Path,
    shard_chunk_dir: Path,
    num_shards: int,
    workers: int,
    reset: bool,
) -> list[Path]:
    if reset:
        reset_dir(shard_chunk_dir)
    else:
        shard_chunk_dir.mkdir(parents=True, exist_ok=True)

    part_files = sorted(part_dir.glob("part_*.pkl"))
    if not part_files:
        raise FileNotFoundError(f"no part files found in {part_dir}")

    print(f"partitioning {len(part_files)} parts into {num_shards} shards")

    if workers <= 1:
        for index, part_path in enumerate(part_files, start=1):
            print(f"start partition {format_progress(index, len(part_files))}: {part_path.name}")
            partition_part_file(part_path, shard_chunk_dir, num_shards)
        return sorted(shard_chunk_dir.glob("*.shard_*.pkl"))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(partition_part_file, part_path, shard_chunk_dir, num_shards)
            for part_path in part_files
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            print(f"done partition {format_progress(index, len(part_files))}")

    return sorted(shard_chunk_dir.glob("*.shard_*.pkl"))


def reduce_shard_group(
    shard_index: int,
    shard_chunk_paths: list[Path],
    shard_output_dir: Path,
) -> Path:
    # 1 shard 分だけ合算するので、必要メモリは全体よりかなり小さい。
    merged = defaultdict(dict)
    chunk_total = len(shard_chunk_paths)

    for chunk_index, shard_chunk_path in enumerate(shard_chunk_paths, start=1):
        with open(shard_chunk_path, "rb") as f:
            chunk = pickle.load(f)

        for context, next_tokens in chunk.items():
            existing = merged.get(context)
            if existing is None:
                merged[context] = dict(next_tokens)
            else:
                merge_next_tokens(existing, next_tokens)

        if chunk_index % 10 == 0 or chunk_index == chunk_total:
            print(
                f"reduce shard_{shard_index:03d}: {format_progress(chunk_index, chunk_total)}"
            )

    output_path = shard_output_dir / f"shard_{shard_index:03d}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dict(merged), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved shard: {output_path}")
    return output_path


def reduce_all_shards(
    shard_chunk_dir: Path,
    shard_output_dir: Path,
    workers: int,
    reset: bool,
) -> list[Path]:
    if reset:
        reset_dir(shard_output_dir)
    else:
        shard_output_dir.mkdir(parents=True, exist_ok=True)

    shard_chunk_files = sorted(shard_chunk_dir.glob("*.shard_*.pkl"))
    if not shard_chunk_files:
        raise FileNotFoundError(f"no shard chunk files found in {shard_chunk_dir}")

    shard_groups: dict[int, list[Path]] = defaultdict(list)
    for path in shard_chunk_files:
        shard_index = int(path.stem.split(".shard_")[1])
        shard_groups[shard_index].append(path)

    shard_items = sorted(shard_groups.items())
    print(f"reducing {len(shard_items)} shards from {len(shard_chunk_files)} shard chunk files")

    if workers <= 1:
        shard_paths = []
        for index, (shard_index, shard_chunk_paths) in enumerate(shard_items, start=1):
            print(f"start reduce {format_progress(index, len(shard_items))}: shard_{shard_index:03d}")
            shard_paths.append(reduce_shard_group(shard_index, shard_chunk_paths, shard_output_dir))
        return shard_paths

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(reduce_shard_group, shard_index, shard_chunk_paths, shard_output_dir)
            for shard_index, shard_chunk_paths in shard_items
        ]
        shard_paths = []
        for index, future in enumerate(futures, start=1):
            shard_paths.append(future.result())
            print(f"done reduce {format_progress(index, len(shard_items))}")
        return sorted(shard_paths)


def write_manifest(shard_paths: list[Path], manifest_path: Path) -> Path:
    manifest = {
        "format": "trigram-sharded-v1",
        "num_shards": len(shard_paths),
        "shards": [path.name for path in sorted(shard_paths)],
    }
    with open(manifest_path, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"manifest saved: {manifest_path}")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-dir", type=Path, default=PART_DIR)
    parser.add_argument("--shard-chunk-dir", type=Path, default=SHARD_CHUNK_DIR)
    parser.add_argument("--shard-output-dir", type=Path, default=SHARD_OUTPUT_DIR)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-partition", action="store_true")
    parser.add_argument("--skip-reduce", action="store_true")
    parser.add_argument("--keep-shard-chunks", action="store_true")
    parser.add_argument("--keep-shard-outputs", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.skip_partition:
        partition_all_parts(
            part_dir=args.part_dir,
            shard_chunk_dir=args.shard_chunk_dir,
            num_shards=args.num_shards,
            workers=args.workers,
            reset=not args.keep_shard_chunks,
        )

    if not args.skip_reduce:
        shard_paths = reduce_all_shards(
            shard_chunk_dir=args.shard_chunk_dir,
            shard_output_dir=args.shard_output_dir,
            workers=args.workers,
            reset=not args.keep_shard_outputs,
        )
        write_manifest(shard_paths, args.shard_output_dir / "manifest.pkl")


if __name__ == "__main__":
    main()
