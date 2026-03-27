from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from predict3 import (
    BASE_DIR,
    BIGRAM_MODEL_PATH,
    BOS_ID,
    EOS_ID,
    BigramLookupCache,
    build_cache_scores,
    is_cacheable_token,
    load_resources,
    lookup_trigram_counts,
    strip_symbols,
)


DEFAULT_OUTPUT_CSV = BASE_DIR / "predict3_tree.csv"


@dataclass
class TreeNode:
    node_id: int
    parent_id: int | None
    depth: int
    choice_rank: int
    token_id: int | None
    piece: str
    count: float
    score: float
    probability: float
    cumulative_score: float
    text: str
    is_terminal: bool
    context_ids: list[int]
    generated_ids: list[int]


SINGLE_CHAR_WORDS = {
    "が", "を", "に", "へ", "と", "で", "は", "も", "や", "か", "の", "な", "ね", "よ", "ぞ", "さ",
    "た", "だ", "し", "て", "で", "から", "まで",
}


def build_ranked_candidates(
    next_tokens: dict[int, int] | list[tuple[int, int]] | None,
    sp,
    cache_scores: dict[int, float] | None,
    cache_alpha: float,
    top_k: int,
    temperature: float,
    generated_len: int,
    min_len_before_eos: int,
) -> list[tuple[int, float, float]]:
    if not next_tokens:
        return []

    if isinstance(next_tokens, list):
        items = list(next_tokens)
    else:
        items = sorted(next_tokens.items(), key=lambda item: item[1], reverse=True)

    ranked: list[tuple[int, float, float]] = []
    for token_id, count in items:
        if token_id == BOS_ID:
            continue
        if token_id != EOS_ID and not is_cacheable_token(token_id, sp):
            continue

        weight = float(count)
        if token_id == EOS_ID:
            if generated_len < min_len_before_eos:
                weight *= 0.03
            else:
                weight *= 0.5

        if cache_scores:
            weight *= 1.0 + cache_alpha * cache_scores.get(token_id, 0.0)

        adjusted = max(weight, 1e-8) ** (1.0 / max(temperature, 1e-5))
        ranked.append((token_id, float(count), adjusted))
        if len(ranked) >= top_k:
            break

    total_weight = sum(weight for _, _, weight in ranked)
    if total_weight <= 0:
        return []

    return [
        (token_id, count, weight / total_weight)
        for token_id, count, weight in sorted(ranked, key=lambda item: item[2], reverse=True)
    ]


def get_next_counts(
    context_ids: list[int],
    model,
    model_type: str,
    top_k: int,
    get_bigram_lookup,
) -> dict[int, int] | list[tuple[int, int]] | None:
    next_counts = None
    if model_type in {"sqlite_trigram", "sharded_trigram", "trigram"}:
        context = (
            (context_ids[-2], context_ids[-1])
            if len(context_ids) >= 2
            else (BOS_ID, BOS_ID)
        )
        next_counts = lookup_trigram_counts(model, context, top_k)

    if not next_counts:
        prev_token = context_ids[-1] if context_ids else BOS_ID
        fallback_lookup = get_bigram_lookup()
        next_counts = fallback_lookup.get(prev_token) if fallback_lookup is not None else None

    return next_counts


def is_word_like_text(text: str) -> bool:
    if not text:
        return False
    if text in SINGLE_CHAR_WORDS:
        return True
    if len(text) >= 2:
        return True

    category = unicodedata.category(text[-1])
    return not (category.startswith("L") or category.startswith("N"))


def grow_branch_to_word(
    parent: TreeNode,
    first_token_id: int,
    first_count: float,
    first_probability: float,
    sp,
    model,
    model_type: str,
    top_k: int,
    temperature: float,
    min_len_before_eos: int,
    cache_alpha: float,
    cache_recent_window: int,
    get_bigram_lookup,
    max_subtokens_per_word: int,
) -> tuple[list[int], list[int], str, bool, float, float, float]:
    if first_token_id == EOS_ID:
        return (
            parent.generated_ids[:],
            parent.context_ids[:],
            "<EOS>",
            True,
            first_count,
            first_probability,
            parent.cumulative_score - math.log(max(first_probability, 1e-12)),
        )

    generated_ids = parent.generated_ids[:] + [first_token_id]
    context_ids = parent.context_ids[:] + [first_token_id]
    probability = first_probability
    count = first_count
    cumulative_score = parent.cumulative_score - math.log(max(first_probability, 1e-12))

    for _ in range(max_subtokens_per_word - 1):
        added_text = strip_symbols(sp.decode(generated_ids[len(parent.generated_ids):]))
        if is_word_like_text(added_text):
            return generated_ids, context_ids, added_text, False, count, probability, cumulative_score

        next_counts = get_next_counts(
            context_ids=context_ids,
            model=model,
            model_type=model_type,
            top_k=top_k,
            get_bigram_lookup=get_bigram_lookup,
        )
        if not next_counts:
            break

        cache_scores = build_cache_scores(context_ids, sp, recent_window=cache_recent_window)
        candidates = build_ranked_candidates(
            next_tokens=next_counts,
            sp=sp,
            cache_scores=cache_scores,
            cache_alpha=cache_alpha,
            top_k=top_k,
            temperature=temperature,
            generated_len=len(generated_ids) - len(parent.generated_ids),
            min_len_before_eos=min_len_before_eos,
        )
        if not candidates:
            break

        token_id, count, next_probability = candidates[0]
        if token_id == EOS_ID:
            break
        generated_ids.append(token_id)
        context_ids.append(token_id)
        probability *= next_probability
        cumulative_score -= math.log(max(next_probability, 1e-12))

    added_text = strip_symbols(sp.decode(generated_ids[len(parent.generated_ids):]))
    return generated_ids, context_ids, added_text, False, count, probability, cumulative_score


def expand_prediction_tree(
    prompt: str,
    sp,
    model,
    model_type: str,
    max_depth: int,
    branch_factor: int,
    top_k: int,
    temperature: float,
    min_len_before_eos: int,
    cache_alpha: float,
    cache_recent_window: int,
    max_subtokens_per_word: int,
) -> list[TreeNode]:
    prompt_ids = sp.encode(prompt, out_type=int)
    root = TreeNode(
        node_id=0,
        parent_id=None,
        depth=0,
        choice_rank=0,
        token_id=None,
        piece="<ROOT>",
        count=0.0,
        score=1.0,
        probability=1.0,
        cumulative_score=0.0,
        text=strip_symbols(sp.decode(prompt_ids)) if prompt_ids else strip_symbols(prompt),
        is_terminal=False,
        context_ids=[BOS_ID, BOS_ID] + prompt_ids,
        generated_ids=prompt_ids[:],
    )

    nodes: list[TreeNode] = [root]
    frontier = [root]
    next_node_id = 1
    bigram_lookup = BigramLookupCache(model, sp) if model_type == "bigram" else None

    def get_bigram_lookup() -> BigramLookupCache | None:
        nonlocal bigram_lookup
        if bigram_lookup is not None:
            return bigram_lookup
        if not BIGRAM_MODEL_PATH.exists():
            return None
        with open(BIGRAM_MODEL_PATH, "rb") as f:
            bigram_model = pickle.load(f)
        bigram_lookup = BigramLookupCache(bigram_model, sp)
        return bigram_lookup

    for depth in range(1, max_depth + 1):
        next_frontier: list[TreeNode] = []

        for parent in frontier:
            if parent.is_terminal:
                continue

            next_counts = get_next_counts(
                context_ids=parent.context_ids,
                model=model,
                model_type=model_type,
                top_k=top_k,
                get_bigram_lookup=get_bigram_lookup,
            )
            if not next_counts:
                continue

            cache_scores = build_cache_scores(parent.context_ids, sp, recent_window=cache_recent_window)
            candidates = build_ranked_candidates(
                next_tokens=next_counts,
                sp=sp,
                cache_scores=cache_scores,
                cache_alpha=cache_alpha,
                top_k=top_k,
                temperature=temperature,
                generated_len=parent.depth,
                min_len_before_eos=min_len_before_eos,
            )

            for rank, (token_id, count, probability) in enumerate(candidates[:branch_factor], start=1):
                generated_ids, context_ids, piece, is_terminal, count, probability, cumulative_score = grow_branch_to_word(
                    parent=parent,
                    first_token_id=token_id,
                    first_count=count,
                    first_probability=probability,
                    sp=sp,
                    model=model,
                    model_type=model_type,
                    top_k=top_k,
                    temperature=temperature,
                    min_len_before_eos=min_len_before_eos,
                    cache_alpha=cache_alpha,
                    cache_recent_window=cache_recent_window,
                    get_bigram_lookup=get_bigram_lookup,
                    max_subtokens_per_word=max_subtokens_per_word,
                )
                score = parent.score * probability

                node = TreeNode(
                    node_id=next_node_id,
                    parent_id=parent.node_id,
                    depth=depth,
                    choice_rank=rank,
                    token_id=token_id,
                    piece=piece,
                    count=count,
                    score=score,
                    probability=probability,
                    cumulative_score=cumulative_score,
                    text=strip_symbols(sp.decode(generated_ids)),
                    is_terminal=is_terminal,
                    context_ids=context_ids,
                    generated_ids=generated_ids,
                )
                nodes.append(node)
                next_node_id += 1

                if not is_terminal:
                    next_frontier.append(node)

        frontier = next_frontier
        if not frontier:
            break

    return nodes


def save_tree_to_csv(nodes: list[TreeNode], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token"])
        for node in nodes:
            if node.parent_id is None:
                writer.writerow([node.text])
                continue

            token = "" if node.token_id is None else strip_symbols(node.piece.replace("▁", ""))
            if node.is_terminal and not token:
                token = "<EOS>"
            writer.writerow([token])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="お客様に警戒警備のお願いです駅構内車内で不審なものを見かけましたらお近くの東京メトロ社員または",
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--min-len-before-eos", type=int, default=32)
    parser.add_argument("--cache-alpha", type=float, default=0.18)
    parser.add_argument("--cache-recent-window", type=int, default=48)
    parser.add_argument("--max-subtokens-per-word", type=int, default=4)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sp, model, model_type = load_resources()
    print(f"モデルのロードが完了しました。 type={model_type}")

    nodes = expand_prediction_tree(
        prompt=args.prompt,
        sp=sp,
        model=model,
        model_type=model_type,
        max_depth=args.max_depth,
        branch_factor=args.branch_factor,
        top_k=args.top_k,
        temperature=args.temperature,
        min_len_before_eos=args.min_len_before_eos,
        cache_alpha=args.cache_alpha,
        cache_recent_window=args.cache_recent_window,
        max_subtokens_per_word=args.max_subtokens_per_word,
    )
    save_tree_to_csv(nodes, args.output_csv)

    if args.json:
        print(
            json.dumps(
                {
                    "model_type": model_type,
                    "output_csv": str(args.output_csv),
                    "node_count": len(nodes),
                },
                ensure_ascii=False,
            )
        )
        return

    print(f"保存先: {args.output_csv}")
    print(f"ノード数: {len(nodes)}")


if __name__ == "__main__":
    main()
