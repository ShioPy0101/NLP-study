import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import sentencepiece as spm


BASE_DIR = Path(__file__).resolve().parent
SENTENCEPIECE_MODEL_PATH = BASE_DIR / "sentencepiece.model"
TRIGRAM_MODEL_PATH = BASE_DIR / "trigram_model.pkl"
BIGRAM_MODEL_PATH = BASE_DIR / "bigram_model.pkl"
TRIGRAM_PARTS_DIR = BASE_DIR / "trigram_parts"
TRIGRAM_SHARDS_DIR = BASE_DIR / "trigram_shards"
TRIGRAM_MANIFEST_PATH = TRIGRAM_SHARDS_DIR / "manifest.pkl"

BOS_ID = -1
EOS_ID = -2

BAD_TOKEN_TEXTS = {
    "<BOS>", "「", "」", "『", "』", "[", "]", "(", ")", "{", "}",
}

NOISY_TOKEN_TEXTS = {
    "「", "」", "『", "』", "[", "]", "(", ")", "{", "}",
    "A", "B", "C", ".", ",", ":", ";", "*", "=",
}


class ShardedTrigramModel:
    def __init__(self, shards_dir: Path, manifest_path: Path):
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)

        self.shards_dir = shards_dir
        self.num_shards = manifest["num_shards"]
        self.shard_files = manifest["shards"]
        self.cache: dict[int, dict[tuple[int, int], dict[int, int]]] = {}

    def _shard_index(self, context: tuple[int, int]) -> int:
        w1, w2 = context
        return ((w1 * 1000003) ^ w2) % self.num_shards

    def get(self, context: tuple[int, int]) -> dict[int, int] | None:
        shard_index = self._shard_index(context)
        shard = self.cache.get(shard_index)
        if shard is None:
            shard_path = self.shards_dir / self.shard_files[shard_index]
            with open(shard_path, "rb") as f:
                shard = pickle.load(f)
            self.cache[shard_index] = shard
        return shard.get(context)


def load_resources(
    sentencepiece_model_path: Path = SENTENCEPIECE_MODEL_PATH,
) -> tuple[spm.SentencePieceProcessor, object, str]:
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sentencepiece_model_path))

    if TRIGRAM_MANIFEST_PATH.exists():
        model = ShardedTrigramModel(TRIGRAM_SHARDS_DIR, TRIGRAM_MANIFEST_PATH)
        return sp, model, "sharded_trigram"

    if TRIGRAM_MODEL_PATH.exists() and TRIGRAM_MODEL_PATH.stat().st_size > 0:
        with open(TRIGRAM_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return sp, model, "trigram"

    if TRIGRAM_PARTS_DIR.exists() and any(TRIGRAM_PARTS_DIR.glob("part_*.pkl")):
        raise FileNotFoundError(
            " 先に `python3 merge_trigram_parts.py` を実行して trigram_shards/manifest.pkl を作ってください。"
        )

    with open(BIGRAM_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return sp, model, "bigram"


def is_ascii_fragment(token: str) -> bool:
    return all(ord(c) < 128 for c in token) and any(c.isalnum() for c in token)


def choose_next_token_from_counts(
    next_tokens: dict[int, int] | None,
    sp,
    top_k: int = 20,
    temperature: float = 0.95,
    recent_tokens: list[int] | None = None,
    repetition_window: int = 24,
    generated_len: int = 0,
    min_len_before_eos: int = 80,
) -> int | None:
    if not next_tokens:
        return None

    items = sorted(next_tokens.items(), key=lambda x: x[1], reverse=True)
    items = items[:top_k]

    recent_counts = Counter()
    if recent_tokens:
        recent_counts.update(recent_tokens[-repetition_window:])

    adjusted_items = []
    for token_id, count in items:
        if token_id == BOS_ID:
            continue

        weight = float(count)
        token_text = "<EOS>" if token_id == EOS_ID else sp.IdToPiece(token_id)

        if token_id == EOS_ID:
            if generated_len < min_len_before_eos:
                weight *= 0.03
            else:
                weight *= 0.5

        if recent_counts[token_id] > 0:
            weight /= (1.0 + recent_counts[token_id] * 2.4)

        if token_text in NOISY_TOKEN_TEXTS:
            weight *= 0.18

        if is_ascii_fragment(token_text):
            weight *= 0.25

        if len(token_text) == 1 and not token_text.isalnum() and token_text not in "。、！？":
            weight *= 0.2

        weight = max(weight, 1e-8)
        adjusted_items.append((token_id, weight ** (1.0 / temperature)))

    if not adjusted_items:
        return None

    token_ids = [token_id for token_id, _ in adjusted_items]
    weights = [weight for _, weight in adjusted_items]
    return random.choices(token_ids, weights=weights, k=1)[0]


def lookup_trigram_counts(model, context: tuple[int, int]) -> dict[int, int] | None:
    if hasattr(model, "get"):
        return model.get(context)
    return model.get(context)


def lookup_bigram_counts(model, prev_token: int, sp) -> dict[int, int] | None:
    if not model:
        return None

    # 旧 bigram モデルは文字列キーなので互換のためにここだけ piece に戻す。
    prev_piece = "<BOS>" if prev_token == BOS_ID else sp.IdToPiece(prev_token)
    next_tokens = model.get(prev_piece)
    if not next_tokens:
        return None

    converted: dict[int, int] = {}
    for token_text, count in next_tokens.items():
        if token_text == "<EOS>":
            token_id = EOS_ID
        elif token_text == "<BOS>":
            token_id = BOS_ID
        else:
            token_id = sp.PieceToId(token_text)
            if token_id < 0:
                continue
        converted[token_id] = count
    return converted


def generate_long_text(
    prompt: str,
    sp,
    model,
    model_type: str,
    max_tokens: int = 220,
    top_k: int = 20,
    temperature: float = 0.95,
    min_len_before_eos: int = 80,
) -> str:
    prompt_ids = sp.encode(prompt, out_type=int)
    generated_ids = prompt_ids[:]
    recent_generated: list[int] = []

    context_ids = [BOS_ID, BOS_ID] + prompt_ids

    for _ in range(max_tokens):
        next_counts = None

        if model_type in {"sharded_trigram", "trigram"}:
            context = (context_ids[-2], context_ids[-1]) if len(context_ids) >= 2 else (BOS_ID, BOS_ID)
            next_counts = lookup_trigram_counts(model, context)

        if next_counts is None:
            prev_token = context_ids[-1] if context_ids else BOS_ID
            next_counts = lookup_bigram_counts(model, prev_token, sp) if model_type == "bigram" else None

        if next_counts is None:
            break

        nxt = choose_next_token_from_counts(
            next_counts,
            sp,
            top_k=top_k,
            temperature=temperature,
            recent_tokens=recent_generated,
            generated_len=len(recent_generated),
            min_len_before_eos=min_len_before_eos,
        )

        if nxt is None or nxt == EOS_ID:
            break

        generated_ids.append(nxt)
        recent_generated.append(nxt)
        context_ids.append(nxt)

    return sp.decode(generated_ids)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="お客様に警戒警備のお願いです駅構内車内で不審なものを見かけましたらお近くの東京メトロ社員または",
    )
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--min-len-before-eos", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    sp, model, model_type = load_resources()
    print(f"モデルのロードが完了しました。 type={model_type}")
    text = generate_long_text(
        args.prompt,
        sp,
        model,
        model_type=model_type,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        min_len_before_eos=args.min_len_before_eos,
    )

    if args.json:
        print(json.dumps({"text": text, "model_type": model_type}, ensure_ascii=False))
        return

    print(text)


if __name__ == "__main__":
    main()
