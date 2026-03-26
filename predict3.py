import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import sentencepiece as spm


BASE_DIR = Path(__file__).resolve().parent
SENTENCEPIECE_MODEL_PATH = BASE_DIR / "sentencepiece.model"
BIGRAM_MODEL_PATH = BASE_DIR / "bigram_model.pkl"

BAD_TOKENS = {
    "<BOS>", "「", "」", "『", "』", "[", "]", "(", ")", "{", "}",
}

# 先頭や途中で出ると不自然になりやすいものを弱める
NOISY_TOKENS = {
    "「", "」", "『", "』", "[", "]", "(", ")", "{", "}",
    "A", "B", "C", ".", ",", ":", ";", "*", "=",
}


def load_resources(
    sentencepiece_model_path: Path = SENTENCEPIECE_MODEL_PATH,
    bigram_model_path: Path = BIGRAM_MODEL_PATH,
):
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sentencepiece_model_path))

    with open(bigram_model_path, "rb") as f:
        model = pickle.load(f)

    return sp, model


def is_ascii_fragment(token: str) -> bool:
    # 英数字だけの細切れトークンを弱める
    return all(ord(c) < 128 for c in token) and any(c.isalnum() for c in token)


def choose_next_token(
    prev_token: str,
    model: dict[str, dict[str, int]],
    top_k: int = 20,
    temperature: float = 0.95,
    recent_tokens: list[str] | None = None,
    repetition_window: int = 24,
    generated_len: int = 0,
    min_len_before_eos: int = 80,
) -> str | None:
    next_tokens = model.get(prev_token)
    if not next_tokens:
        return None

    items = sorted(next_tokens.items(), key=lambda x: x[1], reverse=True)
    items = items[:top_k]

    recent_counts = Counter()
    if recent_tokens:
        recent_counts.update(recent_tokens[-repetition_window:])

    adjusted_items = []
    for token, count in items:
        if token in BAD_TOKENS:
            continue

        weight = float(count)

        # ある程度の長さになるまでは EOS をかなり抑制
        if token == "<EOS>":
            if generated_len < min_len_before_eos:
                weight *= 0.03
            else:
                weight *= 0.5

        # 最近使ったトークンは強めに減衰
        if recent_counts[token] > 0:
            weight /= (1.0 + recent_counts[token] * 2.4)

        # 記号や英数字断片を少し抑える
        if token in NOISY_TOKENS:
            weight *= 0.18

        if is_ascii_fragment(token):
            weight *= 0.25

        # 単独記号っぽいものも抑える
        if len(token) == 1 and not token.isalnum() and token not in "。、！？":
            weight *= 0.2

        # 温度適用
        weight = max(weight, 1e-8)
        adjusted_items.append((token, weight ** (1.0 / temperature)))

    if not adjusted_items:
        return None

    tokens = [token for token, _ in adjusted_items]
    weights = [weight for _, weight in adjusted_items]
    return random.choices(tokens, weights=weights, k=1)[0]


def generate_long_text(
    prompt: str,
    sp,
    model,
    max_tokens: int = 220,
    top_k: int = 20,
    temperature: float = 0.95,
    min_len_before_eos: int = 80,
) -> str:
    prompt_pieces = sp.encode(prompt, out_type=str)
    generated_pieces = prompt_pieces[:]
    recent_generated: list[str] = []

    prev = prompt_pieces[-1] if prompt_pieces else "<BOS>"

    for _ in range(max_tokens):
        nxt = choose_next_token(
            prev,
            model,
            top_k=top_k,
            temperature=temperature,
            recent_tokens=recent_generated,
            generated_len=len(recent_generated),
            min_len_before_eos=min_len_before_eos,
        )

        if nxt is None or nxt == "<EOS>":
            break

        generated_pieces.append(nxt)
        recent_generated.append(nxt)
        prev = nxt

    generated_pieces = [p for p in generated_pieces if p not in ("<BOS>", "<EOS>")]
    return sp.decode(generated_pieces)


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
    sp, model = load_resources()
    print("モデルのロードが完了しました。")
    text = generate_long_text(
        args.prompt,
        sp,
        model,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        min_len_before_eos=args.min_len_before_eos,
    )

    if args.json:
        print(json.dumps({"text": text}, ensure_ascii=False))
        return

    print(text)


if __name__ == "__main__":
    main()
