from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from neural_ngram import BetterNeuralNGramLM, NeuralNGramConfig


BASE_DIR = Path(__file__).resolve().parent
SENTENCEPIECE_MODEL = BASE_DIR / "sentencepiece.model"
MODEL_PATH = BASE_DIR / "neural_ngram_model.pt"


def sample_next_token(
    logits: torch.Tensor,
    top_k: int,
    temperature: float,
) -> int:
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)

    if 0 < top_k < probs.size(0):
        values, indices = torch.topk(probs, top_k)
        values = values / values.sum()
        sampled_index = torch.multinomial(values, num_samples=1).item()
        return indices[sampled_index].item()

    return torch.multinomial(probs, num_samples=1).item()


def load_model(model_path: Path, device: torch.device) -> tuple[BetterNeuralNGramLM, NeuralNGramConfig]:
    checkpoint = torch.load(model_path, map_location=device)
    config = NeuralNGramConfig(**checkpoint["config"])
    model = BetterNeuralNGramLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def generate_text(
    prompt: str,
    sp,
    model: BetterNeuralNGramLM,
    config: NeuralNGramConfig,
    max_tokens: int,
    top_k: int,
    temperature: float,
    device: torch.device,
) -> str:
    bos_token_id = config.bos_token_id
    if bos_token_id is None:
        raise ValueError("bos_token_id が設定されていません。学習済み設定を確認してください。")

    prompt_ids = sp.encode(prompt, out_type=int)
    generated_ids = prompt_ids[:]
    history = [bos_token_id] * config.context_size + prompt_ids

    for _ in range(max_tokens):
        context = history[-config.context_size:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0]
        next_token_id = sample_next_token(logits, top_k=top_k, temperature=temperature)
        generated_ids.append(next_token_id)
        history.append(next_token_id)

    return sp.decode(generated_ids)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="東京都内で運転を見合わせています")
    parser.add_argument("--sp-model", type=Path, default=SENTENCEPIECE_MODEL)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.sp_model))
    model, config = load_model(args.model_path, device)
    text = generate_text(
        prompt=args.prompt,
        sp=sp,
        model=model,
        config=config,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        device=device,
    )

    if args.json:
        print(json.dumps({"text": text, "context_size": config.context_size}, ensure_ascii=False))
        return

    print(text)


if __name__ == "__main__":
    main()
