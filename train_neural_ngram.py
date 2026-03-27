from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_ngram import BetterNeuralNGramLM, NeuralNGramConfig


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "wiki.txt"
SENTENCEPIECE_MODEL = BASE_DIR / "sentencepiece.model"
OUTPUT_MODEL = BASE_DIR / "neural_ngram_model.pt"
OUTPUT_CONFIG = BASE_DIR / "neural_ngram_config.json"


def get_piece_size(sp) -> int:
    if hasattr(sp, "GetPieceSize"):
        return sp.GetPieceSize()
    return sp.get_piece_size()


def build_samples(
    input_file: Path,
    sp,
    context_size: int,
    max_lines: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bos_token_id = get_piece_size(sp)
    contexts: list[list[int]] = []
    targets: list[int] = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_index, raw_line in enumerate(f, start=1):
            if max_lines is not None and line_index > max_lines:
                break

            line = raw_line.strip()
            if not line:
                continue

            token_ids = sp.encode(line, out_type=int)
            if not token_ids:
                continue

            history = [bos_token_id] * context_size + token_ids
            for target_index in range(context_size, len(history)):
                contexts.append(history[target_index - context_size:target_index])
                targets.append(history[target_index])

            if line_index % 1000 == 0:
                print(f"sample build: line {line_index}, samples={len(targets)}")

    x = torch.tensor(contexts, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    return x, y


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.sp_model))

    x, y = build_samples(
        input_file=args.input,
        sp=sp,
        context_size=args.context_size,
        max_lines=args.max_lines,
    )
    if len(y) == 0:
        raise ValueError("学習サンプルが 0 件です。入力ファイルまたは SentencePiece モデルを確認してください。")

    config = NeuralNGramConfig(
        vocab_size=get_piece_size(sp) + 1,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        bos_token_id=get_piece_size(sp),
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BetterNeuralNGramLM(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"train start: samples={len(dataset)} context={config.context_size} "
        f"embed={config.embed_dim} hidden={config.hidden_dim} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_index, (batch_x, batch_y) in enumerate(loader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

            if batch_index % args.log_interval == 0:
                print(
                    f"epoch {epoch}/{args.epochs} "
                    f"batch {batch_index}/{len(loader)} "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataset)
        print(f"epoch {epoch}/{args.epochs} avg_loss={avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
        args.output_model,
    )
    with open(args.output_config, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"saved model: {args.output_model}")
    print(f"saved config: {args.output_config}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--sp-model", type=Path, default=SENTENCEPIECE_MODEL)
    parser.add_argument("--output-model", type=Path, default=OUTPUT_MODEL)
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-lines", type=int)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
