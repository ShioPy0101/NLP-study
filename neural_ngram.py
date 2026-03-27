from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class NeuralNGramConfig:
    vocab_size: int
    context_size: int = 4
    embed_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2
    bos_token_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BetterNeuralNGramLM(nn.Module):
    def __init__(self, config: NeuralNGramConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim * config.context_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb = emb.reshape(x.size(0), -1)
        return self.net(emb)
