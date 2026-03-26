import sentencepiece as spm
from collections import defaultdict

sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

model = defaultdict(lambda: defaultdict(int))

with open("wiki.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        tokens = sp.encode(line, out_type=str)

        # 文頭と文末を入れておくと少し扱いやすい
        tokens = ["<BOS>"] + tokens + ["<EOS>"]

        for i in range(len(tokens) - 1):
            model[tokens[i]][tokens[i + 1]] += 1