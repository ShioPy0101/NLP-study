import sentencepiece as spm
from collections import defaultdict
import pickle

sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

model = defaultdict(lambda: defaultdict(int))

with open("wiki.txt", "r", encoding="utf-8") as f:
    current_line = 0
    all_lines = sum(1 for _ in open("wiki.txt", "r", encoding="utf-8"))
    for line in f:
        current_line += 1
        if current_line % 1000 == 0:
            print(f"Processing line {current_line}/{all_lines} ({current_line / all_lines:.2%})")

        line = line.strip()
        if not line:
            continue

        tokens = sp.encode(line, out_type=str)
        tokens = ["<BOS>", "<BOS>"] + tokens + ["<EOS>"]

        for i in range(len(tokens) - 2):
            context = (tokens[i], tokens[i + 1])
            next_token = tokens[i + 2]
            model[context][next_token] += 1

serializable_model = {
    context: dict(next_tokens)
    for context, next_tokens in model.items()
}

with open("trigram_model.pkl", "wb") as f:
    pickle.dump(serializable_model, f)
