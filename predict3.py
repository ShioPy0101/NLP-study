import pickle
import random
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

with open("bigram_model.pkl", "rb") as f:
    model = pickle.load(f)

print("モデルのロードが完了しました。")

def predict_next_token_topk(
    prev_token: str,
    model: dict[str, dict[str, int]],
    top_k: int = 8,
    repeat_penalty_token: str | None = None,
) -> str | None:
    next_tokens = model.get(prev_token)
    if not next_tokens:
        return None

    items = sorted(next_tokens.items(), key=lambda x: x[1], reverse=True)

    if repeat_penalty_token is not None:
        adjusted = []
        for token, count in items:
            if token == repeat_penalty_token:
                adjusted.append((token, max(1, count // 3)))
            else:
                adjusted.append((token, count))
        items = adjusted

    items = items[:top_k]

    tokens = [token for token, _ in items]
    weights = [count for _, count in items]

    return random.choices(tokens, weights=weights, k=1)[0]

def generate_long_text(
    prompt: str,
    sp,
    model,
    max_tokens: int = 200,
    top_k: int = 8,
) -> str:
    pieces = sp.encode(prompt, out_type=str)

    if not pieces:
        prev = "<BOS>"
        generated = []
    else:
        prev = pieces[-1]
        generated = pieces[:]

    last_generated = None

    for _ in range(max_tokens):
        nxt = predict_next_token_topk(
            prev,
            model,
            top_k=top_k,
            repeat_penalty_token=last_generated,
        )

        if nxt is None or nxt == "<EOS>":
            break

        generated.append(nxt)
        last_generated = nxt
        prev = nxt

    generated = [p for p in generated if p not in ("<BOS>", "<EOS>")]
    return sp.decode(generated)

def generate_sentences(
    prompt: str,
    sp,
    model,
    sentence_count: int = 5,
    max_tokens_per_sentence: int = 50,
    top_k: int = 8,
) -> str:
    current_prompt = prompt
    sentences = []

    for _ in range(sentence_count):
        text = generate_long_text(
            current_prompt,
            sp,
            model,
            max_tokens=max_tokens_per_sentence,
            top_k=top_k,
        ).strip()

        if not text:
            break

        sentences.append(text)

        pieces = sp.encode(text, out_type=str)
        if not pieces:
            break
        current_prompt = sp.decode(pieces[-2:]) if len(pieces) >= 2 else text

    return "\n".join(sentences)


print(generate_sentences("お客様に警戒警備のお願いです駅構内車内で不審なものを見かけましたらお近くの東京メトロ社員または", sp, model, sentence_count=5, max_tokens_per_sentence=40))