
# predict.py
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

with open("vtokyo-raw.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        for i in range(len(line)):
            text = line[0:i+1]

            pieces = sp.EncodeAsPieces(text)
            ids = sp.EncodeAsIds(text)
            print("元文:")
            print(text)
            print()
            print("Piece分割:")
            print(pieces)
            print()
            print("ID列:")
            print(ids)
            print("-" * 50)