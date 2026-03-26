# predict.py
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

text = "中島敦（なかじまあつし、1909年（明治42年）5月5日-1942年（昭和17年）12月4日）は、日本の小説家。代表作は『山月記』『光と風と夢』『弟子』『李陵』など。第一高等学校、東京帝国大学を卒業後、横浜高等女学校の教員勤務のかたわら小説執筆を続け、パラオ南洋庁の官吏（教科書編修書記）を経て専業作家になるも、同年中に持病の喘息悪化のため33歳で病没。死後に出版された全集は毎日出版文化賞を受賞した。"

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