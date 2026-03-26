# train.py
import sentencepiece as spm

# 学習
spm.SentencePieceTrainer.Train('--input=wiki.txt --model_prefix=sentencepiece --character_coverage=1.0 --vocab_size=32000 --input_sentence_size=200000')