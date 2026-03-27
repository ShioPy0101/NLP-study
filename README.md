# NLP-study

SentencePiece で分割した日本語テキストから、bigram / trigram の簡易言語モデルや neural n-gram を作って文章生成する実験用リポジトリです。現状は `wiki.txt` を入力にして学習し、`predict3.py` または HTTP サーバー経由で推論できます。

## 主なファイル

- `train.py`: `wiki.txt` から `sentencepiece.model` / `sentencepiece.vocab` を学習
- `train2.py`: `sentencepiece.model` を使って `bigram_model.pkl` を作成
- `trigram.py`: `wiki.txt` から trigram の partial を作成し、必要に応じて `trigram_model.pkl` と `trigram_counts.sqlite` を出力
- `merge_trigram_parts.py`: `trigram_parts/` をシャード分割し、`trigram_shards/` と `manifest.pkl` を作成
- `predict.py`: SentencePiece の分割確認用
- `predict2.py`: `vtokyo-raw.txt` を 1 文字ずつ伸ばしながら分割確認
- `predict3.py`: 文章生成のメイン CLI
- `neural_ngram.py`: `context_size` 可変の neural n-gram 本体
- `train_neural_ngram.py`: neural 5-gram / 7-gram などの学習
- `predict_neural_ngram.py`: 学習済み neural n-gram で生成
- `server.py`: ローカル HTTP API
- `client.py`: HTTP API クライアント

## 依存

Python 3 系と `sentencepiece`, `torch` が必要です。

```bash
python3 -m pip install sentencepiece torch
```

## 学習の流れ

### 1. SentencePiece モデル作成

```bash
python3 train.py
```

生成物:

- `sentencepiece.model`
- `sentencepiece.vocab`

### 2. bigram モデル作成

```bash
python3 train2.py
```

生成物:

- `bigram_model.pkl`

### 3. trigram モデル作成

`trigram.py` は `wiki.txt` をチャンクに分けて `trigram_parts/part_*.pkl` を作り、その後にマージします。

```bash
python3 trigram.py --workers 8 --lines-per-chunk 100000
```

主なオプション:

- `--workers`: 並列 worker 数
- `--lines-per-chunk`: 1 partial あたりの行数
- `--skip-build`: 既存 `trigram_parts/` を使って build を省略
- `--skip-merge`: partial 作成のみ実行
- `--export-sqlite`: `trigram_model.pkl` から `trigram_counts.sqlite` を出力

生成物:

- `trigram_parts/part_*.pkl`
- `trigram_model.pkl`
- `trigram_counts.sqlite` (`--export-sqlite` 指定時)

### 4. trigram partial をシャード化

巨大な `trigram_model.pkl` を避けたい場合は、`merge_trigram_parts.py` で `trigram_parts/` からシャードを作ります。

```bash
python3 merge_trigram_parts.py --workers 8 --num-shards 64
```

主な生成物:

- `trigram_shard_chunks/`
- `trigram_shards/shard_*.pkl`
- `trigram_shards/manifest.pkl`

主なオプション:

- `--skip-partition`: shard chunk 作成を省略
- `--skip-reduce`: reduce を省略
- `--keep-shard-chunks`: `trigram_shard_chunks/` を消さずに再利用
- `--keep-shard-outputs`: `trigram_shards/` を消さずに再利用

### 5. neural n-gram を学習

`train_neural_ngram.py` は fixed-length の直前文脈から次トークンを当てる MLP ベースの言語モデルです。デフォルトでは `context_size=4` なので、neural 5-gram 相当になります。

```bash
python3 train_neural_ngram.py \
  --context-size 4 \
  --embed-dim 128 \
  --hidden-dim 256 \
  --dropout 0.2 \
  --epochs 3 \
  --batch-size 256
```

主な生成物:

- `neural_ngram_model.pt`
- `neural_ngram_config.json`

主なオプション:

- `--context-size`: 直前何トークンを入力に使うか
- `--max-lines`: 学習行数を制限して小さく試す
- `--device`: `cpu`, `cuda` などを明示

### 6. neural n-gram で生成

```bash
python3 predict_neural_ngram.py "東京都内で運転を見合わせています" \
  --max-tokens 120 \
  --top-k 12 \
  --temperature 0.9
```

## 推論

### CLI

```bash
python3 predict3.py "東京都内で運転を見合わせています"
```

オプション例:

```bash
python3 predict3.py "東京都内で運転を見合わせています" \
  --max-tokens 220 \
  --top-k 20 \
  --temperature 0.95 \
  --min-len-before-eos 80
```

`predict3.py` のモデル選択順は次の通りです。

1. `trigram_shards/manifest.pkl` があれば sharded trigram を使用
2. `trigram_model.pkl` があれば単一 pickle の trigram を使用
3. trigram partial だけ存在する場合は、先に `python3 merge_trigram_parts.py` の実行を要求
4. それ以外は `bigram_model.pkl` を使用

### HTTP サーバー

起動:

```bash
python3 server.py
```

疎通確認:

```bash
curl http://127.0.0.1:8000/health
```

生成:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"東京都内で運転を見合わせています","max_tokens":220,"top_k":20,"temperature":0.95,"min_len_before_eos":80}'
```

クライアント経由:

```bash
python3 client.py "東京都内で運転を見合わせています"
```

JSON で受ける場合:

```bash
python3 client.py "東京都内で運転を見合わせています" --json
```

## 補助スクリプト

- `predict.py`: 任意文字列を SentencePiece で piece / id に分解
- `predict2.py`: `vtokyo-raw.txt` の各行を先頭から 1 文字ずつ伸ばしながら piece / id を確認

## 現在の大きめ生成物

ローカルには次のような大きいファイルがあります。

- `wiki.txt`
- `jawiki-latest-pages-articles.xml`
- `bigram_model.pkl`
- `trigram_counts.sqlite`
- `trigram_parts/`
- `trigram_shard_chunks/`
- `trigram_shards/`

必要に応じて README 更新時点の実ファイル構成に合わせて整理してください。
