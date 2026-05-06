# reddit-history-retrieval
Reddit thread recommendation system for historical queries. Given a natural language question, it retrieves and ranks relevant discussions (focused on r/AskHistorians) using lexical (TF-IDF/BM25) and semantic (embeddings) methods, evaluated with IR metrics to compare performance.

### Setup
Use one shared virtual environment at the repository root.

1. Create and activate the environment (Python 3.12):

```powershell
cd "path\reddit-history-retrieval"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3. Verify interpreter version:

```powershell
python -c "import sys; print(sys.version.split()[0])"
```

Expected output: `3.12.x`

Notes:
- Do not create `search_engine/.venv`; keep only `.venv` at repo root.
- Use the activated root environment for all Python work in this project.

### Download + Quick Validation (AskHistorians)
Note: if you would like to skip downloading/processing the data on your own machine, feel free to use our finished data! The link is in the [preprocessing section](#preprocessing).

Run from `search_engine` after activating the root environment.

```powershell
cd "path\reddit-history-retrieval\search_engine"
New-Item -ItemType Directory -Force -Path ..\data\raw | Out-Null
python -c "from convokit import download; p=download('subreddit-AskHistorians', data_dir=r'..\data\raw'); print(p)"
```

Validate the corpus:

```powershell
python -c "from pathlib import Path; p=Path('..')/'data'/'raw'/'subreddit-AskHistorians'/'utterances.jsonl'; print(sum(1 for _ in p.open('r', encoding='utf-8')))"
python -c "import json; from pathlib import Path; p=Path('..')/'data'/'raw'/'subreddit-AskHistorians'/'utterances.jsonl'; print(json.loads(p.open('r', encoding='utf-8').readline()).keys())"
```

Expected result:
- The first command prints a large line count (about 2 million).
- The second command prints keys including `id`, `user`, `root`, `reply_to`, `timestamp`, `text`, `meta`.

Carrying forward, all next commands below assume:
- You created/activated the root `.venv` (see Setup above)
- You are running from the repository root unless stated otherwise

### Preprocessing
This produces:
- `data/processed/threads.csv`
- `data/processed/threads_preprocessed.csv`
- `data/processed/topic_labels.csv`

Note that preprocessing prepares a large file (can be 1-2 GB) and can take over an hour.
If you would like to instead download the CSV, please [click here](https://drive.google.com/drive/folders/1EJ3oJgJsaZpWx1gWF0aLedisRLFc1iIs?usp=sharing). If you decide to use our file, please place the file inside `/data/processed/` after downloading. The full path should be `reddit-history-retrieval/data/processed/threads_preprocessed.csv`. The topic labels list is also included, which is needed for the topic modeling viewing script. Please insert it into the same directory (`reddit-history-retrieval/data/processed/topic_labels.csv`).

If you would like to run the preprocessing yourself, please use the following command usage:
```powershell
python -m search_engine.preprocess --build-threads-csv `
  --input "data\processed\threads.csv" `
  --num-topics 100 `
  --passes 5 `
  --chunk-size 2000
```

Arguments:
- `--build-threads-csv`: build `threads.csv` from the downloaded raw corpus
- `--input`: path to `threads.csv` (used for preprocessing + topic training)
- `--num-topics`: number of LDA topics (default 100)
- `--passes`: number of LDA training passes (default 5; each pass takes a long time if running on a personal machine)
- `--chunk-size`: docs per update chunk (default 2000)

### Retrieval

#### BM25 (lexical)
Demo: `python -m search_engine.text_retrieval`

Usage:
```powershell
python -m search_engine.text_retrieval --query "roman empire collapse" --top-k 10 `
  --input "data\processed\threads_preprocessed.csv"
```

Arguments:
- `--query`: query text (if omitted, runs a 5-query demo)
- `--top-k`: number of results to print (top and bottom; default 10)
- `--input`: path to `threads_preprocessed.csv` (this is the default; may be changed to a different file)

#### Word2Vec (semantic)
Demo: `python -m search_engine.word2vec_retrieval.py`

Usage:
```powershell
python -m search_engine.word2vec_retrieval --query "roman empire collapse" --top-k 10 `
  --input "data\processed\threads_preprocessed.csv" `
  --model "glove-wiki-gigaword-50"
```

Arguments:
- `--query`: query text (if omitted, runs a 5-query demo)
- `--top-k`: number of results to print (default 10)
- `--input`: path to `threads_preprocessed.csv` (this is the default; may be changed to a different file)
- `--model`: gensim model key (default `glove-wiki-gigaword-50`)

### Topic Modeling
Topic training happens during preprocessing (see above). Use this script to explore the trained topics.

List topics:

```powershell
python search_engine\topic_modeling.py topics
```

List documents for a topic id or label:

```powershell
python search_engine\topic_modeling.py docs 12
python search_engine\topic_modeling.py docs "rome_empire_western"
```
