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
python -m pip install -r .\search_engine\requirements.txt
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

### Current Goal
1. Retrieval target: thread-level ranking (title + post + top comments merged into one document).
2. Primary dataset: Cornell Reddit October 2018, AskHistorians first.
3. Baseline model: BM25.
4. Semantic model: one sentence-transformer + FAISS.
5. Metrics: Precision@5, Precision@10, MRR.
6. Milestone 1 goal: “One query returns top 10 threads through web app from local index.”

### Potential API Endpoint

POST /search
Input: query, top_k, method (bm25 or dense)
Output: ranked list with thread_id, title, subreddit, score, snippet, url
