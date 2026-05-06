from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from search_engine.text_retrieval import TextRetrieval

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_UTTERANCES_PATH = REPO_ROOT / "data" / "raw" / "subreddit-AskHistorians" / "utterances.jsonl"
RAW_CONVERSATIONS_PATH = REPO_ROOT / "data" / "raw" / "subreddit-AskHistorians" / "conversations.json"

THREADS_CSV_PATH = REPO_ROOT / "data" / "processed" / "threads.csv"
THREADS_PREPROCESSED_PATH = REPO_ROOT / "data" / "processed" / "threads_preprocessed.csv"
TOPIC_LABELS_PATH = REPO_ROOT / "data" / "processed" / "topic_labels.csv"

DEFAULT_NUM_TOPICS = 100
DEFAULT_LDA_PASSES = 5
DEFAULT_LDA_CHUNK_SIZE = 2000


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def build_threads_csv(
    *,
    input_path: Path = RAW_UTTERANCES_PATH,
    conversation_path: Path = RAW_CONVERSATIONS_PATH,
    output_path: Path = THREADS_CSV_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with conversation_path.open("r", encoding="utf-8") as f:
        conversations = json.load(f)

    threads: dict[str, dict] = {}

    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="reading utterances", unit="line"):
            row = json.loads(line)

            text = row.get("text", "").replace("\u2028", " ").replace("\u2029", " ").strip()
            if text == "":
                continue

            thread_id = row.get("root")
            reply_to = row.get("reply_to")
            meta = row.get("meta", {})
            permalink = meta.get("permalink", "")

            title = conversations.get(thread_id, {}).get("title", "")

            if thread_id not in threads:
                threads[thread_id] = {
                    "thread_id": thread_id,
                    "title": title,
                    "content_parts": [],
                    "url": "",
                }

            if reply_to is None and permalink:
                threads[thread_id]["url"] = "https://reddit.com" + permalink

            if text:
                threads[thread_id]["content_parts"].append(text)

    rows = []
    for thread in tqdm(list(threads.values()), desc="assembling threads", unit="thread"):
        content = " ".join(thread["content_parts"]).strip()
        if thread["title"] == "" and content == "":
            continue

        rows.append(
            {
                "thread_id": thread["thread_id"],
                "title": thread["title"],
                "content": content,
                "url": thread["url"],
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["thread_id", "title", "content", "url"])
        w.writeheader()
        w.writerows(rows)

    print("Saved", len(rows), "threads to", output_path)
    return output_path


def _simple_tokens(text: str) -> list[str]:
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split(" ") if len(t) > 2]


def preprocess_threads(input_path: Path = THREADS_CSV_PATH) -> pd.DataFrame:
    """
    Preprocess the `content` field into a normalized token string.

    We keep this aligned with `TextRetrieval` (same punctuation/stopwords),
    but do the loop here so we can show a progress bar.
    """
    tr = TextRetrieval()  # provides punctuations + stopwords setup

    df = pd.read_csv(input_path, low_memory=False).fillna("")
    if df.shape[0] == 0:
        return df

    punctuations = tr.punctuations
    stop_words = tr.stop_words
    digits = "0123456789"

    prepro_content: list[str] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="preprocessing", unit="thread"):
        line = str(getattr(row, "content", ""))

        new_line = ""
        i = 0
        while i < len(line):
            if line[i] == "<":
                next_close_idx = line.find(">", i)
                if next_close_idx != -1:
                    next_open_idx = line.find("<", i + 1, next_close_idx)
                    if next_open_idx == -1:
                        i = next_close_idx + 1
                        continue
            new_line += line[i]
            i += 1

        words = new_line.split()
        updated_words: list[str] = []
        for w in words:
            if w == "":
                continue
            w = w.lower()
            for p in punctuations:
                w = w.replace(p, "")
            for d in digits:
                w = w.replace(d, "")
            if w and w not in stop_words:
                updated_words.append(w)

        prepro_content.append(" ".join(updated_words))

    df["content"] = prepro_content
    return df


def train_topics_on_preprocessed(
    df: pd.DataFrame,
    *,
    num_topics: int = DEFAULT_NUM_TOPICS,
    lda_passes: int = DEFAULT_LDA_PASSES,
    chunk_size: int = DEFAULT_LDA_CHUNK_SIZE,
) -> tuple[list[int], list[float], pd.DataFrame]:
    import gensim
    from gensim import corpora
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser

    if "title" not in df.columns or "content" not in df.columns:
        raise ValueError("Expected columns: title, content")

    full_texts = ((df["title"].fillna("") + " ") * 3 + df["content"].fillna("")).tolist()
    tokenized_docs = [_simple_tokens(t) for t in tqdm(full_texts, desc="tokenizing", unit="doc")]

    bigram = Phrases(tokenized_docs, min_count=10, threshold=20)
    bigram_phraser = Phraser(bigram)
    tokenized_docs = [bigram_phraser[doc] for doc in tqdm(tokenized_docs, desc="applying bigrams", unit="doc")]

    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.4)

    corpus = [dictionary.doc2bow(doc) for doc in tqdm(tokenized_docs, desc="building corpus", unit="doc")]

    lda_passes = int(lda_passes)
    chunk_size = int(chunk_size)
    num_topics = int(num_topics)

    ldamodel = gensim.models.ldamodel.LdaModel(
        id2word=dictionary,
        num_topics=num_topics,
        alpha="auto",
        eta="auto",
        chunksize=chunk_size,
        passes=1,
        minimum_probability=0.0,
        random_state=42,
    )

    num_docs = len(corpus)
    num_chunks = (num_docs + chunk_size - 1) // chunk_size if chunk_size > 0 else 0
    for p in range(1, lda_passes + 1):
        with tqdm(total=num_chunks, desc=f"LDA pass {p}/{lda_passes}", unit="chunk") as pbar:
            for start in range(0, num_docs, chunk_size):
                end = min(start + chunk_size, num_docs)
                ldamodel.update(corpus[start:end])
                pbar.update(1)

    def _dominant_topic(bow):
        dist = ldamodel.get_document_topics(bow, minimum_probability=0.0)
        return max(dist, key=lambda x: x[1])

    dominant_topics: list[int] = []
    confidences: list[float] = []
    for bow in tqdm(corpus, total=num_docs, desc="assigning dominant topic", unit="doc"):
        topic_id, confidence = _dominant_topic(bow)
        dominant_topics.append(int(topic_id))
        confidences.append(float(confidence))

    labels = []
    for topic_id in range(num_topics):
        top_words = [w for w, _ in ldamodel.show_topic(topic_id, topn=3)]
        labels.append({"topic_id": topic_id, "label": "_".join(top_words)})
    labels_df = pd.DataFrame(labels)

    return dominant_topics, confidences, labels_df


def run(
    *,
    input_path: Path = THREADS_CSV_PATH,
    write_outputs: bool = True,
    num_topics: int = DEFAULT_NUM_TOPICS,
    lda_passes: int = DEFAULT_LDA_PASSES,
    chunk_size: int = DEFAULT_LDA_CHUNK_SIZE,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = preprocess_threads(input_path=input_path)

    topics, confidences, labels_df = train_topics_on_preprocessed(
        df, num_topics=num_topics, lda_passes=lda_passes, chunk_size=chunk_size
    )

    df["topic_confidence"] = confidences
    df["topic"] = topics  # must be the LAST column

    if write_outputs:
        THREADS_PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(THREADS_PREPROCESSED_PATH, index=False)
        labels_df.to_csv(TOPIC_LABELS_PATH, index=False)

    return df, labels_df


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_on_path()

    p = argparse.ArgumentParser(
        prog="prepare_data.py",
        description="Build threads.csv (optional), preprocess threads, and train LDA topics.",
    )
    p.add_argument("--build-threads-csv", action="store_true", help="Build data/processed/threads.csv from raw JSONL.")
    p.add_argument("--input", default=str(THREADS_CSV_PATH), help="Path to threads.csv input (after build, if used).")
    p.add_argument("--num-topics", type=int, default=DEFAULT_NUM_TOPICS, help="Number of LDA topics.")
    p.add_argument("--passes", type=int, default=DEFAULT_LDA_PASSES, help="How many training passes.")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_LDA_CHUNK_SIZE, help="Docs per update chunk.")
    args = p.parse_args(argv)

    if args.build_threads_csv:
        build_threads_csv()

    df, _labels = run(
        input_path=Path(args.input),
        num_topics=args.num_topics,
        lda_passes=args.passes,
        chunk_size=args.chunk_size,
    )

    print(f"Saved preprocessed data to: {THREADS_PREPROCESSED_PATH}")
    print(f"Saved topic labels to: {TOPIC_LABELS_PATH}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())