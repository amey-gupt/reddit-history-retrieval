import numpy as np
import pandas as pd
from pathlib import Path
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys

class Word2VecRetrieval:

    def __init__(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.preprocessed_path = repo_root / "data" / "processed" / "threads_preprocessed.csv"
        self.dataset = None
        self.w2v_model = None
        self.doc_vectors = None

    def load_preprocessed_data(self):
        if not self.preprocessed_path.exists():
            print("Preprocessed data not found. Run preprocess.py first, or download the data from the link provided.")
            return False
        self.dataset = pd.read_csv(self.preprocessed_path)
        print(f"Loaded {len(self.dataset)} documents.")
        return True

    def load_model(self):
        print("Loading pretrained GloVe model...")
        self.w2v_model = api.load("glove-wiki-gigaword-50")
        print("Model loaded.")
        self._build_doc_vectors()

    def _text_to_vector(self, text):
        words = str(text).split()
        vectors = []
        for word in words:
            try:
                vectors.append(self.w2v_model[word])
            except KeyError:
                continue
        if not vectors:
            return np.zeros(self.w2v_model.vector_size)
        return np.mean(vectors, axis=0)

    def _build_doc_vectors(self):
        print("Building document vectors...")
        col = "full_text" if "full_text" in self.dataset.columns else "content"
        self.doc_vectors = np.array([
            self._text_to_vector(str(row[col]))
            for _, row in self.dataset.iterrows()
        ])
        print("Document vectors ready.")

    def execute_search_word2vec(self, query):
        if self.w2v_model is None or self.doc_vectors is None:
            print("Model not ready. Call load_model() first.")
            return np.array([])

        query_vector = self._text_to_vector(query).reshape(1, -1)
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        return scores


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="word2vec_retrieval.py",
        description="Semantic retrieval using averaged pretrained GloVe vectors.",
    )
    p.add_argument("--input", default=None, help="Path to preprocessed CSV (defaults to data/processed/threads_preprocessed.csv).")
    p.add_argument("--query", default=None, help="Search query text. If omitted, runs a 5-query demo.")
    p.add_argument("--top-k", type=int, default=10, help="Number of results to print.")
    p.add_argument(
        "--model",
        default="glove-wiki-gigaword-50",
        help="Gensim downloader key (default: glove-wiki-gigaword-50).",
    )
    args = p.parse_args(argv)

    w2v = Word2VecRetrieval()
    if args.input:
        w2v.preprocessed_path = Path(args.input)

    if not w2v.load_preprocessed_data():
        return 2

    print(f"Loading pretrained vectors: {args.model}")
    w2v.w2v_model = api.load(args.model)
    w2v._build_doc_vectors()

    demo_queries = [
        "roman empire collapse",
        "medieval trade routes",
        "american civil war causes",
        "french revolution",
        "fall of berlin wall",
    ]
    queries = [args.query] if args.query else demo_queries

    k = max(1, int(args.top_k))
    print("Results for Word2Vec (GloVe pretrained)")

    for query in queries:
        print("\nQUERY:", query)
        scores = w2v.execute_search_word2vec(query)
        if scores.size == 0:
            print("No scores computed.")
            return 3

        idxs = np.argsort(scores)

        print(f"\ntop {k} most relevant:")
        for i in reversed(idxs[-k:]):
            print(f"document: {w2v.dataset.iloc[i, 1]}, score: {scores[i]:.4f}")

        print(f"\nbottom {k} least relevant:")
        for i in idxs[:k]:
            print(f"document: {w2v.dataset.iloc[i, 1]}, score: {scores[i]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))