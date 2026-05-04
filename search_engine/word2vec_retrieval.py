import numpy as np
import pandas as pd
from pathlib import Path
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecRetrieval:

    def __init__(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.preprocessed_path = repo_root / "data" / "processed" / "threads_preprocessed.csv"
        self.dataset = None
        self.w2v_model = None
        self.doc_vectors = None

    def load_preprocessed_data(self):
        if not self.preprocessed_path.exists():
            print("Preprocessed data not found. Run preprocess.py first.")
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
        self.doc_vectors = np.array([
            self._text_to_vector(str(row.iloc[2]))
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


if __name__ == '__main__':
    w2v = Word2VecRetrieval()

    if not w2v.load_preprocessed_data():
        exit()

    w2v.load_model()

    queries = ["roman empire collapse", "medieval trade routes", "american civil war causes"]
    print("#########\n")
    print("Results for Word2Vec (GloVe pretrained)")
    for query in queries:
        print("\nQUERY:", query)
        scores = w2v.execute_search_word2vec(query)
        idxs = np.argsort(scores)

        print("\ntop 5 most relevant:")
        for i in reversed(idxs[-5:]):
            print(f"document: {w2v.dataset.iloc[i, 1]}, score: {scores[i]:.4f}")

        print("\nbottom 5 least relevant:")
        for i in idxs[:5]:
            print(f"document: {w2v.dataset.iloc[i, 1]}, score: {scores[i]:.4f}")