import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecRetrieval:

    def __init__(self):
        repo_root = Path(__file__).resolve().parent.parent
        self.preprocessed_path = repo_root / "data" / "processed" / "preprocessed_utterances.csv"
        self.model_path = repo_root / "data" / "processed" / "word2vec.model"
        self.dataset = None
        self.model = None
        self.doc_vectors = None

    def load_preprocessed_data(self):
        if not self.preprocessed_path.exists():
            print("Preprocessed data not found. Run preprocess.py first.")
            return False
        self.dataset = pd.read_csv(self.preprocessed_path)
        print(f"Loaded {len(self.dataset)} documents.")
        return True

    def train(self, vector_size=100, window=5, min_count=2, epochs=10):
        if self.dataset is None:
            print("No data loaded.")
            return

        sentences = [
            str(row.iloc[2]).split()
            for _, row in self.dataset.iterrows()
        ]

        print("Training Word2Vec...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=4
        )
        self.model.save(str(self.model_path))
        print(f"Model saved to: {self.model_path}")
        self._build_doc_vectors()

    def load_model(self):
        if not self.model_path.exists():
            print("No saved model found. Run train() first.")
            return False
        self.model = Word2Vec.load(str(self.model_path))
        self._build_doc_vectors()
        print("Model loaded.")
        return True

    def _text_to_vector(self, text):
        words = str(text).split()
        vectors = [
            self.model.wv[w]
            for w in words
            if w in self.model.wv
        ]
        if not vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def _build_doc_vectors(self):
        print("Building document vectors...")
        self.doc_vectors = np.array([
            self._text_to_vector(str(row.iloc[2]))
            for _, row in self.dataset.iterrows()
        ])
        print("Document vectors ready.")

    def execute_search_word2vec(self, query):
        if self.model is None or self.doc_vectors is None:
            print("Model not ready. Call train() or load_model() first.")
            return np.array([])

        query_vector = self._text_to_vector(query).reshape(1, -1)
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        return scores


if __name__ == '__main__':
    w2v = Word2VecRetrieval()

    if not w2v.load_preprocessed_data():
        exit()

    if not w2v.load_model():
        w2v.train()

    queries = ["roman empire collapse", "medieval trade routes", "american civil war causes"]
    print("#########\n")
    print("Results for Word2Vec")
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