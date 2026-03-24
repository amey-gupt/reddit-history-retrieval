# reddit-history-retrieval
Reddit thread recommendation system for historical queries. Given a natural language question, it retrieves and ranks relevant discussions (focused on r/AskHistorians) using lexical (TF-IDF/BM25) and semantic (embeddings) methods, evaluated with IR metrics to compare performance.

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