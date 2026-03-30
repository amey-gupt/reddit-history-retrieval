#used assignment 1 code ec as reference
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
import math
import json
from pathlib import Path

class TextRetrieval():

  #For preprocessing
  punctuations = ""
  stop_words=set()

  #For VSM definition
  vocab = np.zeros(200)
  dataset = None
  K = 3 #
  B = 0.5 #[0,1]

  def __init__(self):
    #grab data
    repo_root = Path(__file__).resolve().parent.parent
    self.input_path = repo_root / "data" / "raw" / "subreddit-AskHistorians" / "utterances.jsonl"
    self.max_docs = 50000
    
    #use preprocessing described in assignment 1
    self.punctuations = "\'\"\\,<>./?@#$%^&*_~/!()-[]{};:"

    nltk.download('stopwords')
    self.stop_words = set(stopwords.words('english'))


  def read_and_preprocess_Data_File(self):
    ### Reads AskHistorians utterances and iterates over every document content (entry in column 2)
    ### preprocesses
    ### Stores the formated information in the same "dataset" object

    records = []
    with self.input_path.open("r", encoding="utf-8") as handle:
      for line in handle:
        obj = json.loads(line)
        text = obj.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
          continue
        # Keep [1] as display field and [2] as retrieval text to preserve your code flow.
        records.append([obj.get("subreddit", "AskHistorians"), obj.get("id", ""), text])
        if len(records) >= self.max_docs:
          break

    dataset = pd.DataFrame(records)
    punctuations = self.punctuations
    stop_words = self.stop_words
    digits = "0123456789"

    word_sum = 0

    dataset.head()
    for index, row in dataset.iterrows():
      line = row[2]
      new_line = ""
      i = 0
      while (i < len(line)):
        if line[i] == '<':
          next_close_idx = line.find('>', i)
          if next_close_idx != -1:
            next_open_idx = line.find('<', i+1, next_close_idx)
            if next_open_idx == -1:
              i = next_close_idx+1
              continue
        new_line += line[i]
        i += 1
      line = new_line
      #TODO: Implement removing stopwords and punctuation
      words = line.split()
      updated_words = []
      for i, w in enumerate(words):
          if w == "":
              continue
          w = w.lower()

          for p in punctuations:
              w = w.replace(p, "")
          for d in digits:
              w = w.replace(d, "")
          if w not in stop_words:
              updated_words.append(w)

      word_sum += len(updated_words)
      dataset.loc[index, 2] = ' '.join(updated_words)
    
    self.avdl = word_sum/dataset.shape[0]
    self.dataset = dataset #Set dataset as object attribute

  def build_vocabulary(self): #,collection):
    ### Return an array of 200 most frequent works in the collection
    ### dataset has to be read before calling the vocabulary construction

    frq = {}
    for index, row in self.dataset.iterrows():
      line = row[2]
      words = line.split()
      for w in words:
        frq[w] = frq.get(w, 0) + 1

    vocab_w_freq = sorted(frq.items(), key=lambda x: x[1], reverse=True)[:200]
    vocab = ["" for i in range(200)]
    for i, v in enumerate(vocab_w_freq):
      vocab[i] = v[0]

    
    self.vocab = np.array(vocab)

  def adapt_vocab_query(self,query):
    ### Updates the vocabulary to add the words in the query

    words = query.split()
    vocab_set = set(self.vocab)
    for word in words:
      if word not in vocab_set:
        self.vocab = np.append(self.vocab, word)
        vocab_set.add(word)


  
  def compute_IDF(self,M,collection):
    ### M number of documents in the collection; collection: documents (i.e., column 3 (index 2) in the dataset)

    self.IDF  = np.zeros(self.vocab.size) #Initialize the IDFs to zero
    for doc in collection:
      words = set(doc.split())
      for i, v in enumerate(self.vocab):
          if v in words:
            self.IDF[i] += 1
      
    for i, v in enumerate(self.IDF):
      self.IDF[i] = math.log((M+1)/(self.IDF[i] + 1))




  #### Okapi-BM25 with pivoted length normalization

  def text2BM25PLN(self,text, applyBM25_and_IDF=True):
    ### returns the bit vector representation of the text

    BM25PLNVector = np.zeros(self.vocab.size)

    words = text.split()
    freq = {}
    for w in words:
      freq[w] = freq.get(w, 0) + 1
    for i, v in enumerate(self.vocab):
      BM25PLNVector[i] = freq.get(v, 0)
      if applyBM25_and_IDF:
        BM25PLNVector[i] = (((self.K+1)*BM25PLNVector[i]) / (self.K*(1-self.B+(self.B*len(words)/self.avdl))+BM25PLNVector[i]))*self.IDF[i]

    return BM25PLNVector

  def BM25PLN_score(self,query,doc, applyBM25_and_IDF=False):
    q = self.text2BM25PLN(query)
    d = self.text2BM25PLN(doc, applyBM25_and_IDF)

    relevance = np.dot(q, d)
    return relevance

  def execute_search_BM25PLN(self,query):
    self.adapt_vocab_query(query) #Ensure query is part of the "common language" of documents and query 

    # global IDF
    self.compute_IDF(self.dataset.shape[0],self.dataset[2]) 

    relevances = np.zeros(self.dataset.shape[0]) #Initialize relevances of all documents to 0

    for i in range(relevances.size):
      relevances[i] = self.BM25PLN_score(query, self.dataset.iloc[i, 2], True)

    return relevances #in the same order of the documents in the dataset


if __name__ == '__main__':
    tr = TextRetrieval()
    tr.read_and_preprocess_Data_File() #builds the collection
    tr.build_vocabulary()#builds an initial vocabulary based on common words
    queries = ["roman empire collapse", "medieval trade routes", "american civil war causes"]
    print("#########\n")
    print("Results for BM25PLN")
    for query in queries:
      print("\nQUERY:",query)
      relevance_docs = tr.execute_search_BM25PLN(query)
      idxs = np.argsort(relevance_docs)
      print("\ntop 5 most relevant:")
      for i in reversed(idxs[-5:]):
        print(f"document: {tr.dataset.loc[i, 1]}, score: {relevance_docs[i]}")

      print("\nbottom 5 least relevant:")
      for i in idxs[:5]:
        print(f"document: {tr.dataset.loc[i, 1]}, score: {relevance_docs[i]}")
