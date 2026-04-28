from text_retrieval import TextRetrieval


if __name__ == '__main__':
  tr = TextRetrieval()
  tr.read_and_preprocess_Data_File()
  tr.save_preprocessed_data()
  print(f"Saved preprocessed data to: {tr.preprocessed_path}")
