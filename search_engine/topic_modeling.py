import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
THREADS_PREPROCESSED_PATH = REPO_ROOT / "data" / "processed" / "threads_preprocessed.csv"
LABELS_PATH = REPO_ROOT / "data" / "processed" / "topic_labels.csv"


def list_topics():
    if not LABELS_PATH.exists():
        print("no trained model found.") 
        print("please run: python -m search_engine.preprocess")
        return
    labels_df = pd.read_csv(LABELS_PATH)
    for _, row in labels_df.iterrows():
        print(f"{row['topic_id']:>3}: {row['label']}")


def list_documents(topic_query):
    if not THREADS_PREPROCESSED_PATH.exists() or not LABELS_PATH.exists():
        print("no trained model found.") 
        print("please run: python -m search_engine.preprocess")
        return

    labels_df = pd.read_csv(LABELS_PATH)
    df = pd.read_csv(THREADS_PREPROCESSED_PATH)

    match = labels_df[labels_df["label"].str.lower() == topic_query.lower()]
    if match.empty:
        try:
            topic_id = int(topic_query)
            if topic_id not in labels_df["topic_id"].values:
                print(f"Topic id {topic_id} not found.")
                return
        except ValueError:
            print(f"No topic matching '{topic_query}'. Use 'topics' to see available labels.")
            return
    else:
        topic_id = int(match.iloc[0]["topic_id"])

    label = labels_df[labels_df["topic_id"] == topic_id].iloc[0]["label"]
    topic_col = "dominant_topic" if "dominant_topic" in df.columns else "topic"
    if topic_col not in df.columns:
        print(f"Missing topic id column (dominant_topic/topic) in {THREADS_PREPROCESSED_PATH}.")
        return

    conf_col = None
    if "topic_conf" in df.columns:
        conf_col = "topic_conf"
    elif "topic_confidence" in df.columns:
        conf_col = "topic_confidence"

    matching = df[df[topic_col] == topic_id]
    if conf_col:
        matching = matching.sort_values(conf_col, ascending=False)

    print(f"Topic {topic_id} ({label}) — {len(matching)} documents\n")
    for _, row in matching.iterrows():
        conf = row.get(conf_col, None) if conf_col else None
        if conf is None or (isinstance(conf, float) and pd.isna(conf)):
            print(f"  {row.get('title','')}")
        else:
            print(f"  [{float(conf):.3f}] {row.get('title','')}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python topic_modeling.py topics")
        print("  python topic_modeling.py docs <topic_label_or_id>")
        print("")
        print("Note: topic training is done during preprocessing:")
        print("  python -m search_engine.preprocess")
        return

    command = sys.argv[1]
    if command == "topics":
        list_topics()
    elif command == "docs":
        if len(sys.argv) < 3:
            print("incorrect number of arguments")
            return
        list_documents(" ".join(sys.argv[2:]))
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
