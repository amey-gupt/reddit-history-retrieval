import json
from pathlib import Path
import csv

input_path = Path("..") / "data" / "raw" / "subreddit-AskHistorians" / "utterances.jsonl"
conversation_path = Path("..") / "data" / "raw" / "subreddit-AskHistorians" / "conversations.json"
output_path = Path("..") / "data" / "processed" / "threads.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

# Load real thread titles from conversations.json
with conversation_path.open("r", encoding="utf-8") as f:
    conversations = json.load(f)

threads = {}

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        text = row.get("text", "").replace("\u2028", " ").replace("\u2029", " ").strip()
        if text == "":
            continue

        thread_id = row.get("root")
        reply_to = row.get("reply_to")
        meta = row.get("meta", {})
        permalink = meta.get("permalink", "")

        # Real title comes from conversations.json
        title = conversations.get(thread_id, {}).get("title", "")

        if thread_id not in threads:
            threads[thread_id] = {
                "thread_id": thread_id,
                "title": title,
                "content_parts": [],
                "url": "",
                #"subreddit": meta.get("subreddit", "AskHistorians")
            }

        if reply_to is None and permalink:
            threads[thread_id]["url"] = "https://reddit.com" + permalink

        if text:
            threads[thread_id]["content_parts"].append(text)

rows = []
for thread in threads.values():
    content = " ".join(thread["content_parts"]).strip()
    if thread["title"] == "" and content == "":
        continue

    rows.append({
        "thread_id": thread["thread_id"],
        "title": thread["title"],
        "content": content,
        "url": thread["url"],
    })

with output_path.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["thread_id", "title", "content", "url"])
    w.writeheader()
    w.writerows(rows)

print("Saved", len(rows), "threads to", output_path)