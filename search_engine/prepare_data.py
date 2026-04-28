import json
from pathlib import Path

input_path = Path("..") / "data" / "raw" / "subreddit-AskHistorians" / "utterances.jsonl"
conversation_path = Path("..") / "data" / "raw" / "subreddit-AskHistorians" / "conversations.json"
output_path = Path("..") / "data" / "processed" / "threads.jsonl"

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
                "initial_post": "",
                "comments": [],
                "url": "https://reddit.com" + permalink if permalink else "",
                #"subreddit": meta.get("subreddit", "AskHistorians")
            }

        # Original post
        if reply_to is None:
            threads[thread_id]["initial_post"] = text

            if title:
                threads[thread_id]["title"] = title

            if permalink:
                threads[thread_id]["url"] = "https://reddit.com" + permalink

        # Comments and replies
        else:
            threads[thread_id]["comments"].append(text)

with output_path.open("w", encoding="utf-8") as f:
    saved_count = 0

    for thread in threads.values():
        if (
            thread["title"] == ""
            and thread["initial_post"] == ""
            and len(thread["comments"]) == 0
        ):
            continue

        f.write(json.dumps(thread, ensure_ascii=False) + "\n")
        saved_count += 1

print("Saved", saved_count, "threads to", output_path)