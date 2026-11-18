"""
Split already-downloaded Gutenberg books into train/val at the book level.

- Reads every books/*.txt (nothing is moved or modified)
- Groups editions of the same title so they land on the same side of the split
- Optionally drops non-English books
- Seeded shuffle -> same split every run
- Writes splits.json: {"train": [...], "val": [...], "dropped": {...}, "books": {...}}
"""
import json
import os
import random
import re
from collections import defaultdict

BOOKS_DIR = "books"
SPLITS_FILE = "splits.json"
VAL_FRACTION = 0.10
SEED = 43
ENGLISH_ONLY = True


def read_header(path):
    """Pull Title / Author / Language from the Gutenberg header."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        head = f.read(5000)

    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", head, re.M | re.I)
        return m.group(1).strip() if m else None

    return {"title": field("Title"), "author": field("Author"), "language": field("Language")}


def title_key(title, book_id):
    """Normalize a title so different editions match: 'Frankenstein; Or, The Modern Prometheus' -> 'frankenstein'."""
    if not title:
        return f"__untitled_{book_id}"
    t = re.split(r"[;:(\[]", title)[0].lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"^(the|a|an) ", "", t.strip())
    return re.sub(r"\s+", " ", t).strip() or f"__untitled_{book_id}"


def main():
    files = sorted(f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt"))
    books, dropped = {}, {}
    groups = defaultdict(list)

    for fname in files:
        book_id = fname[:-4]
        meta = read_header(os.path.join(BOOKS_DIR, fname))
        lang = meta["language"] or ""
        if ENGLISH_ONLY and lang and "english" not in lang.lower():
            dropped[book_id] = f"language: {lang}"
            continue
        meta["group"] = title_key(meta["title"], book_id)
        books[book_id] = meta
        groups[meta["group"]].append(book_id)

    # Show grouped editions so you can sanity-check the matching
    for key, ids in sorted(groups.items()):
        if len(ids) > 1:
            titles = [books[i]["title"] for i in ids]
            print(f"Same work, kept together: {ids} -> {titles}")

    # Shuffle groups (not individual books) with a fixed seed
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)

    n_val_target = round(VAL_FRACTION * len(books))
    train, val = [], []
    for key in keys:
        members = sorted(groups[key])
        if len(val) + len(members) <= n_val_target:
            val += members
        else:
            train += members

    splits = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train": sorted(train),
        "val": sorted(val),
        "dropped": dropped,
        "books": books,
    }
    with open(SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"\n{len(files)} files -> {len(train)} train, {len(val)} val, {len(dropped)} dropped")
    for book_id, reason in dropped.items():
        print(f"  dropped {book_id}: {reason}")
    print("Val books:")
    for book_id in sorted(val):
        print(f"  {book_id}: {books[book_id]['title']}")
    print(f"Saved -> {SPLITS_FILE}")


if __name__ == "__main__":
    main()