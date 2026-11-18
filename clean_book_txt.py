import json 
import os
import re

START_RE = re.compile(r'^\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG E-?BOOK', re.IGNORECASE)
END_RE = re.compile(r'^\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK', re.IGNORECASE)

ITALIC_RE = re.compile(r'_(?=\S)(.{1,200}?)(?<=\S)_')   # _word_ / _a phrase_ -> word / a phrase
WS_RE = re.compile(r'[ \t]+')                           # runs of spaces/tabs -> one space
ILLUSTRATION_RE= re.compile(r'\[(?=\S)(.{1,200}?)(?<=\S)\]') # remove illustrations

def clean_paragraph(lines):
    """Join hard-wrapped lines into one paragraph and clean it."""
    text = " ".join(lines)          # single newlines -> spaces
    text = ITALIC_RE.sub(r'\1', text)
    text = ILLUSTRATION_RE.sub('', text)
    text = WS_RE.sub(' ', text)
    return text.strip()

def read_lines(path):
    """Read a file as text; utf-8-sig drops a BOM. Fall back to latin-1 for old files."""
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return f.readlines()      # text mode turns CRLF into \n
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1') as f:
            return f.readlines()

        

def extract_paragraphs(path):
    """Return (paragraphs, found_start, found_end) for one Gutenberg book."""
    paragraphs = []
    current = []
    found_start = False
    found_end = False

    for line in read_lines(path):
        stripped = line.strip()

        # Skip everything up to and including the START marker
        if not found_start:
            if START_RE.match(stripped):
                found_start = True
            continue

        # Stop at the END marker (license block is after it)
        if END_RE.match(stripped):
            found_end = True
            break

        if stripped == "":
            # Blank line = paragraph break. Flush only if we have something.
            if current:
                cleaned = clean_paragraph(current)
                if cleaned:
                    paragraphs.append(cleaned)
                current = []
                
        elif stripped.isdigit():
            continue  # Skip standalone page numbers
        else:
            current.append(stripped)

    if current:
        paragraphs.append(clean_paragraph(current))

    return paragraphs, found_start, found_end


def  prepare_llm_dataset(books_folder, splits_file, output_folder, tokenizer_txt):
    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    os.makedirs(output_folder, exist_ok=True)
    tok_dir = os.path.dirname(tokenizer_txt)
    if tok_dir:
        os.makedirs(tok_dir, exist_ok=True)

    with open(tokenizer_txt, 'w', encoding='utf-8') as txt_file:
        # NEW: loop over the split lists instead of os.listdir(...)
        for split in ("train", "val"):
            output_jsonl = os.path.join(output_folder, f"{split}.jsonl")
            output_plain = os.path.join(output_folder, f"{split}.txt")   # NEW: plain-text copy
            book_count = 0
            para_count = 0

            with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file, \
                    open(output_plain, 'w', encoding='utf-8') as plain_file:
                for book_id in splits[split]:
                    path = os.path.join(books_folder, f"{book_id}.txt")
                    if not os.path.exists(path):
                        print(f"WARNING: {path} listed in {splits_file} but not found; skipped")
                        continue

                    paragraphs, found_start, found_end = extract_paragraphs(path)

                    if not found_start:
                        print(f"WARNING: no START marker in {book_id}.txt; skipped")
                        continue
                    if not found_end:
                        print(f"WARNING: no END marker in {book_id}.txt; license text may be included")

                    # NEW: one record per book, paragraphs separated by a blank line
                    title = splits.get("books", {}).get(book_id, {}).get("title")
                    record = {"id": book_id, "title": title, "text": "\n\n".join(paragraphs)}
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # NEW: same text as plain text; paragraphs and books separated by a blank line
                    if book_count > 0:
                        plain_file.write("\n\n")
                    plain_file.write(record["text"])

                    # NEW: tokenizer text comes from train books only
                    if split == "train":
                        for p in paragraphs:
                            txt_file.write(p + "\n")   # one paragraph per line, for sentencepiece

                    book_count += 1
                    para_count += len(paragraphs)
                    print(f"[{split}] {book_id}.txt: {len(paragraphs)} paragraphs")

                if book_count > 0:
                    plain_file.write("\n")   # end the file with a newline

            print(f"{split}: {book_count} books, {para_count} paragraphs -> '{output_jsonl}' and '{output_plain}'\n")

    print(f"Done! Tokenizer training text (train only) -> '{tokenizer_txt}'")



if __name__ == "__main__":
    prepare_llm_dataset(
        books_folder='books',
        splits_file='splits.json',
        output_folder='clean',
        tokenizer_txt='clean/tokenizer_train.txt',
    )