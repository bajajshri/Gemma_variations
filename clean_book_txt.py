import json 
import os
import re

START_RE = re.compile(r'^\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERGE-?BOOK', re.IGNORECASE)
END_RE = re.compile(r'^\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG E-?BOOK', re.IGNORECASE)

ITALIC_RE = re.compile(r'_(?=\S)(.{1,200}?)(?<=\S)_')   # _word_ / _a phrase_ -> word / a phrase
WS_RE = re.compile(r'[ \t]+')                           # runs of spaces/tabs -> one space

def clean_paragraph(lines):
    """Join hard-wrapped lines into one paragraph and clean it."""
    text = " ".join(lines)          # single newlines -> spaces
    text = ITALIC_RE.sub(r'\1', text)
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


def  prepare_llm_dataset(books_folder, splits_file, output_folder, tokenizer_txt):
    


if __name__ == "__main__":
    prepare_llm_dataset(
        books_folder='books',
        splits_file='splits.json',
        output_folder='clean',
        tokenizer_txt='clean/tokenizer_train.txt',
    )