#!/usr/bin/env python3
"""
Train a BPE tokenizer with SentencePiece, check it, then tokenize the splits.

Inputs (from clean_gutenberg.py):
    clean/tokenizer_train.txt   train books, one paragraph per line
    clean/train.jsonl           one record per book: {"id", "title", "text"}
    clean/val.jsonl

Outputs:
    tokenizer/gb.model              the tokenizer (load this everywhere else)
    tokenizer/gb.vocab              human-readable list of pieces and scores
    clean/train.bin, clean/val.bin  token ids as uint16, books separated by </s>
    clean/tokenized_meta.json       vocab size, special ids, token counts

Tokenizer settings match the assignment exactly:
    model_type="bpe", vocab_size=4096, max_sentence_length=32768,
    character_coverage=1.0, normalization_rule_name="nmt_nfkc_cf"

Usage:
    python train_tokenizer.py
"""
import argparse
import json
import os

import numpy as np
import sentencepiece as spm


def train(input_txt, model_prefix, vocab_size):
    # Arguments exactly as given in the assignment
    spm.SentencePieceTrainer.train(
        input=input_txt,                   # plain text; each line is one "sentence" (here: a paragraph)
        model_type="bpe",                  # byte pair encoding (SentencePiece's default is "unigram")
        model_prefix=model_prefix,         # writes <prefix>.model and <prefix>.vocab
        vocab_size=vocab_size,             # total tokens, including <unk>, <s>, </s>
        max_sentence_length=32768,         # lines longer than this many BYTES are skipped (default ~4 KB)
        character_coverage=1.0,            # every character seen in training becomes a base symbol
        normalization_rule_name="nmt_nfkc_cf",  # NFKC normalization + lowercase; "\n" becomes a space
        # Not from the assignment, only affect speed and console output:
        num_threads=os.cpu_count() or 1,
        minloglevel=1,
    )


def check(sp, sample_text):
    """Round-trip and inspect the tokenizer on held-out text."""
    ids = sp.encode(sample_text)
    n_unk = sum(i == sp.unk_id() for i in ids)
    print(f"Vocab size:        {sp.get_piece_size()}")
    if n_unk == 0:
        # nmt_nfkc_cf means decoding gives back the *normalized* text (lowercase, newlines as spaces)
        expected = sp.normalize(sample_text).replace("▁", " ").lstrip(" ")
        assert sp.decode(ids) == expected, "decode(encode(text)) != normalized text"
        print(f"Round-trip:        exact (after normalization) on {len(sample_text):,} chars of val text")
    else:
        print("Round-trip:        skipped, <unk> tokens can't be decoded back to their characters")
    print(f"Chars per token:   {len(sample_text) / max(1, len(ids)):.2f}")
    print(f"Unknown tokens:    {n_unk} (characters in val text never seen in train)")
    snippet = sample_text[:120]
    print(f"Example pieces:    {sp.encode(snippet, out_type=str)[:25]}")


def encode_split(sp, jsonl_path, bin_path):
    """Tokenize every book in a split and write one flat uint16 array, <eos> after each book."""
    assert sp.get_piece_size() < 2**16, "vocab too large for uint16"
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            chunks.append(np.array(sp.encode(text) + [sp.eos_id()], dtype=np.uint16))
    tokens = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    tokens.tofile(bin_path)
    return len(chunks), len(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default="clean")
    ap.add_argument("--tokenizer-dir", default="tokenizer")
    ap.add_argument("--vocab-size", type=int, default=4096)   # assignment value
    args = ap.parse_args()

    os.makedirs(args.tokenizer_dir, exist_ok=True)
    model_prefix = os.path.join(args.tokenizer_dir, "gb")

    print("Training BPE tokenizer...")
    train(os.path.join(args.clean_dir, "tokenizer_train.txt"), model_prefix, args.vocab_size)
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")

    # Check on validation text: it was never seen during tokenizer training
    with open(os.path.join(args.clean_dir, "val.txt"), encoding="utf-8") as f:
        val_sample = f.read(200_000)
    print()
    check(sp, val_sample)

    print()
    meta = {"vocab_size": sp.get_piece_size(), "dtype": "uint16",
            "unk_id": sp.unk_id(), "bos_id": sp.bos_id(), "eos_id": sp.eos_id(),
            "pad_id": sp.pad_id(), "tokenizer": model_prefix + ".model"}
    for split in ("train", "val"):
        n_books, n_tokens = encode_split(
            sp, os.path.join(args.clean_dir, f"{split}.jsonl"),
            os.path.join(args.clean_dir, f"{split}.bin"))
        meta[f"{split}_books"] = n_books
        meta[f"{split}_tokens"] = n_tokens
        print(f"{split}: {n_books} books -> {n_tokens:,} tokens -> {args.clean_dir}/{split}.bin")

    with open(os.path.join(args.clean_dir, "tokenized_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Load later with np.fromfile('{args.clean_dir}/train.bin', dtype=np.uint16)")


if __name__ == "__main__":
    main()