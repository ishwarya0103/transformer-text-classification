"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

try:
    from .base import Tokenizer, get_stats, merge
except ImportError:
    from base import Tokenizer, get_stats, merge

import re
import pandas as pd
import numpy as np

PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"

_whitespace_re = re.compile(r"[ \t]+")
_newlines_re = re.compile(r"\n{2,}")

def clean_text_for_tokenizer(s: str) -> str:
    """Normalize text before tokenization: lowercase, clean whitespace."""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _whitespace_re.sub(" ", s)
    s = _newlines_re.sub("\n", s)
    return s


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.pad_id = None
        self.mask_id = None

    def load(self, model_file):
        super().load(model_file)
        self.pad_id = self.special_tokens.get(PAD_TOKEN)
        self.mask_id = self.special_tokens.get(MASK_TOKEN)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

        # Register special tokens
        next_id = len(self.vocab)
        self.pad_id = next_id
        self.mask_id = next_id + 1
        self.special_tokens = {PAD_TOKEN: self.pad_id, MASK_TOKEN: self.mask_id}
        self.vocab = self._build_vocab()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_with_padding(self, text, max_len):
        """Encode text and pad/truncate to max_len. Returns (ids, attention_mask)."""
        ids = self.encode(text)
        if len(ids) > max_len:
            ids = ids[:max_len]
        attn_mask = [1] * len(ids)
        pad_len = max_len - len(ids)
        ids = ids + [self.pad_id] * pad_len
        attn_mask = attn_mask + [0] * pad_len
        return ids, attn_mask


# ── Main: train tokenizer on mammography reports ─────────────────────────────

if __name__ == "__main__":
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    VOCAB_SIZE = 4096
    SAVE_PREFIX = f"mammo_tokenizer_{VOCAB_SIZE}"

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    all_reports = pd.concat([train_df["report"], test_df["report"]], ignore_index=True)
    all_reports = all_reports.fillna("")

    # Clean text before tokenizer training
    all_reports_clean = all_reports.apply(clean_text_for_tokenizer)
    corpus = "\n".join(all_reports_clean.tolist())

    print(f"Corpus: {len(all_reports)} reports, {len(corpus)} characters")
    print(f"Training BPE tokenizer with vocab_size={VOCAB_SIZE}...")

    tok = BasicTokenizer()
    tok.train(corpus, vocab_size=VOCAB_SIZE, verbose=False)

    print(f"Done. Vocab size: {len(tok.vocab)}")
    print(f"  [PAD] id={tok.pad_id}, [MASK] id={tok.mask_id}")
    print(f"  Merges learned: {len(tok.merges)}")

    tok.save(SAVE_PREFIX)
    print(f"Saved to {SAVE_PREFIX}.model and {SAVE_PREFIX}.vocab")

    # Sanity check on cleaned text
    sample = clean_text_for_tokenizer(train_df["report"].iloc[0])
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"\n--- Sanity check ---")
    print(f"Original length: {len(sample)} chars")
    print(f"Encoded length:  {len(encoded)} tokens")
    print(f"Compression:     {len(sample.encode('utf-8')) / len(encoded):.1f}x")
    print(f"Round-trip OK:   {decoded == sample}")

    # Token length distribution on cleaned text
    lengths = [len(tok.encode(r)) for r in all_reports_clean]
    lengths = np.array(lengths)
    print(f"\nToken lengths: min={lengths.min()}, median={int(np.median(lengths))}, "
          f"mean={lengths.mean():.1f}, p95={int(np.percentile(lengths, 95))}, max={lengths.max()}")
    print(f"Suggested MAX_LEN: {int(np.percentile(lengths, 99))}")