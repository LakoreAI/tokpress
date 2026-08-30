"""Whole-corpus byte-level BPE trainer: learns a valid, transitively-complete `mergeable_ranks` dict that can be loaded straight into a real `tiktoken.Encoding`.

A restricted vocabulary is only a valid tokenizer if every non-single-byte token is the concatenation of two lower-ranked tokens (the earlier mined-piece approach violated this). This trainer produces a genuine BPE chain: it starts from the 256 single bytes and repeatedly merges the most frequent adjacent pair, so every merged token is provably the concatenation of two already-defined, lower-rank tokens (verified by `validate_mergeable_ranks`).

Training must match how tiktoken encodes: `Encoding._encode_bytes` routes valid UTF-8 through `encode_ordinary`, which splits the input on the regex `pat_str` and runs BPE per regex piece (only invalid-UTF-8 tails get pure whole-input byte BPE). A vocabulary trained with naive whole-input BPE therefore fragments on piece boundaries and under-performs, so this trainer pre-tokenizes the corpus with the same `pat_str` and forbids merges across piece boundaries -- the standard GPT-style pipeline -- making the trained vocab's encoding exactly reproducible by tiktoken.

Correctness-first, deliberately not a scale-optimized trainer: the byte-level BPE loop is O(num_merges * corpus_size) in the pure-Python list rebuild, so training is meant for a sampled corpus (default cap 256KB), an offline one-time cost.
"""

import re

_STRIDE = 1 << 16  # single-int pair keys: a * STRIDE + b, valid for vocab sizes up to 2^16

# ASCII-ish pre-tokenizer pattern, deliberately simple so Python's stdlib
# `re` and tiktoken's `regex` engine split identically (no \p{...}, no
# lookarounds). Only consulted for str-level encode / training pretokenization.
DEFAULT_PAT_STR = r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"


def _get_stats(ids: list[int], boundaries: list[bool] | None = None) -> dict[int, int]:
    counts: dict[int, int] = {}
    for i in range(1, len(ids)):
        if boundaries is not None and boundaries[i - 1]:
            continue  # never count a pair that straddles a piece boundary
        key = ids[i - 1] * _STRIDE + ids[i]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _merge(
    ids: list[int], boundaries: list[bool] | None, pair_key: int, new_id: int
) -> tuple[list[int], list[bool] | None]:
    a, b = divmod(pair_key, _STRIDE)
    out: list[int] = []
    new_bound: list[bool] | None = [] if boundaries is not None else None
    i = 0
    n = len(ids)
    while i < n:
        prev_bound = boundaries[i - 1] if i > 0 and boundaries is not None else None
        if i + 1 < n and (boundaries is None or not boundaries[i]) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            if new_bound is not None and prev_bound is not None:
                new_bound.append(prev_bound)
            i += 2
        else:
            out.append(ids[i])
            if new_bound is not None and prev_bound is not None:
                new_bound.append(prev_bound)
            i += 1
    return out, new_bound


def _pretokenize(text: str, pat_str: str = DEFAULT_PAT_STR) -> list[str]:
    return re.findall(pat_str, text)


def _build_ids_and_boundaries(pieces: list[bytes]) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    boundaries: list[bool] = []  # boundaries[i] == True -> no merge between ids[i] and ids[i+1]
    for piece in pieces:
        if not piece:
            continue
        if ids:
            boundaries.append(True)
        for _ in range(len(piece) - 1):
            boundaries.append(False)
        ids.extend(piece)
    return ids, boundaries


def _pretokenize_bytes(corpus: bytes) -> list[bytes]:
    try:
        text = corpus.decode("utf-8")
    except UnicodeDecodeError:
        return [corpus]
    return [p.encode("utf-8") for p in _pretokenize(text)]


def _train_pieces(
    pieces: list[bytes], vocab_size: int, progress=None
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    ids, boundaries = _build_ids_and_boundaries(pieces)
    if not ids:
        raise ValueError("empty corpus: nothing to train a vocabulary on")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merge_sequence: list[tuple[bytes, bytes]] = []
    stats = _get_stats(ids, boundaries)

    num_merges = vocab_size - 256
    for i in range(num_merges):
        if not stats:
            break
        # highest count, then lowest key, for reproducibility
        pair_key, _count = max(stats.items(), key=lambda kv: (kv[1], -kv[0]))
        new_id = 256 + i
        ids, boundaries = _merge(ids, boundaries, pair_key, new_id)
        a, b = divmod(pair_key, _STRIDE)
        merge_sequence.append((vocab[a], vocab[b]))
        vocab[new_id] = vocab[a] + vocab[b]
        stats = _get_stats(ids, boundaries)
        if progress is not None and (i % 250 == 0 or i == num_merges - 1):
            progress(i + 1, num_merges)

    mergeable_ranks: dict[bytes, int] = {vocab[i]: i for i in range(256 + len(merge_sequence))}
    return mergeable_ranks, merge_sequence


def train_mergeable_ranks(corpus: bytes, vocab_size: int, progress=None) -> dict[bytes, int]:
    """Learn a byte-level BPE vocabulary from `corpus` (raw bytes, already sampled/capped by the caller if desired). If the corpus is valid UTF-8, it is pre-tokenized with DEFAULT_PAT_STR and merges never cross piece boundaries (matching how tiktoken's `_encode_bytes` actually encodes); otherwise whole-input byte BPE is used. Returns mergeable_ranks: bytes -> rank, rank 0..255 = the single bytes, then one rank per merge.

    Deterministic for a given corpus: the most frequent pair is picked with a fixed tie-break (highest count, then lowest pair key).
    """
    ranks, _seq = _train(corpus, vocab_size, progress)
    return ranks


def train_with_merge_sequence(
    corpus: bytes, vocab_size: int, progress=None
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """Like train_mergeable_ranks, but also returns the exact (left, right) byte-piece merge sequence, for verifying tiktoken's encoder reproduces the trained chain exactly."""
    return _train(corpus, vocab_size, progress)


def _train(corpus: bytes, vocab_size: int, progress=None) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    if not 256 < vocab_size <= _STRIDE:
        raise ValueError(f"vocab_size must be in (256, {_STRIDE}]")
    return _train_pieces(_pretokenize_bytes(corpus), vocab_size, progress)


def encode_with_merge_sequence(corpus: bytes, merge_sequence: list[tuple[bytes, bytes]]) -> list[int]:
    """Re-apply a trained merge sequence to a corpus (with the same pre-tokenization/boundaries the trainer used), producing the token-id list the trained chain implies. Used to verify tiktoken's encoder agrees with the trained merge order exactly."""
    ids, boundaries = _build_ids_and_boundaries(_pretokenize_bytes(corpus))
    ids_to_bytes = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for left, right in merge_sequence:
        a = next(k for k, v in ids_to_bytes.items() if v == left)
        b = next(k for k, v in ids_to_bytes.items() if v == right)
        ids, boundaries = _merge(ids, boundaries, a * _STRIDE + b, next_id)
        ids_to_bytes[next_id] = left + right
        next_id += 1
    return ids


def validate_mergeable_ranks(mergeable_ranks: dict[bytes, int]) -> bool:
    """Verify the merge chain is valid: every single byte 0..255 present, and every multi-byte token is the concatenation of two tokens with strictly lower ranks. A real tiktoken.Encoding relies on this invariant."""
    for b in range(256):
        if mergeable_ranks.get(bytes([b])) != b:
            return False
    by_bytes = {tok: rank for tok, rank in mergeable_ranks.items()}
    for tok, rank in mergeable_ranks.items():
        if len(tok) == 1:
            continue
        # find the split: since every token came from a merge of two pieces,
        # try all splits and require one where both pieces have lower ranks.
        found = False
        for cut in range(1, len(tok)):
            left, right = tok[:cut], tok[cut:]
            if left in by_bytes and right in by_bytes and by_bytes[left] < rank and by_bytes[right] < rank:
                found = True
                break
        if not found:
            return False
    return True


def sample_corpus(blob: bytes, max_bytes: int) -> bytes:
    """Cap a training corpus: head-sampling is fine for vocab training (the distribution of a schema's byte n-grams is stable across the stream) and keeps the O(merges * n) trainer tractable. Deterministic."""
    if len(blob) <= max_bytes:
        return blob
    return blob[:max_bytes]


def dump_rank_file(mergeable_ranks: dict[bytes, int], path: str) -> None:
    import base64

    with open(path, "w") as fh:
        for tok, rank in sorted(mergeable_ranks.items(), key=lambda kv: kv[1]):
            fh.write(f"{base64.b64encode(tok).decode('ascii')} {rank}\n")


def load_rank_file(path: str) -> dict[bytes, int]:
    import base64

    mergeable_ranks = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            b64, _, rank_s = line.partition(" ")
            mergeable_ranks[base64.b64decode(b64)] = int(rank_s)
    return mergeable_ranks


def build_tiktoken_encoding(mergeable_ranks: dict[bytes, int], name: str = "tokpress-custom") -> object:
    """Construct a real tiktoken.Encoding from trained ranks. Byte-exact tokenization (`_encode_bytes`/`decode_bytes`) works for arbitrary binary, exactly like o200k_base's adapter path. The pat_str is the same one used for training pre-tokenization, so tiktoken's str-level `encode` and the trained chain agree."""
    import tiktoken

    return tiktoken.Encoding(
        name=name,
        pat_str=DEFAULT_PAT_STR,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
