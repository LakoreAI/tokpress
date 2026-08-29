"""ByteTokenizer: a byte-trie tokenizer with greedy longest-prefix match and raw-byte fallback (token ids 0-255 alias raw byte values; learned pieces start at id 256)."""


class ByteTokenizer:
    def __init__(self) -> None:
        self._trie: dict = {}
        self.token_storage: list[bytes | None] = [bytes([i]) for i in range(256)]
        self.vocab_size = 256

    def add_token(self, piece: bytes, token_id: int) -> None:
        node = self._trie
        for b in piece:
            node = node.setdefault(b, {})
        node["$"] = token_id

        if token_id + 1 > len(self.token_storage):
            self.token_storage.extend([None] * (token_id + 1 - len(self.token_storage)))
        self.token_storage[token_id] = piece
        if token_id + 1 > self.vocab_size:
            self.vocab_size = token_id + 1

    def load_vocab(self, vocab) -> None:
        for piece in vocab.pieces:
            self.add_token(piece.bytes, piece.id)

    def encode(self, data: bytes) -> list[int]:
        tokens = []
        pos = 0
        n = len(data)
        while pos < n:
            node = self._trie
            longest_match_len = 0
            matched_id = -1
            lookup_pos = pos
            while lookup_pos < n:
                b = data[lookup_pos]
                if b not in node:
                    break
                node = node[b]
                if "$" in node:
                    longest_match_len = (lookup_pos - pos) + 1
                    matched_id = node["$"]
                lookup_pos += 1
            if longest_match_len > 1 and matched_id != -1:
                tokens.append(matched_id)
                pos += longest_match_len
            else:
                tokens.append(data[pos])
                pos += 1
        return tokens

    def decode(self, tokens: list[int]) -> bytes:
        out = bytearray()
        storage_len = len(self.token_storage)
        for tid in tokens:
            if tid < 256:
                out.append(tid)
            elif tid < storage_len and self.token_storage[tid] is not None:
                out.extend(self.token_storage[tid])
            # else: out-of-range id, silently dropped
        return bytes(out)
