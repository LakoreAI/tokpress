# TokPress (`tokpress`)

### A tiktoken-tokenizer-driven lossless compressor

TokPress is a pure-Python, stdlib-first (aside from `tiktoken`) lossless
compressor: it tokenizes input with tiktoken's real `o200k_base` encoding
(the same tokenizer OpenAI's models use), applies token-level LZ77, then
entropy-codes the result with rANS. The core idea is that an LLM
tokenizer's job — chunking input into frequent, semantically coherent
pieces — is close to what a compressor's dictionary wants, so tokenizing
before compressing gives the LZ and entropy stages a better-shaped alphabet
than raw bytes.

`pip install`, import, done.

---

## Architecture

```
[ Input bytes ]
       │
       ▼
┌──────────────────────────────────────────────┐
│ 1. Tokenizer                                  │
│    - tiktoken's o200k_base encoding           │
│    - byte-exact _encode_bytes/decode_bytes,   │
│      round-trips arbitrary (non-UTF-8) bytes  │
└───────────────────────┬────────────────────────┘
                        │ token ids
                        ▼
┌──────────────────────────────────────────────┐
│ 2. Token-level LZ77                           │
└───────────────────────┬────────────────────────┘
                        │ literal + match-tuple tokens
                        ▼
┌──────────────────────────────────────────────┐
│ 3. Entropy + bitstream encoder                │
│    - tries fixed-width bit-packed / per-record│
│      sparse rANS, keeps the smaller            │
└───────────────────────┬────────────────────────┘
                        │
                        ▼
              [ .tokz binary stream ]
```

---

## Install

```bash
cd tokpress
pip install -e .
```

The only third-party dependency is `tiktoken`.

---

## Usage

### CLI

```bash
tokpress compress path/to/record.json -o record.tokz
tokpress decompress record.tokz -o restored.json
tokpress bench path/to/dataset.txt
```

### Python API

```python
import tokpress

compressed = tokpress.compress(payload)   # payload: bytes or str
original = tokpress.decompress(compressed) # -> bytes, byte-exact

tokpress.compress_file("record.json", "record.tokz")
tokpress.decompress_file("record.tokz", "restored.json")
```

### Trained dictionaries (many small, schema-similar records)

If you have many small records that share structure -- JSON log lines, events, schema-homogeneous documents -- train a `TokDict` once on a sample and reuse it across every future record of that shape. This is the regime TokPress is actually built for (see `docs/VISION.md`): a lone small record compressed on its own can inflate, but a trained dictionary gives it shared cross-record LZ history and a pre-baked entropy table instead of paying a per-record table from scratch.

```bash
tokpress train-dict mydict.tokdict sample1.json sample2.json ...
tokpress compress new_record.json --dict mydict.tokdict -o new_record.tokz
tokpress decompress new_record.tokz --dict mydict.tokdict -o restored.json
```

```python
import tokpress

dictionary = tokpress.TokDict.train(sample_records)  # list[bytes]
dictionary.save("mydict.tokdict")

dictionary = tokpress.TokDict.load("mydict.tokdict")
compressed = tokpress.compress(record, dictionary=dictionary)
restored = tokpress.decompress(compressed, dictionary=dictionary)
```

The same dictionary must be available at decompress time; a stream compressed with a dictionary carries only an 8-byte fingerprint of it, not the dictionary itself, and refuses to decompress against the wrong one.

Measured on real, held-out JSON log records (`scripts/bench.py`'s trained-dictionary regime): `tokpress+dict` beat every dictionary-less baseline (gzip/bz2/lzma/zstd/parmar-style) on records the dictionary had never seen, but lost to `zstd` given that same training data as its own dictionary -- `zstd`'s dictionary training and entropy coding are simply more mature. See `docs/VISION.md` §7 for the numbers and the honest takeaway.

---

## Testing

```bash
pip install -e .
pytest tests/
```

The suite covers bitstream roundtrip, rANS roundtrip (incl. single-symbol-alphabet edge case), token-level LZ77 roundtrip, the tiktoken adapter's byte-exact roundtrip on arbitrary binary input (including invalid UTF-8), full codec roundtrips across a range of payload shapes, and black-box package/CLI tests.

---

## Performance and compression ratio

TokPress is pure Python; expect noticeably lower throughput than a compiled/native implementation, dominated by Python-level per-symbol loop overhead in the rANS coder. It has not been optimized for speed.

Without a trained dictionary, every record is compressed independently, so small or non-repetitive records can come out **larger** than the input (a 70-byte JSON sample measured at 81 bytes compressed). A trained `TokDict` (see above) fixes this for many-small-homogeneous-records workloads, but only when you have representative training data to build one from. See `docs/STATUS.md` for the honest tradeoffs.

---

## Project layout

```
tokpress/
├── src/tokpress/
│   ├── bitstream/       # LSB-first bit reader/writer
│   ├── entropy/          # SymbolStats, ContextTableSet, rANS
│   ├── tokenizer/          # TiktokenTokenizer (o200k_base adapter)
│   ├── codec/                # token-LZ77, encoder/decoder (wire format)
│   ├── dictionary.py            # TokDict: trained cross-record dictionary
│   ├── native.py                  # the runtime encoder/decoder pair
│   ├── core.py                      # public compress/decompress/... API
│   └── cli.py                         # `tokpress` command-line entry point
└── tests/
```
