# Bench corpora

Vendored corpus files used by `scripts/bench.py`, so its numbers reproduce
from a clean checkout (previously these were read from a sibling checkout
and an ad hoc download -- see docs/TODO.md).

- `canterbury/` -- the Canterbury Corpus (alice29.txt, asyoulik.txt,
  cp.html, fields.c, grammar.lsp, kennedy.xls, lcet10.txt, plrabn12.txt,
  ptt5, sum, xargs.1), the classical general-purpose compression litmus
  test. https://corpus.canterbury.ac.nz/
- `real_data/` -- real, schema-homogeneous records used for the
  many-small-records and trained-dictionary regimes:
  - `json_heldout.jsonl` -- 230 real structured-log records (~200-540 B each),
    used for the target-regime progression and TokDict training (184/46 split).
  - `small_records.jsonl` -- a smaller (50-record) sample of the same log
    schema (35/15 split), used as a lower-training-data comparison.
  - `real_python_code.py` -- a ~8000-line Python source file, split into
    function/class snippets for the cross-schema experiment.
  - `real_distinct_logs.json`, `general_heldout.bin`, `code_heldout.txt` --
    additional real samples from the original data collection.

## Not vendored

`long_text/` (enwik8 and War and Peace prefixes) is deliberately not
vendored: enwik8 is the 100 MB Hutter Prize / Large Text Compression
Benchmark test set (mattmahoney.net/dc/text.html) and War and Peace is a
public-domain Project Gutenberg novel. `scripts/bench.py` reads them from
`data/bench/long_text/{enwik8_2mb.txt,warpeace_1mb.txt}` if present and
prints SKIPPED otherwise. To reproduce the long-text rows, download a
prefix yourself, e.g.:

    curl -L https://mattmahoney.net/dc/enwik8.zip -o /tmp/enwik8.zip
    unzip -p /tmp/enwik8.zip | head -c 2097152 > data/bench/long_text/enwik8_2mb.txt

and fetch `warpeace_1mb.txt` from a Project Gutenberg mirror of the novel.
