import os
import subprocess
import sys


def _run(*args):
    return subprocess.run([sys.executable, "-m", "tokpress", *args], capture_output=True, text=True)


def test_cli_help():
    res = _run("--help")
    assert res.returncode == 0
    assert "TokPress" in res.stdout
    assert "compress" in res.stdout
    assert "decompress" in res.stdout


def test_cli_compress_decompress_roundtrip(tmp_path):
    test_file = tmp_path / "sample.py"
    content = "import std::io\ndef main():\n    print('Hello TokPress!')\n" * 50
    test_file.write_text(content)

    tokz_file = tmp_path / "sample.py.tokz"
    restored_file = tmp_path / "sample_restored.py"

    res_comp = _run("compress", str(test_file), "-o", str(tokz_file))
    assert res_comp.returncode == 0, res_comp.stderr
    assert tokz_file.exists()
    assert os.path.getsize(tokz_file) < os.path.getsize(test_file)

    res_decomp = _run("decompress", str(tokz_file), "-o", str(restored_file))
    assert res_decomp.returncode == 0, res_decomp.stderr
    assert restored_file.exists()

    assert restored_file.read_text() == content


def test_cli_bench(tmp_path):
    test_file = tmp_path / "sample.py"
    test_file.write_text("def f(x):\n    return x * 2\n" * 50)

    res = _run("bench", str(test_file))
    assert res.returncode == 0, res.stderr
    assert "lossless:" in res.stdout
    assert "OK" in res.stdout


def test_cli_train_dict_splits_jsonl_into_records(tmp_path):
    """A .jsonl sample file must train on its individual records, not on the
    whole file as one giant record -- the project's core many-small-records
    regime. Before the fix, 'samples: N' reflected the number of files, so a
    single jsonl logged only 1 sample."""
    jsonl = tmp_path / "logs.jsonl"
    lines = []
    for i in range(40):
        lines.append(f'{{"user": "u{i}", "action": "click", "ts": {1700000000 + i}}}')
    jsonl.write_text("\n".join(lines) + "\n")

    dict_file = tmp_path / "dict.tokdict"
    res = _run("train-dict", str(dict_file), str(jsonl))
    assert res.returncode == 0, res.stderr
    assert "samples:         40" in res.stdout

    # a held-out record of the same shape must round-trip through the CLI
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text('{"user": "u99", "action": "click", "ts": 1700000099}\n')
    tokz = tmp_path / "heldout.tokz"
    restored = tmp_path / "restored.jsonl"

    assert _run("compress", str(heldout), "-o", str(tokz), "--dict", str(dict_file)).returncode == 0
    assert _run("decompress", str(tokz), "-o", str(restored), "--dict", str(dict_file)).returncode == 0
    assert restored.read_bytes() == heldout.read_bytes()


def test_cli_train_dict_single_binary_record_stays_whole(tmp_path):
    """A binary sample with no newlines must stay one record, not be split."""
    blob = tmp_path / "blob.bin"
    blob.write_bytes(bytes(range(256)) * 8)
    dict_file = tmp_path / "dict.tokdict"

    res = _run("train-dict", str(dict_file), str(blob))
    assert res.returncode == 0, res.stderr
    assert "samples:         1" in res.stdout


def test_cli_pack_unpack_roundtrip(tmp_path):
    """pack: each file is one record, batch-compressed as one adaptive
    stream; unpack: writes each record back out separately."""
    records = [f'{{"user": "u{i}", "action": "click", "ts": {1700000000 + i}}}'.encode() for i in range(30)]
    rec_paths = []
    for i, rec in enumerate(records):
        p = tmp_path / f"rec_{i}.json"
        p.write_bytes(rec)
        rec_paths.append(str(p))

    packed = tmp_path / "batch.tokz"
    res_pack = _run("pack", str(packed), *rec_paths)
    assert res_pack.returncode == 0, res_pack.stderr
    assert "30 records" in res_pack.stdout
    assert os.path.getsize(packed) < sum(len(r) for r in records)

    out_dir = tmp_path / "out"
    res_unpack = _run("unpack", str(packed), str(out_dir))
    assert res_unpack.returncode == 0, res_unpack.stderr
    assert "30 records" in res_unpack.stdout

    restored = sorted(p.name for p in out_dir.iterdir())
    assert len(restored) == 30
    for i, rec in enumerate(records):
        assert (out_dir / f"{i:04d}.rec").read_bytes() == rec


def test_cli_train_vocab_and_use(tmp_path):
    """train-vocab learns a custom byte-level BPE vocab; compress/decompress
    with --vocab round-trips byte-exact."""
    corpus = tmp_path / "corpus.jsonl"
    lines = []
    for i in range(200):
        lines.append(f'{{"user": "u{i}", "action": "click", "ts": {1700000000 + i}}}')
    corpus.write_text("\n".join(lines) + "\n")

    vocab = tmp_path / "vocab.ranks"
    res = _run("train-vocab", str(vocab), str(corpus), "--vocab-size", "1024")
    assert res.returncode == 0, res.stderr
    assert "Trained vocabulary" in res.stdout
    assert "tokens:" in res.stdout

    payload = tmp_path / "record.json"
    payload.write_text('{"user": "u555", "action": "click", "ts": 1700000555}')
    tokz = tmp_path / "record.tokz"
    out = tmp_path / "record.out"

    assert _run("compress", str(payload), "-o", str(tokz), "--vocab", str(vocab)).returncode == 0
    assert _run("decompress", str(tokz), "-o", str(out), "--vocab", str(vocab)).returncode == 0
    assert out.read_bytes() == payload.read_bytes()

    # packing/unpacking with the same custom vocab must also round-trip
    packed = tmp_path / "packed.tokz"
    assert _run("pack", str(packed), str(payload), "--vocab", str(vocab)).returncode == 0
    out_dir = tmp_path / "vout"
    assert _run("unpack", str(packed), str(out_dir), "--vocab", str(vocab)).returncode == 0
    assert (out_dir / "0000.rec").read_bytes() == payload.read_bytes()


def test_cli_tokenize_stats(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog\n" * 50)

    res = _run("tokenize-stats", str(corpus))
    assert res.returncode == 0, res.stderr
    assert "tokens:" in res.stdout
    assert "adjacent MI" in res.stdout


def test_cli_pack_indexed_and_read(tmp_path):
    """pack --indexed writes a TOKBI batch; `read` fetches any single record
    without decoding the others."""
    records = [f'{{"user": "u{i}", "action": "click"}}'.encode() for i in range(20)]
    rec_paths = []
    for i, rec in enumerate(records):
        p = tmp_path / f"r{i}.json"
        p.write_bytes(rec)
        rec_paths.append(str(p))

    packed = tmp_path / "batch.tokz"
    res = _run("pack", str(packed), *rec_paths, "--indexed")
    assert res.returncode == 0, res.stderr
    assert packed.read_bytes().startswith(b"TOKBI")

    for i in [0, 7, 19]:
        out = tmp_path / f"r{i}.out"
        assert _run("read", str(packed), str(i), "-o", str(out)).returncode == 0
        assert out.read_bytes() == records[i]

    # unpack also reads the indexed batch back in full
    out_dir = tmp_path / "out"
    assert _run("unpack", str(packed), str(out_dir)).returncode == 0
    for i, rec in enumerate(records):
        assert (out_dir / f"{i:04d}.rec").read_bytes() == rec


def test_cli_fit(tmp_path):
    """fit trains a vocab + dict end-to-end on a corpus; both work together."""
    corpus = tmp_path / "corpus.jsonl"
    lines = [f'{{"user": "u{i}", "action": "click", "ts": {1700000000 + i}}}' for i in range(120)]
    corpus.write_text("\n".join(lines) + "\n")

    prefix = str(tmp_path / "fit")
    res = _run("fit", prefix, str(corpus), "--vocab-size", "1024")
    assert res.returncode == 0, res.stderr
    assert "Fitted" in res.stdout
    ranks = tmp_path / "fit.ranks"
    tokdict = tmp_path / "fit.tokdict"
    assert ranks.is_file() and tokdict.is_file()

    payload = tmp_path / "rec.json"
    payload.write_text('{"user": "u999", "action": "click", "ts": 1700000999}')
    tokz = tmp_path / "rec.tokz"
    out = tmp_path / "rec.out"
    assert (
        _run("compress", str(payload), "-o", str(tokz), "--vocab", str(ranks), "--dict", str(tokdict)).returncode == 0
    )
    assert _run("decompress", str(tokz), "-o", str(out), "--vocab", str(ranks), "--dict", str(tokdict)).returncode == 0
    assert out.read_bytes() == payload.read_bytes()
