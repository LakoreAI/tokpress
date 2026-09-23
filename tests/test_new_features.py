import json
import random

import tokpress
from tokpress.cli import main


def _records(n=80, seed=1):
    rng = random.Random(seed)
    return [
        json.dumps(
            {"id": i, "user": f"u{rng.randrange(30)}", "status": rng.choice(["ok", "fail"]), "ms": rng.randrange(900)}
        ).encode()
        for i in range(n)
    ]


def test_compress_each_matches_sequential_and_roundtrips():
    recs = _records(40)
    d = tokpress.TokDict.train(recs[:30])
    par = tokpress.compress_each(recs, dictionary=d, workers=4)
    assert par == [tokpress.compress(r, dictionary=d) for r in recs]
    assert tokpress.decompress_each(par, dictionary=d, workers=4) == recs
    assert tokpress.compress_each(["héllo", b"x"], workers=1) == [tokpress.compress("héllo"), tokpress.compress(b"x")]
    assert tokpress.compress_each([]) == []


def test_cover_params_change_dictionary():
    recs = _records(50)
    a = tokpress.TokDict.train(recs)
    b = tokpress.TokDict.train(recs, segment_len=64, dmer_len=4)
    assert a.fingerprint != b.fingerprint


def test_dict_info_cli(tmp_path, capsys):
    recs = _records(30)
    p = tmp_path / "s.jsonl"
    p.write_bytes(b"\n".join(recs))
    out = tmp_path / "d.tokdict"
    assert main(["train-dict", str(out), str(p)]) == 0
    capsys.readouterr()
    assert main(["dict-info", str(out)]) == 0
    info = tokpress.TokDict.load(str(out)).info()
    text = capsys.readouterr().out
    assert info["id"] in text and info["priming_tokens"] > 0
