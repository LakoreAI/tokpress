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

    res_comp = _run("compress", str(test_file), "-o", str(tokz_file), "--vocab", "code")
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
