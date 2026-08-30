import pytest

from tokpress.codec.token_lz import TokenLZMatch


def test_token_lz_basic_repetition_roundtrip():
    lz = TokenLZMatch()
    pattern = list(range(256, 261))
    tokens = pattern * 100  # 500 tokens total
    encoded = lz.encode(tokens, [])
    assert len(encoded) < len(tokens)
    decoded = lz.decode(encoded, [])
    assert decoded == tokens


def test_token_lz_dictionary_primed_matching():
    # A record that is a pure substring of the dictionary, with zero
    # internal repetition of its own -- it can only compress by matching
    # into the shared cross-record dictionary history, proving dictionary
    # priming (not just in-record LZ) is what's doing the work.
    lz = TokenLZMatch()
    dictionary = list(range(300, 350))  # 50 tokens
    record = list(range(310, 320))  # 10 tokens, a substring of dictionary

    encoded = lz.encode(record, dictionary)
    assert len(encoded) < len(record)

    decoded = lz.decode(encoded, dictionary)
    assert decoded == record


def test_token_lz_no_match_below_min_match_len():
    # No repeated 2-token prefixes anywhere -> nothing to match, literals only.
    lz = TokenLZMatch()
    tokens = list(range(300, 320))
    encoded = lz.encode(tokens, [])
    assert encoded == tokens
    assert lz.decode(encoded, []) == tokens


def test_token_lz_empty_input():
    lz = TokenLZMatch()
    assert lz.encode([], []) == []
    assert lz.decode([], []) == []


def test_token_lz_decode_rejects_bad_match_distance():
    """Regression: a malformed stream with a match distance beyond the decoded
    history used to index the output list negatively (silently corrupting the
    result, or crashing with IndexError) instead of raising."""
    lz = TokenLZMatch()
    flag = lz.match_flag
    # distance 0xFF00 >> nothing in history: start = len(output) - 65280 < 0
    corrupt = [flag, 0xFF, 0x00, 5]
    with pytest.raises(ValueError):
        lz.decode(corrupt, [])


def test_token_lz_decode_rejects_zero_length_match():
    """A (match_flag, dist_hi, dist_lo, length) tuple with length==0 and a
    nonzero distance is malformed: the encoder only emits that tuple shape
    for the escaped-literal case (dist==0, length==0)."""
    lz = TokenLZMatch()
    flag = lz.match_flag
    corrupt = [flag, 0x00, 0x10, 0]
    with pytest.raises(ValueError):
        lz.decode(corrupt, [])
