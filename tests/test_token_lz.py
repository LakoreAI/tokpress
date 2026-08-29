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
