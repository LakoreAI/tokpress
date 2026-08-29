"""Token-level LZ77 with dictionary priming (zstd/FemtoZip-style shared cross-record match history)."""

MATCH_FLAG = 0x0FFF  # 4095, a symbol above normal vocab range
MATCH_WINDOW = 32768  # keeps distances < 2**16 (2-byte distance field)
MIN_MATCH_LEN = 5  # matches of length 3-4 are a net loss given the 4-token match tuple


def _prefix_hash(t0: int, t1: int) -> int:
    return (t0 * 2654435761 + t1 * 40503) & 0xFFFF


class TokenLZMatch:
    @staticmethod
    def encode(tokens: list[int], dictionary: list[int] = (), match_flag: int = MATCH_FLAG) -> list[int]:
        d = len(dictionary)
        combined = list(dictionary) + list(tokens)
        n = len(combined)

        output: list[int] = []
        head: dict[int, int] = {}

        p = 0
        while p + 2 < d:
            head[_prefix_hash(combined[p], combined[p + 1])] = p
            p += 1

        i = d
        while i < n:
            best_len = 0
            best_dist = 0
            if i + 2 < n:
                h = _prefix_hash(combined[i], combined[i + 1])
                prev_pos = head.get(h, -1)
                head[h] = i
                if prev_pos != -1 and (i - prev_pos < MATCH_WINDOW) and (prev_pos < i):
                    dist = i - prev_pos
                    match_len = 0
                    while (
                        (i + match_len < n)
                        and (match_len < 255)
                        and (combined[prev_pos + match_len] == combined[i + match_len])
                    ):
                        match_len += 1
                    if match_len >= MIN_MATCH_LEN:
                        best_len = match_len
                        best_dist = dist
            if best_len >= MIN_MATCH_LEN:
                output.append(match_flag)
                output.append((best_dist >> 8) & 0xFF)
                output.append(best_dist & 0xFF)
                output.append(best_len & 0xFF)
                i += best_len
            elif combined[i] == match_flag:
                output.extend([match_flag, 0, 0, 0])
                i += 1
            else:
                output.append(combined[i])
                i += 1
        return output

    @staticmethod
    def decode(lz_tokens: list[int], dictionary: list[int] = (), match_flag: int = MATCH_FLAG) -> list[int]:
        output = list(dictionary)
        d = len(dictionary)
        n = len(lz_tokens)

        i = 0
        while i < n:
            if lz_tokens[i] == match_flag and i + 3 < n:
                d_high = lz_tokens[i + 1]
                d_low = lz_tokens[i + 2]
                length = lz_tokens[i + 3]
                dist = (d_high << 8) | d_low
                if dist == 0 and length == 0:
                    output.append(match_flag)
                else:
                    start = len(output) - dist
                    for k in range(length):
                        output.append(output[start + k])
                i += 4
            else:
                output.append(lz_tokens[i])
                i += 1

        return output[d:]
