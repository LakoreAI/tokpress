//! Token-level LZ77 with dictionary priming (port of codec/token_lz.py).

pub const MATCH_WINDOW: usize = 32768;
pub const MIN_MATCH_LEN: usize = 3;

#[inline]
fn prefix_hash(t0: u32, t1: u32) -> usize {
    ((t0 as u64)
        .wrapping_mul(2654435761)
        .wrapping_add((t1 as u64).wrapping_mul(40503))
        & 0xFFFF) as usize
}

pub fn encode(tokens: &[u32], dictionary: &[u32], match_flag: u32) -> Vec<u32> {
    let d = dictionary.len();
    let mut combined: Vec<u32> = Vec::with_capacity(d + tokens.len());
    combined.extend_from_slice(dictionary);
    combined.extend_from_slice(tokens);
    let n = combined.len();

    let mut output: Vec<u32> = Vec::with_capacity(tokens.len());
    let mut head: Vec<i64> = vec![-1; 1 << 16];

    let mut p = 0usize;
    while p + 2 < d {
        head[prefix_hash(combined[p], combined[p + 1])] = p as i64;
        p += 1;
    }

    let mut i = d;
    while i < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + 2 < n {
            let h = prefix_hash(combined[i], combined[i + 1]);
            let prev = head[h];
            head[h] = i as i64;
            if prev != -1 {
                let prev_pos = prev as usize;
                if i - prev_pos < MATCH_WINDOW && prev_pos < i {
                    let mut ml = 0usize;
                    while i + ml < n && ml < 255 && combined[prev_pos + ml] == combined[i + ml] {
                        ml += 1;
                    }
                    if ml >= MIN_MATCH_LEN {
                        best_len = ml;
                        best_dist = i - prev_pos;
                    }
                }
            }
        }
        if best_len >= MIN_MATCH_LEN {
            output.push(match_flag);
            output.push(((best_dist >> 8) & 0xFF) as u32);
            output.push((best_dist & 0xFF) as u32);
            output.push((best_len & 0xFF) as u32);
            i += best_len;
        } else if combined[i] == match_flag {
            output.extend_from_slice(&[match_flag, 0, 0, 0]);
            i += 1;
        } else {
            output.push(combined[i]);
            i += 1;
        }
    }
    output
}

pub fn decode(lz_tokens: &[u32], dictionary: &[u32], match_flag: u32) -> Result<Vec<u32>, String> {
    let d = dictionary.len();
    let mut output: Vec<u32> = Vec::with_capacity(d + lz_tokens.len());
    output.extend_from_slice(dictionary);
    let n = lz_tokens.len();
    let mut i = 0usize;
    while i < n {
        if lz_tokens[i] == match_flag && i + 3 < n {
            let dist = ((lz_tokens[i + 1] as usize) << 8) | lz_tokens[i + 2] as usize;
            let length = lz_tokens[i + 3] as usize;
            if dist == 0 && length == 0 {
                output.push(match_flag);
            } else if dist > 0 && dist <= output.len() && length > 0 {
                let start = output.len() - dist;
                for k in 0..length {
                    let v = output[start + k];
                    output.push(v);
                }
            } else {
                return Err(format!(
                    "corrupt LZ stream: invalid match at token {} (distance={}, length={})",
                    i, dist, length
                ));
            }
            i += 4;
        } else {
            output.push(lz_tokens[i]);
            i += 1;
        }
    }
    output.drain(..d);
    Ok(output)
}
