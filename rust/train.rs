//! Offline trainers: byte-level BPE merges and the COVER priming-segment picker.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

const STRIDE: u64 = 1 << 16;

fn pair_key(a: u32, b: u32) -> u64 {
    a as u64 * STRIDE + b as u64
}

/// Greedy BPE over unique pre-tokenised pieces. Merges never cross piece
/// boundaries; the winner each round is the highest count, then the lowest
/// pair key -- identical to the reference trainer's selection rule.
pub fn bpe_merges(pieces: &[Vec<u8>], vocab_size: usize) -> Vec<(u32, u32)> {
    let mut uniq: HashMap<&[u8], usize> = HashMap::new();
    let mut words: Vec<Vec<u32>> = Vec::new();
    let mut freq: Vec<i64> = Vec::new();
    for p in pieces {
        if p.is_empty() {
            continue;
        }
        match uniq.get(p.as_slice()) {
            Some(&i) => freq[i] += 1,
            None => {
                uniq.insert(p.as_slice(), words.len());
                words.push(p.iter().map(|&b| b as u32).collect());
                freq.push(1);
            }
        }
    }

    let mut counts: HashMap<u64, i64> = HashMap::new();
    let mut index: HashMap<u64, Vec<usize>> = HashMap::new();
    for (wi, w) in words.iter().enumerate() {
        for pair in w.windows(2) {
            let k = pair_key(pair[0], pair[1]);
            *counts.entry(k).or_insert(0) += freq[wi];
            index.entry(k).or_default().push(wi);
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<u64>)> =
        counts.iter().map(|(&k, &c)| (c, Reverse(k))).collect();

    let mut merges = Vec::new();
    let num_merges = vocab_size.saturating_sub(256);
    for m in 0..num_merges {
        let best = loop {
            match heap.pop() {
                None => break None,
                Some((c, Reverse(k))) => {
                    if counts.get(&k).copied().unwrap_or(0) == c && c > 0 {
                        break Some(k);
                    }
                }
            }
        };
        let Some(key) = best else { break };
        let (a, b) = ((key / STRIDE) as u32, (key % STRIDE) as u32);
        let new_id = 256 + m as u32;
        merges.push((a, b));

        let mut affected = index.remove(&key).unwrap_or_default();
        affected.sort_unstable();
        affected.dedup();
        let mut touched: Vec<u64> = Vec::new();
        for wi in affected {
            let w = &words[wi];
            let mut merged: Vec<u32> = Vec::with_capacity(w.len());
            let mut i = 0;
            while i < w.len() {
                if i + 1 < w.len() && w[i] == a && w[i + 1] == b {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(w[i]);
                    i += 1;
                }
            }
            if merged.len() == w.len() {
                continue;
            }
            let f = freq[wi];
            for pair in w.windows(2) {
                let k = pair_key(pair[0], pair[1]);
                *counts.get_mut(&k).unwrap() -= f;
                touched.push(k);
            }
            for pair in merged.windows(2) {
                let k = pair_key(pair[0], pair[1]);
                *counts.entry(k).or_insert(0) += f;
                index.entry(k).or_default().push(wi);
                touched.push(k);
            }
            words[wi] = merged;
        }
        touched.sort_unstable();
        touched.dedup();
        for k in touched {
            let c = counts.get(&k).copied().unwrap_or(0);
            if c > 0 {
                heap.push((c, Reverse(k)));
            } else {
                counts.remove(&k);
            }
        }
    }
    merges
}

/// The zstd-COVER-style segment picker (see dictionary.py `_priming_tokens_cover`).
pub fn cover_priming(
    tokenized: &[Vec<u32>],
    max_priming: usize,
    segment_len: usize,
    dmer_len: usize,
    discount: f64,
) -> Vec<u32> {
    let mut ids: HashMap<&[u32], u32> = HashMap::new();
    let mut freq: Vec<u64> = Vec::new();
    // per-record d-mer id at each start offset
    let mut rec_dmers: Vec<Vec<u32>> = Vec::with_capacity(tokenized.len());
    for toks in tokenized {
        let mut v = Vec::new();
        if toks.len() >= dmer_len {
            for i in 0..=toks.len() - dmer_len {
                let key = &toks[i..i + dmer_len];
                let id = *ids.entry(key).or_insert_with(|| {
                    freq.push(0);
                    (freq.len() - 1) as u32
                });
                freq[id as usize] += 1;
                v.push(id);
            }
        }
        rec_dmers.push(v);
    }
    if freq.is_empty() {
        return Vec::new();
    }

    let stride = std::cmp::max(1, segment_len / 2);
    let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
    for (ridx, toks) in tokenized.iter().enumerate() {
        let n = toks.len();
        if n < dmer_len {
            continue;
        }
        if n <= segment_len {
            candidates.push((ridx, 0, n));
            continue;
        }
        let mut start = 0;
        while start < n {
            let end = std::cmp::min(start + segment_len, n);
            if end - start >= dmer_len {
                candidates.push((ridx, start, end));
            }
            if end == n {
                break;
            }
            start += stride;
        }
    }

    let mut priming: Vec<u32> = Vec::new();
    let mut remaining = candidates;
    while !remaining.is_empty() && priming.len() < max_priming {
        let mut best_score = 0.0f64;
        let mut best_pos: Option<usize> = None;
        for (pos, &(ridx, start, end)) in remaining.iter().enumerate() {
            let mut s = 0.0f64;
            for &id in &rec_dmers[ridx][start..=end - dmer_len] {
                s += (freq[id as usize] as f64).ln_1p();
            }
            if s > best_score {
                best_score = s;
                best_pos = Some(pos);
            }
        }
        let Some(pos) = best_pos else { break };
        let (ridx, start, end) = remaining.remove(pos);
        let take = std::cmp::min(end - start, max_priming - priming.len());
        priming.extend_from_slice(&tokenized[ridx][start..start + take]);
        for &id in &rec_dmers[ridx][start..=end - dmer_len] {
            freq[id as usize] = (freq[id as usize] as f64 * discount) as u64;
        }
    }
    priming
}
