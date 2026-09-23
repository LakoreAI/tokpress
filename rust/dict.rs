//! Read-only view of a trained TokDict for the Rust codec paths.

use crate::stats::{Stats, RANS_M};
use std::collections::HashMap;

pub struct DictData {
    pub priming: Vec<u32>,
    pub order0: Stats,
    pub contexts: HashMap<u32, Stats>,
    pub escape_symbol: u32,
    pub fingerprint: Vec<u8>,
}

fn build_table(pairs: Vec<(u32, u32)>) -> Result<Stats, String> {
    let mut pairs = pairs;
    pairs.sort_unstable_by_key(|p| p.0);
    let mut sum: u64 = 0;
    let mut active = Vec::with_capacity(pairs.len());
    let mut freq = Vec::with_capacity(pairs.len());
    for (s, f) in pairs {
        if f == 0 {
            continue;
        }
        if let Some(&last) = active.last() {
            if last >= s {
                return Err("TokDict table has duplicate symbol ids".to_string());
            }
        }
        sum += f as u64;
        active.push(s);
        freq.push(f);
    }
    if !active.is_empty() && sum != RANS_M as u64 {
        return Err(format!(
            "TokDict table frequencies sum to {} (expected {})",
            sum, RANS_M
        ));
    }
    Ok(Stats::from_active_freq(active, freq))
}

impl DictData {
    pub fn new(
        alphabet_size: u32,
        priming: Vec<u32>,
        order0: Vec<(u32, u32)>,
        contexts: Vec<(u32, Vec<(u32, u32)>)>,
        fingerprint: Vec<u8>,
    ) -> Result<DictData, String> {
        if alphabet_size == 0 {
            return Err("TokDict alphabet_size must be >= 1".to_string());
        }
        if fingerprint.len() != 8 {
            return Err(format!(
                "TokDict fingerprint must be 8 bytes, got {}",
                fingerprint.len()
            ));
        }
        let mut ctx = HashMap::with_capacity(contexts.len());
        for (c, pairs) in contexts {
            ctx.insert(c, build_table(pairs)?);
        }
        Ok(DictData {
            priming,
            order0: build_table(order0)?,
            contexts: ctx,
            escape_symbol: alphabet_size - 1,
            fingerprint,
        })
    }
}
