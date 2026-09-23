//! 64-bit-state, 16-bit-word rANS at table-log 16 (port of entropy/rans.py).

use crate::stats::{Stats, RANS_M, RANS_M_BITS};

pub const RANS_L: u64 = (RANS_M as u64) << 4;

pub struct RansEncoder {
    pub state: u64,
    pub words: Vec<u16>,
}

impl RansEncoder {
    pub fn new() -> Self {
        RansEncoder {
            state: RANS_L,
            words: Vec::new(),
        }
    }

    #[inline]
    pub fn encode(&mut self, freq: u32, cum: u32) {
        let f = freq as u64;
        let max_x = ((RANS_L >> RANS_M_BITS) << 16) * f;
        while self.state >= max_x {
            self.words.push((self.state & 0xFFFF) as u16);
            self.state >>= 16;
        }
        let q = self.state / f;
        let r = self.state % f;
        self.state = (q << RANS_M_BITS).wrapping_add(r).wrapping_add(cum as u64);
    }

    /// Encode a symbol that the caller guarantees is active in `st`.
    #[inline]
    pub fn encode_symbol(&mut self, sym: u32, st: &Stats) {
        let (f, c) = st
            .lookup(sym)
            .expect("rANS encode of a zero-frequency symbol");
        self.encode(f, c);
    }
}

pub struct RansDecoder<'a> {
    pub state: u64,
    words: &'a [u16],
    remaining: usize,
}

impl<'a> RansDecoder<'a> {
    pub fn new(state: u64, words: &'a [u16]) -> Self {
        RansDecoder {
            state,
            words,
            remaining: words.len(),
        }
    }

    /// Decode one symbol. The table must be non-empty and sum to RANS_M.
    #[inline]
    pub fn decode_symbol(&mut self, st: &Stats) -> Result<u32, String> {
        if st.active.is_empty() {
            return Err(
                "corrupt TokPress stream: decode from an empty frequency table".to_string(),
            );
        }
        let slot = (self.state & (RANS_M as u64 - 1)) as u32;
        let idx = st.cum.partition_point(|&c| c <= slot) - 1;
        let f = st.freq[idx] as u64;
        let c = st.cum[idx];
        self.state = f
            .wrapping_mul(self.state >> RANS_M_BITS)
            .wrapping_add((slot - c) as u64);
        while self.state < RANS_L && self.remaining > 0 {
            self.remaining -= 1;
            self.state = (self.state << 16) | self.words[self.remaining] as u64;
        }
        Ok(st.active[idx])
    }
}
