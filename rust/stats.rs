//! Frequency-table normalisation (bit-exact port of entropy/frequency.py) in a sparse layout.

pub const RANS_M_BITS: u32 = 16;
pub const RANS_M: u32 = 1 << RANS_M_BITS;

/// A normalised frequency table over a sparse alphabet: `active` is ascending,
/// `freq`/`cum` are parallel to it.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub active: Vec<u32>,
    pub freq: Vec<u32>,
    pub cum: Vec<u32>,
    dense: bool,
}

impl Stats {
    pub fn from_active_freq(active: Vec<u32>, freq: Vec<u32>) -> Stats {
        let mut cum = Vec::with_capacity(freq.len());
        let mut c = 0u32;
        for &f in &freq {
            cum.push(c);
            c = c.wrapping_add(f);
        }
        let dense = active
            .last()
            .is_none_or(|&l| l as usize + 1 == active.len());
        Stats {
            active,
            freq,
            cum,
            dense,
        }
    }

    #[inline]
    pub fn find(&self, sym: u32) -> Option<usize> {
        if self.dense {
            if (sym as usize) < self.active.len() {
                Some(sym as usize)
            } else {
                None
            }
        } else {
            self.active.binary_search(&sym).ok()
        }
    }

    /// (freq, cum) of a symbol, or None when its frequency is zero.
    #[inline]
    pub fn lookup(&self, sym: u32) -> Option<(u32, u32)> {
        self.find(sym).map(|i| (self.freq[i], self.cum[i]))
    }

    #[inline]
    pub fn has(&self, sym: u32) -> bool {
        self.find(sym).is_some()
    }

    pub fn total(&self) -> u64 {
        self.freq.iter().map(|&f| f as u64).sum()
    }

    /// Scale sparse raw counts (`active` ascending, all counts > 0) so they sum to RANS_M,
    /// distributing rounding drift round-robin exactly like the Python reference.
    pub fn normalize(
        active: Vec<u32>,
        counts: &[u64],
        total_symbols: u64,
    ) -> Result<Stats, String> {
        if total_symbols == 0 || active.is_empty() {
            return Ok(Stats::default());
        }
        let distinct = active.len();
        if distinct > RANS_M as usize {
            return Err(format!(
                "{} distinct symbols exceeds RANS_M={}: every active symbol needs freq >= 1, so their \
                 frequencies can never be rebalanced down to sum to RANS_M.",
                distinct, RANS_M
            ));
        }
        let target = RANS_M as u64;
        let max_allowed = target - 1;
        let mut freq = Vec::with_capacity(distinct);
        let mut current_sum: u64 = 0;
        for &c in counts.iter().take(distinct) {
            let mut f = c * target / total_symbols;
            if f == 0 {
                f = 1;
            } else if f > max_allowed {
                f = max_allowed;
            }
            freq.push(f);
            current_sum += f;
        }
        if current_sum != target {
            let true_ceiling = target - (distinct as u64 - 1);
            let mut diff = target as i64 - current_sum as i64;
            let mut cursor = 0usize;
            while diff != 0 {
                let i = cursor % distinct;
                if diff > 0 && freq[i] < true_ceiling {
                    freq[i] += 1;
                    diff -= 1;
                } else if diff < 0 && freq[i] > 1 {
                    freq[i] -= 1;
                    diff += 1;
                }
                cursor += 1;
            }
        }
        Ok(Stats::from_active_freq(
            active,
            freq.into_iter().map(|f| f as u32).collect(),
        ))
    }

    /// Normalise a dense count array (`counts[i]` for symbol i).
    pub fn normalize_dense(counts: &[u64], total_symbols: u64) -> Result<Stats, String> {
        let mut active = Vec::new();
        let mut cs = Vec::new();
        for (i, &c) in counts.iter().enumerate() {
            if c > 0 {
                active.push(i as u32);
                cs.push(c);
            }
        }
        Stats::normalize(active, &cs, total_symbols)
    }

    /// Count `symbols` (ignoring any >= alphabet_size, like SymbolStats.count_symbols) and normalise.
    pub fn count_symbols(symbols: &[u32], alphabet_size: u32) -> Result<Stats, String> {
        let mut counts = vec![0u64; alphabet_size as usize];
        let mut total = 0u64;
        for &s in symbols {
            if s < alphabet_size {
                counts[s as usize] += 1;
                total += 1;
            }
        }
        Stats::normalize_dense(&counts, total)
    }
}
