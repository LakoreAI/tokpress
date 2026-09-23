//! Wire-format encoders: bit-exact ports of codec/encoder.py's candidate builders.

use crate::bitio::BitWriter;
use crate::dict::DictData;
use crate::lz;
use crate::rans::RansEncoder;
use crate::stats::{Stats, RANS_M, RANS_M_BITS};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

pub const MODE_RAW_TOKENS: u8 = 0;
pub const MODE_RANS_SPARSE: u8 = 1;
pub const MODE_RAW_FALLBACK: u8 = 2;
pub const MODE_RANS_DICT: u8 = 3;
pub const MODE_RANS_ADAPTIVE: u8 = 4;
pub const MODE_RANS_SPLIT: u8 = 5;
pub const MODE_RANS_ADAPTIVE_SPLIT: u8 = 6;
pub const MODE_RANS_PPM: u8 = 7;
pub const MODE_RANS_PPM_SPLIT: u8 = 8;
pub const MODE_FLAG_EXT: u8 = 0x20;

pub const ADAPTIVE_MIN_CHUNK: u64 = 256;
pub const ADAPTIVE_WORK_BUDGET: u64 = 2_000_000;
pub const ADAPTIVE_MIN_SYMBOLS: usize = 512;
pub const MIN_CONTEXT_TRANSITIONS: u64 = 20;

const CAP: usize = RANS_M as usize - 1;

pub fn adaptive_chunk_size(n: usize, k: usize) -> usize {
    std::cmp::max(
        ADAPTIVE_MIN_CHUNK,
        (n as u64 * k as u64) / ADAPTIVE_WORK_BUDGET,
    ) as usize
}

fn write_header(w: &mut BitWriter, mode: u8, n_raw: u32) {
    for &b in b"TOKZ" {
        w.write_byte(b);
    }
    w.write_byte(1);
    w.write_byte(mode);
    w.write_u32(n_raw);
}

fn write_tail(w: &mut BitWriter, enc: &RansEncoder) {
    w.write_u64(enc.state);
    w.write_u32(enc.words.len() as u32);
    for &word in &enc.words {
        w.write_u16(word);
    }
}

fn write_escapes(w: &mut BitWriter, escapes: &[u32]) {
    w.write_u32(escapes.len() as u32);
    for &s in escapes {
        w.write_u32(s);
    }
}

/// Sorted-ascending (symbol, count) pairs of `tokens`.
fn count_sorted(tokens: &[u32]) -> Vec<(u32, u64)> {
    let mut v = tokens.to_vec();
    v.sort_unstable();
    let mut out: Vec<(u32, u64)> = Vec::new();
    for s in v {
        match out.last_mut() {
            Some(l) if l.0 == s => l.1 += 1,
            _ => out.push((s, 1)),
        }
    }
    out
}

/// Keep the `cap` most frequent (ties: lower id first); result ascending by id.
fn top_symbols(counts: &[(u32, u64)], cap: usize) -> Vec<(u32, u64)> {
    let mut v = counts.to_vec();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(cap);
    v.sort_by_key(|p| p.0);
    v
}

fn write_freq_table(w: &mut BitWriter, st: &Stats) {
    w.write_symbol_list(&st.active);
    for &f in &st.freq {
        w.write_bits((f - 1) as u64, RANS_M_BITS);
    }
}

pub struct Events {
    pub positions: Vec<(usize, usize)>,
    pub role_bits: Vec<u32>,
    pub literals: Vec<u32>,
    pub dist_hi: Vec<u32>,
    pub dist_lo: Vec<u32>,
    pub length: Vec<u32>,
}

pub fn split_events(lz: &[u32], match_flag: u32) -> Events {
    let mut e = Events {
        positions: Vec::new(),
        role_bits: Vec::new(),
        literals: Vec::new(),
        dist_hi: Vec::new(),
        dist_lo: Vec::new(),
        length: Vec::new(),
    };
    let n = lz.len();
    let mut i = 0;
    while i < n {
        if lz[i] == match_flag && i + 3 < n {
            e.role_bits.push(1);
            e.dist_hi.push(lz[i + 1]);
            e.dist_lo.push(lz[i + 2]);
            e.length.push(lz[i + 3]);
            e.positions.push((i, 4));
            i += 4;
        } else {
            e.role_bits.push(0);
            e.literals.push(lz[i]);
            e.positions.push((i, 1));
            i += 1;
        }
    }
    e
}

pub struct MetaStats {
    pub role: Stats,
    pub hi: Stats,
    pub lo: Stats,
    pub len: Stats,
}

fn meta_stats(e: &Events) -> Result<MetaStats, String> {
    Ok(MetaStats {
        role: Stats::count_symbols(&e.role_bits, 2)?,
        hi: Stats::count_symbols(&e.dist_hi, 256)?,
        lo: Stats::count_symbols(&e.dist_lo, 256)?,
        len: Stats::count_symbols(&e.length, 256)?,
    })
}

fn write_meta_tables(w: &mut BitWriter, m: &MetaStats) {
    write_freq_table(w, &m.hi);
    write_freq_table(w, &m.lo);
    write_freq_table(w, &m.len);
    write_freq_table(w, &m.role);
}

fn encode_match(enc: &mut RansEncoder, lz: &[u32], start: usize, m: &MetaStats) {
    enc.encode_symbol(lz[start + 3], &m.len);
    enc.encode_symbol(lz[start + 2], &m.lo);
    enc.encode_symbol(lz[start + 1], &m.hi);
    enc.encode_symbol(1, &m.role);
}

pub fn encode_raw(lz: &[u32], n_raw: u32, bits: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_header(&mut w, MODE_RAW_TOKENS, n_raw);
    w.write_u32(lz.len() as u32);
    w.write_byte(bits as u8);
    for &t in lz {
        w.write_bits(t as u64, bits);
    }
    w.finish()
}

/// Literal table with the RANS_M-1 most frequent symbols kept, the rest folded into `escape`.
fn capped_literal_stats(literals: &[u32], escape: u32) -> Result<Stats, String> {
    let mut counts = count_sorted(literals);
    let total = literals.len() as u64;
    if counts.len() > CAP {
        let kept = top_symbols(&counts, CAP);
        let kept_total: u64 = kept.iter().map(|p| p.1).sum();
        counts = kept;
        counts.push((escape, total - kept_total));
    }
    let active: Vec<u32> = counts.iter().map(|p| p.0).collect();
    let cs: Vec<u64> = counts.iter().map(|p| p.1).collect();
    Stats::normalize(active, &cs, total)
}

pub fn encode_sparse(lz: &[u32], n_raw: u32, match_flag: u32) -> Result<Vec<u8>, String> {
    let escape = match_flag + 1;
    let stats = capped_literal_stats(lz, escape)?;
    let has_escape = stats.has(escape);
    let mut coded: Vec<u32> = Vec::with_capacity(lz.len());
    let mut escapes = Vec::new();
    for &s in lz {
        if !has_escape || stats.has(s) {
            coded.push(s);
        } else {
            coded.push(escape);
            escapes.push(s);
        }
    }
    let mut enc = RansEncoder::new();
    for &s in coded.iter().rev() {
        enc.encode_symbol(s, &stats);
    }
    let mut w = BitWriter::new();
    write_header(&mut w, MODE_RANS_SPARSE, n_raw);
    w.write_u32(lz.len() as u32);
    write_freq_table(&mut w, &stats);
    write_escapes(&mut w, &escapes);
    write_tail(&mut w, &enc);
    Ok(w.finish())
}

pub fn encode_split(lz: &[u32], n_raw: u32, match_flag: u32) -> Result<Vec<u8>, String> {
    let escape = match_flag + 1;
    let ev = split_events(lz, match_flag);
    let meta = meta_stats(&ev)?;
    let lit_stats = if ev.literals.is_empty() {
        Stats::default()
    } else {
        capped_literal_stats(&ev.literals, escape)?
    };

    let mut enc = RansEncoder::new();
    let mut escapes = Vec::new();
    for &(start, span) in ev.positions.iter().rev() {
        if span == 4 {
            encode_match(&mut enc, lz, start, &meta);
        } else {
            let sym = lz[start];
            if lit_stats.has(sym) {
                enc.encode_symbol(sym, &lit_stats);
            } else {
                enc.encode_symbol(escape, &lit_stats);
                escapes.push(sym);
            }
            enc.encode_symbol(0, &meta.role);
        }
    }
    escapes.reverse();

    let mut w = BitWriter::new();
    write_header(&mut w, MODE_RANS_SPLIT, n_raw);
    w.write_u32(lz.len() as u32);
    w.write_u32(ev.positions.len() as u32);
    write_freq_table(&mut w, &lit_stats);
    write_meta_tables(&mut w, &meta);
    write_escapes(&mut w, &escapes);
    write_tail(&mut w, &enc);
    Ok(w.finish())
}

/// Local (dense) alphabet over `literals` capped at `cap` real symbols; returns
/// (distinct ascending, coded local indices, escapes in forward order, local escape index).
fn local_literals(literals: &[u32], cap: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let counts = count_sorted(literals);
    let distinct: Vec<u32> = if counts.len() > cap {
        top_symbols(&counts, cap).iter().map(|p| p.0).collect()
    } else {
        counts.iter().map(|p| p.0).collect()
    };
    let k = distinct.len() as u32;
    let local_escape = k;
    let mut coded = Vec::with_capacity(literals.len());
    let mut escapes = Vec::new();
    for &s in literals {
        match distinct.binary_search(&s) {
            Ok(i) => coded.push(i as u32),
            Err(_) => {
                coded.push(local_escape);
                escapes.push(s);
            }
        }
    }
    (distinct, coded, escapes, local_escape)
}

pub fn encode_adaptive_split(lz: &[u32], n_raw: u32, match_flag: u32) -> Result<Vec<u8>, String> {
    let ev = split_events(lz, match_flag);
    let meta = meta_stats(&ev)?;
    let n_lit = ev.literals.len();
    let (distinct, coded, escapes, local_escape) = local_literals(&ev.literals, CAP);
    let k = distinct.len() + 1;
    debug_assert_eq!(local_escape as usize, k - 1);

    let chunk_size = adaptive_chunk_size(n_lit, k);
    let mut cum_counts = vec![1u64; k];
    let mut cum_total = k as u64;
    let all: Vec<u32> = (0..k as u32).collect();
    let n_chunks = if n_lit == 0 {
        0
    } else {
        n_lit.div_ceil(chunk_size)
    };
    let mut chunk_stats: Vec<Stats> = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        chunk_stats.push(Stats::normalize(all.clone(), &cum_counts, cum_total)?);
        let end = std::cmp::min((c + 1) * chunk_size, n_lit);
        for j in c * chunk_size..end {
            cum_counts[coded[j] as usize] += 1;
            cum_total += 1;
        }
    }

    let mut enc = RansEncoder::new();
    let mut lit_idx = n_lit;
    for &(start, span) in ev.positions.iter().rev() {
        if span == 4 {
            encode_match(&mut enc, lz, start, &meta);
        } else {
            lit_idx -= 1;
            enc.encode_symbol(coded[lit_idx], &chunk_stats[lit_idx / chunk_size]);
            enc.encode_symbol(0, &meta.role);
        }
    }

    let mut w = BitWriter::new();
    write_header(&mut w, MODE_RANS_ADAPTIVE_SPLIT, n_raw);
    w.write_u32(lz.len() as u32);
    w.write_u32(ev.positions.len() as u32);
    w.write_u32(n_lit as u32);
    w.write_u32(chunk_size as u32);
    w.write_symbol_list(&distinct);
    write_meta_tables(&mut w, &meta);
    write_escapes(&mut w, &escapes);
    write_tail(&mut w, &enc);
    Ok(w.finish())
}

/// Alphabet shared by the pure adaptive and PPM modes: ascending active ids,
/// local index per symbol, whether an escape slot exists.
fn capped_alphabet(lz: &[u32]) -> (Vec<u32>, bool) {
    let counts = count_sorted(lz);
    if counts.len() > CAP {
        (
            top_symbols(&counts, CAP).iter().map(|p| p.0).collect(),
            true,
        )
    } else {
        (counts.iter().map(|p| p.0).collect(), false)
    }
}

pub fn encode_adaptive(lz: &[u32], n_raw: u32) -> Result<Vec<u8>, String> {
    let (active, has_escape) = capped_alphabet(lz);
    encode_adaptive_with(lz, n_raw, active, has_escape)
}

fn encode_adaptive_with(
    lz: &[u32],
    n_raw: u32,
    active: Vec<u32>,
    has_escape: bool,
) -> Result<Vec<u8>, String> {
    let n = lz.len();
    let escape_local = active.len() as u32;
    let k = active.len() + if has_escape { 1 } else { 0 };
    let mut coded = Vec::with_capacity(n);
    let mut escapes = Vec::new();
    for &s in lz {
        match active.binary_search(&s) {
            Ok(i) => coded.push(i as u32),
            Err(_) => {
                coded.push(escape_local);
                escapes.push(s);
            }
        }
    }
    let chunk_size = adaptive_chunk_size(n, k);
    let mut cum_counts = vec![1u64; k];
    let mut cum_total = k as u64;
    let all: Vec<u32> = (0..k as u32).collect();
    let n_chunks = if n == 0 { 0 } else { n.div_ceil(chunk_size) };
    let mut chunk_stats = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        chunk_stats.push(Stats::normalize(all.clone(), &cum_counts, cum_total)?);
        for j in c * chunk_size..std::cmp::min((c + 1) * chunk_size, n) {
            cum_counts[coded[j] as usize] += 1;
            cum_total += 1;
        }
    }
    let mut enc = RansEncoder::new();
    for j in (0..n).rev() {
        enc.encode_symbol(coded[j], &chunk_stats[j / chunk_size]);
    }
    let mode = MODE_RANS_ADAPTIVE | if has_escape { MODE_FLAG_EXT } else { 0 };
    let mut w = BitWriter::new();
    write_header(&mut w, mode, n_raw);
    w.write_u32(n as u32);
    w.write_u32(chunk_size as u32);
    w.write_symbol_list(&active);
    if has_escape {
        write_escapes(&mut w, &escapes);
    }
    write_tail(&mut w, &enc);
    Ok(w.finish())
}

pub(crate) type CtxCounts = HashMap<u32, BTreeMap<u32, u64>>;

/// Incrementally maintained per-chunk context tables. A context table only
/// changes when that context's monotonically growing counts change, so at each
/// chunk boundary only the contexts touched since the previous boundary need
/// rebuilding; every other table is reused by `Rc`. This yields exactly the
/// tables a full rebuild-every-chunk would produce, but skips most of the
/// (dominant) table-construction work: the number of rebuilt tables drops from
/// n_chunks * eligible_contexts to ~the number of distinct (context, chunk)
/// touch events. Shared by the PPM encoder and decoder.
pub(crate) struct ContextTables {
    current: HashMap<u32, Rc<Stats>>,
    dirty: Vec<u32>,
    stamp: Vec<u32>,
    generation: u32,
}

impl ContextTables {
    /// `max_key` is the largest context key that can occur (so `stamp` can be a
    /// plain array rather than a map, keeping the per-token `mark` cheap).
    pub(crate) fn new(max_key: usize) -> Self {
        ContextTables {
            current: HashMap::new(),
            dirty: Vec::new(),
            stamp: vec![u32::MAX; max_key + 1],
            generation: 0,
        }
    }

    #[inline]
    pub(crate) fn mark(&mut self, ctx: u32) {
        let i = ctx as usize;
        if i >= self.stamp.len() {
            // Out-of-range only for a corrupt stream whose escape value exceeds
            // the symbol range the decoder sized `stamp` for; skipping keeps a
            // malformed input from panicking. Valid streams never hit this.
            return;
        }
        if self.stamp[i] != self.generation {
            self.stamp[i] = self.generation;
            self.dirty.push(ctx);
        }
    }

    /// Rebuild the dirty/eligible tables with `build`, then return a snapshot
    /// for the coming chunk (cheap `Rc` clones) and start a new generation.
    pub(crate) fn advance<F>(
        &mut self,
        counts: &CtxCounts,
        totals: &HashMap<u32, u64>,
        mut build: F,
    ) -> Result<HashMap<u32, Rc<Stats>>, String>
    where
        F: FnMut(&BTreeMap<u32, u64>, u64) -> Result<Stats, String>,
    {
        for &ctx in &self.dirty {
            let (Some(c), Some(&t)) = (counts.get(&ctx), totals.get(&ctx)) else {
                continue;
            };
            if t < MIN_CONTEXT_TRANSITIONS {
                continue;
            }
            self.current.insert(ctx, Rc::new(build(c, t)?));
        }
        self.dirty.clear();
        self.generation = self.generation.wrapping_add(1);
        Ok(self.current.clone())
    }
}

pub fn encode_ppm(lz: &[u32], n_raw: u32) -> Result<Vec<u8>, String> {
    let (active, has_escape) = capped_alphabet(lz);
    encode_ppm_with(lz, n_raw, active, has_escape)
}

fn encode_ppm_with(
    lz: &[u32],
    n_raw: u32,
    active: Vec<u32>,
    has_escape: bool,
) -> Result<Vec<u8>, String> {
    let n = lz.len();
    let order0_escape = active.len() as u32;
    let n_slots = active.len() + if has_escape { 1 } else { 0 };
    let chunk_size = adaptive_chunk_size(n, n_slots);
    let local_of = |s: u32| active.binary_search(&s).ok().map(|i| i as u32);

    let mut order_counts = vec![1u64; n_slots];
    let mut ctx_counts: CtxCounts = HashMap::new();
    let mut ctx_totals: HashMap<u32, u64> = HashMap::new();
    let n_chunks = if n == 0 { 0 } else { n.div_ceil(chunk_size) };
    let all0: Vec<u32> = (0..n_slots as u32).collect();

    let max_ctx = lz.iter().copied().max().unwrap_or(0) as usize;
    let mut ctx_tables = ContextTables::new(max_ctx);
    let mut chunk_order: Vec<Stats> = Vec::with_capacity(n_chunks);
    let mut chunk_ctx: Vec<HashMap<u32, Rc<Stats>>> = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let total: u64 = order_counts.iter().sum();
        chunk_order.push(Stats::normalize(all0.clone(), &order_counts, total)?);
        chunk_ctx.push(ctx_tables.advance(&ctx_counts, &ctx_totals, |counts, t| {
            let esc_mass = std::cmp::max(1, counts.len() as u64);
            let mut entries: Vec<(u32, u64)> = counts.iter().map(|(&l, &c)| (l, c)).collect();
            entries.push((order0_escape, esc_mass));
            Stats::normalize_pairs(&entries, t + esc_mass)
        })?);
        for j in c * chunk_size..std::cmp::min((c + 1) * chunk_size, n) {
            let Some(local) = local_of(lz[j]) else {
                continue;
            };
            order_counts[local as usize] += 1;
            if j > 0 {
                let prev = lz[j - 1];
                *ctx_counts
                    .entry(prev)
                    .or_default()
                    .entry(local)
                    .or_insert(0) += 1;
                *ctx_totals.entry(prev).or_insert(0) += 1;
                ctx_tables.mark(prev);
            }
        }
    }

    let mut enc = RansEncoder::new();
    let mut escapes = Vec::new();
    for c in (0..n_chunks).rev() {
        let o0 = &chunk_order[c];
        let tables = &chunk_ctx[c];
        let lo = c * chunk_size;
        let hi = std::cmp::min(lo + chunk_size, n);
        for j in (lo..hi).rev() {
            let sym = lz[j];
            let local = local_of(sym);
            let ctx: Option<&Stats> = if j > 0 {
                tables.get(&lz[j - 1]).map(|r| r.as_ref())
            } else {
                None
            };
            if let (Some(ct), Some(l)) = (ctx, local) {
                if ct.has(l) {
                    enc.encode_symbol(l, ct);
                    continue;
                }
            }
            match local {
                None => {
                    enc.encode_symbol(order0_escape, o0);
                    escapes.push(sym);
                }
                Some(l) => enc.encode_symbol(l, o0),
            }
            if let Some(ct) = ctx {
                enc.encode_symbol(order0_escape, ct);
            }
        }
    }
    escapes.reverse();

    let mode = MODE_RANS_PPM | if has_escape { MODE_FLAG_EXT } else { 0 };
    let mut w = BitWriter::new();
    write_header(&mut w, mode, n_raw);
    w.write_u32(n as u32);
    w.write_u32(chunk_size as u32);
    w.write_symbol_list(&active);
    if has_escape {
        write_escapes(&mut w, &escapes);
    }
    write_tail(&mut w, &enc);
    Ok(w.finish())
}

pub fn encode_ppm_split(lz: &[u32], n_raw: u32, match_flag: u32) -> Result<Vec<u8>, String> {
    let ev = split_events(lz, match_flag);
    let meta = meta_stats(&ev)?;
    let n_lit = ev.literals.len();
    let (distinct, coded, escapes, local_escape) = local_literals(&ev.literals, CAP - 1);
    let k = distinct.len() as u32;
    debug_assert_eq!(local_escape, k);

    let chunk_size = adaptive_chunk_size(n_lit, k as usize + 1);
    let mut order_counts = vec![1u64; k as usize + 1];
    let mut ctx_counts: CtxCounts = HashMap::new();
    let mut ctx_totals: HashMap<u32, u64> = HashMap::new();
    let n_chunks = if n_lit == 0 {
        0
    } else {
        n_lit.div_ceil(chunk_size)
    };
    let all0: Vec<u32> = (0..=k).collect();

    let mut ctx_tables = ContextTables::new(k as usize + 1);
    let mut chunk_order: Vec<Stats> = Vec::with_capacity(n_chunks);
    let mut chunk_ctx: Vec<HashMap<u32, Rc<Stats>>> = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let total: u64 = order_counts.iter().sum();
        chunk_order.push(Stats::normalize(all0.clone(), &order_counts, total)?);
        chunk_ctx.push(ctx_tables.advance(&ctx_counts, &ctx_totals, |counts, t| {
            let esc_mass = std::cmp::max(1, counts.len() as u64);
            // Real locals 0..k-1 keep their counts; slot k (ctx escape) and k+1
            // (local escape) are overwritten, matching the reference quirk.
            let mut entries: Vec<(u32, u64)> = counts
                .iter()
                .filter(|(&l, _)| l < k)
                .map(|(&l, &cn)| (l, cn))
                .collect();
            entries.push((k, esc_mass));
            entries.push((k + 1, 1));
            Stats::normalize_pairs(&entries, t + esc_mass + 1)
        })?);
        for j in c * chunk_size..std::cmp::min((c + 1) * chunk_size, n_lit) {
            let local = coded[j];
            order_counts[local as usize] += 1;
            if j > 0 {
                let prev = coded[j - 1];
                *ctx_counts
                    .entry(prev)
                    .or_default()
                    .entry(local)
                    .or_insert(0) += 1;
                *ctx_totals.entry(prev).or_insert(0) += 1;
                ctx_tables.mark(prev);
            }
        }
    }

    let mut enc = RansEncoder::new();
    let mut lit_idx = n_lit;
    for &(start, span) in ev.positions.iter().rev() {
        if span == 4 {
            encode_match(&mut enc, lz, start, &meta);
        } else {
            lit_idx -= 1;
            let local = coded[lit_idx];
            let c = lit_idx / chunk_size;
            let o0 = &chunk_order[c];
            let ctx: Option<&Stats> = if lit_idx > 0 {
                chunk_ctx[c].get(&coded[lit_idx - 1]).map(|r| r.as_ref())
            } else {
                None
            };
            if local == local_escape {
                enc.encode_symbol(local_escape, o0);
                if let Some(ct) = ctx {
                    enc.encode_symbol(k + 1, ct);
                }
            } else if ctx.is_some_and(|ct| ct.has(local)) {
                enc.encode_symbol(local, ctx.unwrap());
            } else {
                enc.encode_symbol(local, o0);
                if let Some(ct) = ctx {
                    enc.encode_symbol(k, ct);
                }
            }
            enc.encode_symbol(0, &meta.role);
        }
    }

    let mut w = BitWriter::new();
    write_header(&mut w, MODE_RANS_PPM_SPLIT, n_raw);
    w.write_u32(lz.len() as u32);
    w.write_u32(ev.positions.len() as u32);
    w.write_u32(n_lit as u32);
    w.write_u32(chunk_size as u32);
    w.write_symbol_list(&distinct);
    write_meta_tables(&mut w, &meta);
    write_escapes(&mut w, &escapes);
    write_tail(&mut w, &enc);
    Ok(w.finish())
}

pub fn encode_dict(lz: &[u32], n_raw: u32, d: &DictData) -> Vec<u8> {
    let mut enc = RansEncoder::new();
    let mut escapes = Vec::new();
    let esc = d.escape_symbol;
    for i in (0..lz.len()).rev() {
        let sym = lz[i];
        let ctx = if i > 0 {
            d.contexts.get(&lz[i - 1])
        } else {
            None
        };
        if let Some(ct) = ctx {
            if ct.has(sym) {
                enc.encode_symbol(sym, ct);
                continue;
            }
        }
        if d.order0.has(sym) {
            enc.encode_symbol(sym, &d.order0);
        } else {
            enc.encode_symbol(esc, &d.order0);
            escapes.push(sym);
        }
        if let Some(ct) = ctx {
            enc.encode_symbol(esc, ct);
        }
    }
    escapes.reverse();
    let mut w = BitWriter::new();
    write_header(&mut w, MODE_RANS_DICT, n_raw);
    w.write_u32(lz.len() as u32);
    for &b in &d.fingerprint {
        w.write_byte(b);
    }
    write_escapes(&mut w, &escapes);
    write_tail(&mut w, &enc);
    w.finish()
}

/// Build exactly one mode's payload, or Err when that mode would not be built
/// for this record (short record, or dict mode with no dictionary). Used by
/// `force_mode` so it skips building the candidates it is going to discard.
fn encode_one(
    mode: u8,
    tokens: &[u32],
    lz: &[u32],
    n_raw: u32,
    match_flag: u32,
    bits: u32,
    dict: Option<&DictData>,
) -> Result<Vec<u8>, String> {
    match mode {
        MODE_RAW_TOKENS => Ok(encode_raw(lz, n_raw, bits)),
        MODE_RANS_SPARSE => encode_sparse(lz, n_raw, match_flag),
        MODE_RANS_SPLIT => encode_split(lz, n_raw, match_flag),
        MODE_RANS_ADAPTIVE_SPLIT => encode_adaptive_split(lz, n_raw, match_flag),
        MODE_RANS_ADAPTIVE | MODE_RANS_PPM | MODE_RANS_PPM_SPLIT => {
            if lz.len() < ADAPTIVE_MIN_SYMBOLS {
                return Err(format!("mode {} was not built for this record", mode));
            }
            match mode {
                MODE_RANS_ADAPTIVE => encode_adaptive(lz, n_raw),
                MODE_RANS_PPM => encode_ppm(lz, n_raw),
                _ => encode_ppm_split(lz, n_raw, match_flag),
            }
        }
        MODE_RANS_DICT => match dict {
            Some(d) => {
                let dlz = lz::encode(tokens, &d.priming, match_flag);
                Ok(encode_dict(&dlz, n_raw, d))
            }
            None => Err(format!("mode {} was not built for this record", mode)),
        },
        m => Err(format!("mode {} was not built for this record", m)),
    }
}

/// Build every applicable candidate in the reference order and return the
/// smallest (first on ties), or -- when `force_mode` is set -- build and
/// return only that one candidate's payload.
pub fn compress_tokens(
    tokens: &[u32],
    n_raw: u32,
    match_flag: u32,
    bits: u32,
    dict: Option<&DictData>,
    force_mode: Option<u8>,
) -> Result<Vec<u8>, String> {
    let lz = lz::encode(tokens, &[], match_flag);
    if let Some(fm) = force_mode {
        return encode_one(fm, tokens, &lz, n_raw, match_flag, bits, dict);
    }
    let mut cands: Vec<(u8, Vec<u8>)> = vec![
        (MODE_RAW_TOKENS, encode_raw(&lz, n_raw, bits)),
        (MODE_RANS_SPARSE, encode_sparse(&lz, n_raw, match_flag)?),
    ];
    cands.push((MODE_RANS_SPLIT, encode_split(&lz, n_raw, match_flag)?));
    cands.push((
        MODE_RANS_ADAPTIVE_SPLIT,
        encode_adaptive_split(&lz, n_raw, match_flag)?,
    ));
    if lz.len() >= ADAPTIVE_MIN_SYMBOLS {
        // Adaptive and PPM share the same capped alphabet; count it once instead
        // of re-sorting the whole stream inside each encoder.
        let (active, has_escape) = capped_alphabet(&lz);
        cands.push((
            MODE_RANS_ADAPTIVE,
            encode_adaptive_with(&lz, n_raw, active.clone(), has_escape)?,
        ));
        cands.push((
            MODE_RANS_PPM,
            encode_ppm_with(&lz, n_raw, active, has_escape)?,
        ));
        cands.push((
            MODE_RANS_PPM_SPLIT,
            encode_ppm_split(&lz, n_raw, match_flag)?,
        ));
    }
    if let Some(d) = dict {
        let dlz = lz::encode(tokens, &d.priming, match_flag);
        cands.push((MODE_RANS_DICT, encode_dict(&dlz, n_raw, d)));
    }
    let mut best = 0usize;
    for (i, c) in cands.iter().enumerate() {
        if c.1.len() < cands[best].1.len() {
            best = i;
        }
    }
    Ok(cands.swap_remove(best).1)
}
