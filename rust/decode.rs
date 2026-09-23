//! Wire-format decoder: mirrors codec/decoder.py (payload -> LZ token stream).

use crate::bitio::BitReader;
use crate::dict::DictData;
use crate::encode::*;
use crate::rans::RansDecoder;
use crate::stats::{Stats, RANS_M, RANS_M_BITS};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

pub struct Decoded {
    pub mode: u8,
    pub flags: u8,
    pub size: u32,
    pub lz: Vec<u32>,
    pub uses_dict: bool,
    pub trailer_pos: usize,
}

fn corrupt(msg: &str) -> String {
    format!("corrupt TokPress stream: {}", msg)
}

fn read_table(r: &mut BitReader, alphabet: u32) -> Result<Stats, String> {
    let ids = r.read_symbol_list()?;
    let mut freq = Vec::with_capacity(ids.len());
    let mut prev: Option<u32> = None;
    for &id in &ids {
        if id >= alphabet || prev.is_some_and(|p| p >= id) {
            return Err(corrupt("invalid frequency table"));
        }
        prev = Some(id);
        freq.push(r.read_bits(RANS_M_BITS)? as u32 + 1);
    }
    let st = Stats::from_active_freq(ids, freq);
    if !st.active.is_empty() && st.total() != RANS_M as u64 {
        return Err(corrupt("frequency table does not sum to RANS_M"));
    }
    Ok(st)
}

fn read_escapes(r: &mut BitReader) -> Result<Vec<u32>, String> {
    let n = r.read_u32()?;
    let mut v = Vec::new();
    for _ in 0..n {
        v.push(r.read_u32()?);
    }
    Ok(v)
}

fn read_rans(r: &mut BitReader, words: &mut Vec<u16>) -> Result<u64, String> {
    let state = r.read_u64()?;
    let n = r.read_u32()?;
    for _ in 0..n {
        words.push(r.read_u16()?);
    }
    Ok(state)
}

struct Esc {
    v: Vec<u32>,
    pos: usize,
}
impl Esc {
    fn next(&mut self) -> Result<u32, String> {
        let s = *self
            .v
            .get(self.pos)
            .ok_or_else(|| corrupt("escape list exhausted"))?;
        self.pos += 1;
        Ok(s)
    }
}

fn norm_all(counts: &[u64], total: u64) -> Result<Stats, String> {
    let all: Vec<u32> = (0..counts.len() as u32).collect();
    Stats::normalize(all, counts, total)
}

/// Sparse context table shared by the PPM decoders. `extra`: slots forced after the real counts.
fn ppm_table(
    counts: &BTreeMap<u32, u64>,
    total: u64,
    k: u32,
    split: bool,
) -> Result<Stats, String> {
    let esc_mass = std::cmp::max(1, counts.len() as u64);
    let mut entries: Vec<(u32, u64)> = counts
        .iter()
        .filter(|(&l, _)| l < k)
        .map(|(&l, &c)| (l, c))
        .collect();
    entries.push((k, esc_mass));
    let mut t = total + esc_mass;
    if split {
        entries.push((k + 1, 1));
        t += 1;
    }
    Stats::normalize_pairs(&entries, t)
}

pub fn decode_stream(
    data: &[u8],
    match_flag: u32,
    dict: Option<&DictData>,
) -> Result<Decoded, String> {
    let mut r = BitReader::new(data);
    let mut magic = [0u8; 4];
    for m in magic.iter_mut() {
        *m = r.read_byte()?;
    }
    if &magic != b"TOKZ" {
        return Err("invalid TokPress stream: bad magic bytes".to_string());
    }
    let version = r.read_byte()?;
    if version != 1 {
        return Err(format!(
            "unsupported TokPress stream version {} (expected 1)",
            version
        ));
    }
    let header_mode = r.read_byte()?;
    let flags = header_mode & !0x1F;
    let mode = header_mode & 0x1F;
    let size = r.read_u32()?;
    let mut out = Decoded {
        mode,
        flags,
        size,
        lz: Vec::new(),
        uses_dict: false,
        trailer_pos: 0,
    };

    if size == 0 || mode == MODE_RAW_FALLBACK {
        return Ok(out);
    }
    let num_lz = r.read_u32()? as usize;
    if num_lz as u64 > 2 * size as u64 + 64 {
        return Err(format!(
            "corrupt TokPress stream: impossible LZ-token count {} for declared uncompressed size {}",
            num_lz, size
        ));
    }
    let ext = flags & MODE_FLAG_EXT != 0;
    let mut lz: Vec<u32> = Vec::with_capacity(num_lz);
    let mut words: Vec<u16> = Vec::new();

    match mode {
        MODE_RAW_TOKENS => {
            let bits = r.read_byte()? as u32;
            if bits == 0 || bits > 32 {
                return Err(corrupt("invalid raw token width"));
            }
            for _ in 0..num_lz {
                lz.push(r.read_bits(bits)? as u32);
            }
        }
        MODE_RANS_SPARSE => {
            let alphabet = match_flag + 2;
            let escape = alphabet - 1;
            let stats = read_table(&mut r, alphabet)?;
            let mut esc = Esc {
                v: read_escapes(&mut r)?,
                pos: 0,
            };
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            for _ in 0..num_lz {
                let mut s = dec.decode_symbol(&stats)?;
                if s == escape {
                    s = esc.next()?;
                }
                lz.push(s);
            }
        }
        MODE_RANS_ADAPTIVE => {
            let chunk = r.read_u32()? as usize;
            if chunk == 0 {
                return Err(corrupt("zero chunk size"));
            }
            let active = r.read_symbol_list()?;
            let mut esc = Esc {
                v: if ext {
                    read_escapes(&mut r)?
                } else {
                    Vec::new()
                },
                pos: 0,
            };
            let k = active.len() + usize::from(ext);
            let escape_local = active.len() as u32;
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            let mut cum = vec![1u64; k];
            let mut cum_total = k as u64;
            let mut pos = 0usize;
            while pos < num_lz {
                let end = std::cmp::min(pos + chunk, num_lz);
                let stats = norm_all(&cum, cum_total)?;
                for _ in pos..end {
                    let l = dec.decode_symbol(&stats)?;
                    let s = if ext && l == escape_local {
                        esc.next()?
                    } else {
                        *active
                            .get(l as usize)
                            .ok_or_else(|| corrupt("symbol index out of range"))?
                    };
                    lz.push(s);
                    cum[l as usize] += 1;
                    cum_total += 1;
                }
                pos = end;
            }
        }
        MODE_RANS_PPM => {
            let chunk = r.read_u32()? as usize;
            if chunk == 0 {
                return Err(corrupt("zero chunk size"));
            }
            let active = r.read_symbol_list()?;
            let mut esc = Esc {
                v: if ext {
                    read_escapes(&mut r)?
                } else {
                    Vec::new()
                },
                pos: 0,
            };
            let escape_slot = active.len() as u32;
            let n_slots = active.len() + usize::from(ext);
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            let mut order_counts = vec![1u64; n_slots];
            let mut ctx_counts: HashMap<u32, BTreeMap<u32, u64>> = HashMap::new();
            let mut ctx_totals: HashMap<u32, u64> = HashMap::new();
            let mut ctx_tables = ContextTables::new(match_flag as usize);
            let mut pos = 0usize;
            while pos < num_lz {
                let end = std::cmp::min(pos + chunk, num_lz);
                let total: u64 = order_counts.iter().sum();
                let o0 = norm_all(&order_counts, total)?;
                let tables = ctx_tables.advance(&ctx_counts, &ctx_totals, |counts, t| {
                    ppm_table(counts, t, escape_slot, false)
                })?;
                for _ in pos..end {
                    let prev = lz.last().copied();
                    let ctx = prev.and_then(|p| tables.get(&p)).map(|r| r.as_ref());
                    let local = match ctx {
                        Some(ct) => {
                            let l = dec.decode_symbol(ct)?;
                            if l == escape_slot {
                                dec.decode_symbol(&o0)?
                            } else {
                                l
                            }
                        }
                        None => dec.decode_symbol(&o0)?,
                    };
                    if ext && local == escape_slot {
                        lz.push(esc.next()?);
                        continue;
                    }
                    let s = *active
                        .get(local as usize)
                        .ok_or_else(|| corrupt("symbol index out of range"))?;
                    lz.push(s);
                    order_counts[local as usize] += 1;
                    if let Some(p) = prev {
                        *ctx_counts.entry(p).or_default().entry(local).or_insert(0) += 1;
                        *ctx_totals.entry(p).or_insert(0) += 1;
                        ctx_tables.mark(p);
                    }
                }
                pos = end;
            }
        }
        MODE_RANS_SPLIT => {
            let num_events = r.read_u32()? as usize;
            if num_events > num_lz {
                return Err(corrupt("impossible event count"));
            }
            let alphabet = match_flag + 2;
            let escape = alphabet - 1;
            let lit = read_table(&mut r, alphabet)?;
            let hi = read_table(&mut r, 256)?;
            let lo = read_table(&mut r, 256)?;
            let len = read_table(&mut r, 256)?;
            let role = read_table(&mut r, 2)?;
            let mut esc = Esc {
                v: read_escapes(&mut r)?,
                pos: 0,
            };
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            for _ in 0..num_events {
                if dec.decode_symbol(&role)? == 0 {
                    let mut s = dec.decode_symbol(&lit)?;
                    if s == escape {
                        s = esc.next()?;
                    }
                    lz.push(s);
                } else {
                    let h = dec.decode_symbol(&hi)?;
                    let l = dec.decode_symbol(&lo)?;
                    let n = dec.decode_symbol(&len)?;
                    lz.extend_from_slice(&[match_flag, h, l, n]);
                }
            }
        }
        MODE_RANS_ADAPTIVE_SPLIT => {
            let num_events = r.read_u32()? as usize;
            let n_lit = r.read_u32()? as usize;
            let chunk = r.read_u32()? as usize;
            if num_events > num_lz || n_lit > num_events || chunk == 0 {
                return Err(corrupt("invalid split header"));
            }
            let distinct = r.read_symbol_list()?;
            let k = distinct.len() + 1;
            let local_escape = (k - 1) as u32;
            let hi = read_table(&mut r, 256)?;
            let lo = read_table(&mut r, 256)?;
            let len = read_table(&mut r, 256)?;
            let role = read_table(&mut r, 2)?;
            let mut esc = Esc {
                v: read_escapes(&mut r)?,
                pos: 0,
            };
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            let mut cum = vec![1u64; k];
            let mut cum_total = k as u64;
            let mut lit_pos = 0usize;
            let mut next_boundary = 0usize;
            let mut stats = Stats::default();
            for _ in 0..num_events {
                if dec.decode_symbol(&role)? == 1 {
                    let h = dec.decode_symbol(&hi)?;
                    let l = dec.decode_symbol(&lo)?;
                    let n = dec.decode_symbol(&len)?;
                    lz.extend_from_slice(&[match_flag, h, l, n]);
                } else {
                    if lit_pos == next_boundary {
                        stats = norm_all(&cum, cum_total)?;
                        next_boundary = std::cmp::min(lit_pos + chunk, n_lit);
                    }
                    let l = dec.decode_symbol(&stats)?;
                    cum[l as usize] += 1;
                    cum_total += 1;
                    lit_pos += 1;
                    let s = if l == local_escape {
                        esc.next()?
                    } else {
                        distinct[l as usize]
                    };
                    lz.push(s);
                }
            }
        }
        MODE_RANS_PPM_SPLIT => {
            let num_events = r.read_u32()? as usize;
            let n_lit = r.read_u32()? as usize;
            let chunk = r.read_u32()? as usize;
            if num_events > num_lz || n_lit > num_events || chunk == 0 {
                return Err(corrupt("invalid split header"));
            }
            let distinct = r.read_symbol_list()?;
            let k = distinct.len() as u32;
            let local_escape = k;
            let hi = read_table(&mut r, 256)?;
            let lo = read_table(&mut r, 256)?;
            let len = read_table(&mut r, 256)?;
            let role = read_table(&mut r, 2)?;
            let mut esc = Esc {
                v: read_escapes(&mut r)?,
                pos: 0,
            };
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            let mut order_counts = vec![1u64; k as usize + 1];
            let mut ctx_counts: HashMap<u32, BTreeMap<u32, u64>> = HashMap::new();
            let mut ctx_totals: HashMap<u32, u64> = HashMap::new();
            let mut ctx_tables = ContextTables::new(k as usize + 1);
            let mut lit_pos = 0usize;
            let mut next_boundary = 0usize;
            let mut o0 = Stats::default();
            let mut tables: HashMap<u32, Rc<Stats>> = HashMap::new();
            let mut prev_lit: Option<u32> = None;
            for _ in 0..num_events {
                if dec.decode_symbol(&role)? == 1 {
                    let h = dec.decode_symbol(&hi)?;
                    let l = dec.decode_symbol(&lo)?;
                    let n = dec.decode_symbol(&len)?;
                    lz.extend_from_slice(&[match_flag, h, l, n]);
                } else {
                    if lit_pos == next_boundary {
                        let total: u64 = order_counts.iter().sum();
                        o0 = norm_all(&order_counts, total)?;
                        tables = ctx_tables.advance(&ctx_counts, &ctx_totals, |counts, t| {
                            ppm_table(counts, t, k, true)
                        })?;
                        next_boundary = std::cmp::min(lit_pos + chunk, n_lit);
                    }
                    let ctx = prev_lit.and_then(|p| tables.get(&p)).map(|r| r.as_ref());
                    let local = match ctx {
                        Some(ct) => {
                            let l = dec.decode_symbol(ct)?;
                            if l == k || l == k + 1 {
                                dec.decode_symbol(&o0)?
                            } else {
                                l
                            }
                        }
                        None => dec.decode_symbol(&o0)?,
                    };
                    order_counts[local as usize] += 1;
                    if let Some(p) = prev_lit {
                        *ctx_counts.entry(p).or_default().entry(local).or_insert(0) += 1;
                        *ctx_totals.entry(p).or_insert(0) += 1;
                        ctx_tables.mark(p);
                    }
                    prev_lit = Some(local);
                    lit_pos += 1;
                    let s = if local == local_escape {
                        esc.next()?
                    } else {
                        distinct[local as usize]
                    };
                    lz.push(s);
                }
            }
        }
        MODE_RANS_DICT => {
            let mut fp = [0u8; 8];
            for b in fp.iter_mut() {
                *b = r.read_byte()?;
            }
            let d = dict.ok_or_else(|| {
                "TokPress stream was compressed with a TokDict dictionary (MODE_RANS_DICT), but no dictionary was supplied to this decoder".to_string()
            })?;
            if fp.as_slice() != d.fingerprint.as_slice() {
                return Err("TokPress stream's TokDict fingerprint does not match the loaded dictionary -- wrong dictionary file for this stream".to_string());
            }
            let mut esc = Esc {
                v: read_escapes(&mut r)?,
                pos: 0,
            };
            let state = read_rans(&mut r, &mut words)?;
            let mut dec = RansDecoder::new(state, &words);
            let escape = d.escape_symbol;
            for i in 0..num_lz {
                let mut ctx = if i > 0 {
                    d.contexts.get(&lz[i - 1])
                } else {
                    None
                };
                let mut sym = 0u32;
                if let Some(ct) = ctx {
                    sym = dec.decode_symbol(ct)?;
                    if sym == escape {
                        ctx = None;
                    }
                }
                if ctx.is_none() {
                    sym = dec.decode_symbol(&d.order0)?;
                    if sym == escape {
                        sym = esc.next()?;
                    }
                }
                lz.push(sym);
            }
            out.uses_dict = true;
        }
        _ => return Err(format!("unknown TokPress mode byte: {}", mode)),
    }

    r.align_to_byte();
    out.trailer_pos = r.byte_offset();
    out.lz = lz;
    Ok(out)
}
