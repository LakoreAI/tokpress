//! tokpress Rust core, exposed to Python as `tokpress._rs`.

mod bitio;
mod decode;
mod dict;
mod encode;
mod lz;
mod rans;
mod stats;
mod train;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyclass(name = "RsDict", module = "tokpress._rs")]
struct RsDict {
    inner: dict::DictData,
}

#[pymethods]
impl RsDict {
    #[new]
    fn new(
        alphabet_size: u32,
        priming: Vec<u32>,
        order0: Vec<(u32, u32)>,
        contexts: Vec<(u32, Vec<(u32, u32)>)>,
        fingerprint: Vec<u8>,
    ) -> PyResult<Self> {
        Ok(RsDict {
            inner: dict::DictData::new(alphabet_size, priming, order0, contexts, fingerprint)
                .map_err(err)?,
        })
    }
}

#[pyfunction]
fn lz_encode(py: Python<'_>, tokens: Vec<u32>, dictionary: Vec<u32>, match_flag: u32) -> Vec<u32> {
    py.detach(|| lz::encode(&tokens, &dictionary, match_flag))
}

#[pyfunction]
fn lz_decode(
    py: Python<'_>,
    lz_tokens: Vec<u32>,
    dictionary: Vec<u32>,
    match_flag: u32,
) -> PyResult<Vec<u32>> {
    py.detach(|| lz::decode(&lz_tokens, &dictionary, match_flag))
        .map_err(err)
}

#[pyfunction]
fn encode_mode<'py>(
    py: Python<'py>,
    mode: u8,
    lz_tokens: Vec<u32>,
    n_raw: u32,
    match_flag: u32,
    bits: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py
        .detach(|| match mode {
            encode::MODE_RAW_TOKENS => Ok(encode::encode_raw(&lz_tokens, n_raw, bits)),
            encode::MODE_RANS_SPARSE => encode::encode_sparse(&lz_tokens, n_raw, match_flag),
            encode::MODE_RANS_SPLIT => encode::encode_split(&lz_tokens, n_raw, match_flag),
            encode::MODE_RANS_ADAPTIVE_SPLIT => {
                encode::encode_adaptive_split(&lz_tokens, n_raw, match_flag)
            }
            encode::MODE_RANS_ADAPTIVE => encode::encode_adaptive(&lz_tokens, n_raw),
            encode::MODE_RANS_PPM => encode::encode_ppm(&lz_tokens, n_raw),
            encode::MODE_RANS_PPM_SPLIT => encode::encode_ppm_split(&lz_tokens, n_raw, match_flag),
            m => Err(format!("encode_mode: unsupported mode {}", m)),
        })
        .map_err(err)?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(signature = (tokens, n_raw, match_flag, bits, dictionary=None, force_mode=None))]
fn compress_tokens<'py>(
    py: Python<'py>,
    tokens: Vec<u32>,
    n_raw: u32,
    match_flag: u32,
    bits: u32,
    dictionary: Option<PyRef<'py, RsDict>>,
    force_mode: Option<u8>,
) -> PyResult<Bound<'py, PyBytes>> {
    let d = dictionary.as_ref().map(|d| &d.inner);
    let out = py
        .detach(|| encode::compress_tokens(&tokens, n_raw, match_flag, bits, d, force_mode))
        .map_err(err)?;
    Ok(PyBytes::new(py, &out))
}

/// Parse a TOKZ stream and LZ-decode it. Returns (mode, flags, size, tokens, trailer_pos).
#[pyfunction]
#[pyo3(signature = (data, match_flag, dictionary=None))]
fn decode_stream<'py>(
    py: Python<'py>,
    data: &[u8],
    match_flag: u32,
    dictionary: Option<PyRef<'py, RsDict>>,
) -> PyResult<(u8, u8, u32, Vec<u32>, usize)> {
    let d = dictionary.as_ref().map(|d| &d.inner);
    py.detach(|| {
        let dec = decode::decode_stream(data, match_flag, d)?;
        let priming: &[u32] = if dec.uses_dict {
            &d.unwrap().priming
        } else {
            &[]
        };
        let tokens = if dec.lz.is_empty() {
            Vec::new()
        } else {
            lz::decode(&dec.lz, priming, match_flag)?
        };
        Ok::<_, String>((dec.mode, dec.flags, dec.size, tokens, dec.trailer_pos))
    })
    .map_err(err)
}

#[pyfunction]
fn bpe_merges(py: Python<'_>, pieces: Vec<Vec<u8>>, vocab_size: usize) -> Vec<(u32, u32)> {
    py.detach(|| train::bpe_merges(&pieces, vocab_size))
}

#[pyfunction]
fn cover_priming(
    py: Python<'_>,
    tokenized: Vec<Vec<u32>>,
    max_priming: usize,
    segment_len: usize,
    dmer_len: usize,
    discount: f64,
) -> Vec<u32> {
    py.detach(|| train::cover_priming(&tokenized, max_priming, segment_len, dmer_len, discount))
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RsDict>()?;
    m.add_function(wrap_pyfunction!(lz_encode, m)?)?;
    m.add_function(wrap_pyfunction!(lz_decode, m)?)?;
    m.add_function(wrap_pyfunction!(encode_mode, m)?)?;
    m.add_function(wrap_pyfunction!(compress_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(decode_stream, m)?)?;
    m.add_function(wrap_pyfunction!(bpe_merges, m)?)?;
    m.add_function(wrap_pyfunction!(cover_priming, m)?)?;
    Ok(())
}
