//! LSB-first bit writer/reader plus LEB128 varints and delta-coded symbol lists.

pub struct BitWriter {
    buf: Vec<u8>,
    bit_buf: u128,
    bit_count: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter {
            buf: Vec::new(),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    pub fn write_bits(&mut self, value: u64, count: u32) {
        let mask: u64 = if count >= 64 {
            u64::MAX
        } else {
            (1u64 << count) - 1
        };
        self.bit_buf |= ((value & mask) as u128) << self.bit_count;
        self.bit_count += count;
        while self.bit_count >= 8 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    pub fn write_byte(&mut self, v: u8) {
        self.write_bits(v as u64, 8);
    }
    pub fn write_u16(&mut self, v: u16) {
        self.write_bits(v as u64, 16);
    }
    pub fn write_u32(&mut self, v: u32) {
        self.write_bits(v as u64, 32);
    }
    pub fn write_u64(&mut self, v: u64) {
        self.write_bits(v, 64);
    }

    pub fn write_varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                self.write_byte(byte | 0x80);
            } else {
                self.write_byte(byte);
                return;
            }
        }
    }

    pub fn write_symbol_list(&mut self, sorted_ids: &[u32]) {
        self.write_u32(sorted_ids.len() as u32);
        let mut prev = 0u32;
        for &sid in sorted_ids {
            self.write_varint((sid - prev) as u64);
            prev = sid;
        }
    }

    pub fn flush(&mut self) {
        if self.bit_count > 0 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf = 0;
            self.bit_count = 0;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buf
    }
}

pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u128,
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    pub fn read_bits(&mut self, count: u32) -> Result<u64, String> {
        if count > 64 {
            return Err(format!("invalid read width {} (max 64 bits)", count));
        }
        while self.bit_count < count && self.byte_pos < self.data.len() {
            self.bit_buf |= (self.data[self.byte_pos] as u128) << self.bit_count;
            self.bit_count += 8;
            self.byte_pos += 1;
        }
        if self.bit_count < count {
            return Err(format!(
                "truncated bitstream: requested {} bits but only {} remain",
                count, self.bit_count
            ));
        }
        let mask: u128 = (1u128 << count) - 1;
        let result = (self.bit_buf & mask) as u64;
        self.bit_buf >>= count;
        self.bit_count -= count;
        Ok(result)
    }

    pub fn read_byte(&mut self) -> Result<u8, String> {
        Ok(self.read_bits(8)? as u8)
    }
    pub fn read_u16(&mut self) -> Result<u16, String> {
        Ok(self.read_bits(16)? as u16)
    }
    pub fn read_u32(&mut self) -> Result<u32, String> {
        Ok(self.read_bits(32)? as u32)
    }
    pub fn read_u64(&mut self) -> Result<u64, String> {
        self.read_bits(64)
    }

    pub fn read_varint(&mut self) -> Result<u64, String> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.read_byte()?;
            result |= ((byte & 0x7F) as u64).checked_shl(shift).unwrap_or(0);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift > 63 {
                return Err("corrupt bitstream: varint is longer than 64 bits".to_string());
            }
        }
    }

    pub fn read_symbol_list(&mut self) -> Result<Vec<u32>, String> {
        let n = self.read_u32()?;
        let mut ids = Vec::new();
        let mut prev = 0u64;
        for _ in 0..n {
            prev = prev.wrapping_add(self.read_varint()?);
            if prev > u32::MAX as u64 {
                return Err("corrupt bitstream: symbol id out of range".to_string());
            }
            ids.push(prev as u32);
        }
        Ok(ids)
    }

    /// Discard pad bits so the next read starts on a byte boundary.
    pub fn align_to_byte(&mut self) {
        let n = self.bit_count % 8;
        if n != 0 {
            self.bit_buf >>= n;
            self.bit_count -= n;
        }
    }

    /// Byte offset of the next unread byte; only meaningful after `align_to_byte`.
    pub fn byte_offset(&self) -> usize {
        self.byte_pos - (self.bit_count / 8) as usize
    }
}
