use byteorder::{ByteOrder, LittleEndian};
use crc::crc32;
use std::fmt::{self, Debug};
use std::io::{self, Read};

const LENGTH_CRC_OFFSET: usize = 8;
const DATA_OFFSET: usize = LENGTH_CRC_OFFSET + 4;
const HEADER_LENGTH: usize = DATA_OFFSET;
const FOOTER_LENGTH: usize = 4;

/// State for reading one `TfRecord`, potentially over multiple attempts to handle growing,
/// partially flushed files.
//
// From TensorFlow `record_writer.cc` comments:
// Format of a single record:
//  uint64    length
//  uint32    masked crc of length
//  byte      data[length]
//  uint32    masked crc of data
pub struct TfRecordState {
    /// TFRecord header: little-endian u64 length, u32 length-CRC. This vector always has capacity
    /// `HEADER_LENGTH`.
    //
    // TODO(@wchargin): Consider replacing with an inline `[u8; HEADER_LENGTH]` plus `usize` length
    // field to avoid a level of memory indirection.
    header: Vec<u8>,
    /// Everything past the header in the TFRecord: the data buffer, plus a little-endian u32 CRC
    /// of the data buffer. Once `header.len() == HEADER_LENGTH`, this will have capacity equal to
    /// the data length plus `FOOTER_LENGTH`; before then, it will have no capacity.
    data_plus_footer: Vec<u8>,
}

impl TfRecordState {
    /// Create an empty `TfRecordState`, ready to read a record from its beginning. This allocates
    /// a vector with 12 bytes of capacity, which will be reused for all records read with this
    /// state value.
    pub fn new() -> Self {
        TfRecordState {
            header: Vec::with_capacity(HEADER_LENGTH),
            data_plus_footer: Vec::new(),
        }
    }
}

impl Default for TfRecordState {
    fn default() -> Self {
        Self::new()
    }
}

/// A TFRecord with a data buffer and expected checksum. The checksum may or may not match the
/// actual contents.
#[derive(Debug)]
pub struct TfRecord {
    /// The payload of the TFRecord.
    pub data: Vec<u8>,
    data_crc: MaskedCrc,
}

/// A CRC-32C (Castagnoli) checksum after a masking permutation. This is the checksum format used
/// by TFRecords.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct MaskedCrc(pub u32);

impl Debug for MaskedCrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MaskedCrc(0x{:x?})", self.0)
    }
}

const CRC_MASK_DELTA: u32 = 0xa282ead8;

/// Apply a masking operation to an unmasked CRC.
fn mask_crc(crc: u32) -> MaskedCrc {
    MaskedCrc(((crc >> 15) | (crc << 17)).wrapping_add(CRC_MASK_DELTA))
}

impl MaskedCrc {
    /// Compute a `MaskedCrc` from a data buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustboard_server::tf_record::MaskedCrc;
    ///
    /// // Checksum extracted from real TensorFlow event file with record:
    /// // tf.compat.v1.Event(file_version=b"CRC test, one two")
    /// let data = b"\x1a\x11CRC test, one two";
    /// assert_eq!(MaskedCrc::compute(data), MaskedCrc(0x5794d08a));
    /// ```
    pub fn compute(bytes: &[u8]) -> Self {
        mask_crc(crc32::checksum_castagnoli(bytes))
    }
}

/// A buffer's checksum was computed, but it did not match the expected value.
#[derive(Debug, PartialEq, Eq)]
pub struct ChecksumError {
    /// The actual checksum of the buffer.
    got: MaskedCrc,
    /// The expected checksum.
    want: MaskedCrc,
}

impl TfRecord {
    /// Validates the integrity of the record by computing its CRC-32C and checking it against the
    /// expected value.
    pub fn checksum(&self) -> Result<(), ChecksumError> {
        let got = MaskedCrc::compute(&self.data);
        let want = self.data_crc;
        if got == want {
            Ok(())
        } else {
            Err(ChecksumError { got, want })
        }
    }
}

/// Error returned by [`TfRecordState::read_record`].
#[derive(Debug)]
pub enum ReadRecordError {
    /// Length field failed checksum. The file is corrupt, and reading must abort.
    BadLengthCrc(ChecksumError),
    /// No fatal errors so far, but the record is not complete. Call `read_record` again with the
    /// same state buffer once new data may have been written to the file.
    Truncated,
    /// Record is too large to be represented in memory on this system. In principle, it would be
    /// possible to recover from this error, but in practice this should rarely occur since
    /// serialized protocol buffers do not exceed 2 GiB in size. Thus, no recovery codepath has
    /// been implemented, so reading must abort.
    TooLarge(u64),
    /// Underlying I/O error. May be retryable if the underlying error is.
    Io(io::Error),
}

impl From<io::Error> for ReadRecordError {
    fn from(io: io::Error) -> Self {
        ReadRecordError::Io(io)
    }
}

impl TfRecordState {
    /// Attempt to read a TFRecord, pausing gracefully in the face of truncations. If the record is
    /// truncated, the result is a `Truncated` error, and the state buffer will be updated to
    /// contain the prefix of the raw record that was read. The same state buffer should be passed
    /// to a subsequent call to `read_record` that it may continue where it left off. If the record
    /// is read successfully, this `TfRecordState` is left at its default value (equivalent to
    /// `TfRecordState::new`, but without re-allocating) and may be reused by the caller to read a
    /// fresh record.
    ///
    /// The record's length field is always validated against its checksum, but the full data is
    /// only validated if you call `checksum()` on the resulting record.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustboard_server::tf_record::{ReadRecordError, TfRecordState};
    /// use std::io::Cursor;
    ///
    /// let mut buf: Vec<u8> = Vec::new();
    /// buf.extend(b"\x18\x00\x00\x00\x00\x00\x00\x00"); // length: 24 bytes
    /// buf.extend(b"\xa3\x7f\x4b\x22"); // length checksum (0x224b7fa3)
    /// let contents = b"\x09\x00\x00\x80\x38\x99\xd6\xd7\x41\x1a\x0dbrain.Event:2";
    /// buf.extend(&contents[..5]); // file truncated mid-write
    ///
    /// let mut st = TfRecordState::new();
    ///
    /// // First attempt: read what we can, then encounter truncation.
    /// assert!(matches!(
    ///     st.read_record(&mut Cursor::new(buf)),
    ///     Err(ReadRecordError::Truncated)
    /// ));
    ///
    /// let mut buf: Vec<u8> = Vec::new();
    /// buf.extend(&contents[5..]); // rest of the payload
    /// buf.extend(b"\x12\x4b\x36\xab"); // data checksum (0xab364b12)
    ///
    /// // Second read: read the rest of the record.
    /// let record = st.read_record(&mut Cursor::new(buf)).unwrap();
    /// assert_eq!(record.data, contents);
    /// assert_eq!(record.checksum(), Ok(()));
    /// ```
    pub fn read_record<R: Read>(&mut self, reader: &mut R) -> Result<TfRecord, ReadRecordError> {
        if self.header.len() < self.header.capacity() {
            read_remaining(reader, &mut self.header)?;

            let (length_buf, length_crc_buf) = self.header.split_at(LENGTH_CRC_OFFSET);
            let length_crc = MaskedCrc(LittleEndian::read_u32(length_crc_buf));
            let actual_crc = MaskedCrc::compute(length_buf);
            if length_crc != actual_crc {
                return Err(ReadRecordError::BadLengthCrc(ChecksumError {
                    got: actual_crc,
                    want: length_crc,
                }));
            }

            let length = LittleEndian::read_u64(length_buf);
            let data_plus_footer_length_u64 = length + (FOOTER_LENGTH as u64);
            let data_plus_footer_length = data_plus_footer_length_u64 as usize;
            if data_plus_footer_length as u64 != data_plus_footer_length_u64 {
                return Err(ReadRecordError::TooLarge(length));
            }
            self.data_plus_footer.reserve_exact(data_plus_footer_length);
        }

        if self.data_plus_footer.len() < self.data_plus_footer.capacity() {
            read_remaining(reader, &mut self.data_plus_footer)?;
        }

        let data_length = self.data_plus_footer.len() - FOOTER_LENGTH;
        let data_crc_buf = self.data_plus_footer.split_off(data_length);
        let data = std::mem::take(&mut self.data_plus_footer);
        let data_crc = MaskedCrc(LittleEndian::read_u32(&data_crc_buf));
        self.header.clear(); // reset; caller may use this again
        Ok(TfRecord { data, data_crc })
    }
}

/// Fill `buf`'s remaining capacity from `reader`, or fail with `Truncated` if the reader is dry.
fn read_remaining<R: Read>(reader: &mut R, buf: &mut Vec<u8>) -> Result<(), ReadRecordError> {
    let want = buf.capacity() - buf.len();
    reader.take(want as u64).read_to_end(buf)?;
    if buf.len() < buf.capacity() {
        return Err(ReadRecordError::Truncated);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io::Cursor;

    /// A reader that delegates to a sequence of cursors, reading from each in turn and simulating
    /// EOF after each one.
    struct ScriptedReader(VecDeque<Cursor<Vec<u8>>>);

    impl ScriptedReader {
        fn new<I: IntoIterator<Item = Vec<u8>>>(vecs: I) -> Self {
            ScriptedReader(vecs.into_iter().map(Cursor::new).collect())
        }
    }

    impl Read for ScriptedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let sub = match self.0.front_mut() {
                None => return Ok(0),
                Some(sub) => sub,
            };
            let read = sub.read(buf)?;
            if read == 0 {
                self.0.pop_front();
            }
            Ok(read)
        }
    }

    mod scripted_reader_tests {
        #[test]
        fn test() {
            let mut sr = super::ScriptedReader::new(vec![
                (0..=3).collect(),
                vec![],
                (4..=9).collect(),
                vec![10],
            ]);
            let expected = [
                // Read first buffer, with underread.
                (vec![0, 1, 2], 3),
                (vec![3, 0, 0], 1),
                (vec![0, 0, 0], 0),
                // Read second buffer, which is empty.
                (vec![0, 0, 0], 0),
                // Read third buffer, exactly.
                (vec![4, 5, 6], 3),
                (vec![7, 8, 9], 3),
                (vec![0, 0, 0], 0),
                // Read fourth buffer, with underread.
                (vec![10, 0, 0], 1),
                (vec![0, 0, 0], 0),
                // Read past end of buffer list.
                (vec![0, 0, 0], 0),
            ];
            for (expected_buf, expected_n) in &expected {
                use std::io::Read;
                let mut buf = vec![0; 3];
                let actual_n = sr.read(&mut buf);
                assert_eq!(actual_n.unwrap(), *expected_n);
                assert_eq!(&buf, expected_buf);
            }
        }
    }

    #[test]
    fn test_success() {
        // Event file with `tf.summary.scalar("accuracy", 0.99, step=77)`
        // dumped via `xxd logs/*`.
        let record_1a = b"\x09\x00\x00\x80\x38\x99";
        let record_1b = b"\xd6\xd7\x41\x1a\x0dbrain.Event:2";
        let record_2 = b"\
            \x09\xc4\x05\xb7\x3d\x99\xd6\xd7\x41\
            \x10\x4d\x2a\x25\
            \x0a\x23\x0a\x08accuracy\
            \x42\x0a\x08\x01\x12\x00\x22\x04\xa4\x70\x7d\x3f\x4a\
            \x0b\x0a\x09\x0a\x07scalars\
        ";
        let mut sr = ScriptedReader::new(vec![
            b"\x18\x00\x00\x00\x00".to_vec(),
            b"\x00\x00\x00\xa3\x7f\x4b".to_vec(),
            std::iter::once(b'\x22')
                .chain(record_1a.iter().copied())
                .collect(),
            (record_1b
                .iter()
                .copied()
                .chain(b"\x12\x4b\x36\xab\x32\x00".iter().copied()))
            .collect(),
            b"\x00\x00\x00\x00\x00\x00\x24\x19\x56\xec"
                .iter()
                .copied()
                .chain(record_2.iter().copied())
                .chain(b"\xa5\x5b\x64\x33".iter().copied())
                .collect(),
        ]);

        let mut st = TfRecordState::new();

        #[derive(Debug)]
        enum TestCase {
            Truncated,
            Record(Vec<u8>),
        }
        use TestCase::*;

        let steps: Vec<TestCase> = vec![
            Truncated,
            Truncated,
            Truncated,
            Record(
                record_1a
                    .iter()
                    .copied()
                    .chain(record_1b.iter().copied())
                    .collect(),
            ),
            Truncated,
            Record(record_2.to_vec()),
        ];
        for (i, step) in steps.into_iter().enumerate() {
            let result = st.read_record(&mut sr);
            match (step, result) {
                (Truncated, Err(ReadRecordError::Truncated)) => (),
                (Record(v), Ok(r)) if v == r.data => {
                    r.checksum()
                        .unwrap_or_else(|e| panic!("step {}: checksum failure: {:?}", i + 1, e));
                }
                (step, result) => {
                    panic!("step {}: got {:?}, want {:?}", i + 1, result, step);
                }
            }
        }
    }

    #[test]
    fn test_length_crc_mismatch() {
        let mut file = Vec::new();
        file.extend(b"\x18\x00\x00\x00\x00\x00\x00\x00");
        file.extend(b"\x99\x7f\x4b\x55");
        file.extend(b"123456789abcdef012345678");
        file.extend(b"\x00\x00\x00\x00");

        let mut st = TfRecordState::new();
        match st.read_record(&mut Cursor::new(file)) {
            Err(ReadRecordError::BadLengthCrc(ChecksumError {
                got: MaskedCrc(0x224b7fa3),
                want: MaskedCrc(0x554b7f99),
            })) => (),
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn test_data_crc_mismatch() {
        let mut file = Vec::new();
        file.extend(b"\x18\x00\x00\x00\x00\x00\x00\x00");
        file.extend(b"\xa3\x7f\x4b\x22");
        file.extend(b"123456789abcdef012345678");
        file.extend(b"\xdf\x9b\x57\x13"); // 0x13579bdf

        let mut st = TfRecordState::new();
        let record = st.read_record(&mut Cursor::new(file)).expect("read_record");
        assert_eq!(record.data, b"123456789abcdef012345678".to_vec());
        match record.checksum() {
            Err(ChecksumError {
                want: MaskedCrc(0x13579bdf),
                got: _,
            }) => (),
            other => panic!("{:?}", other),
        }
    }
}
