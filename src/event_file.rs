use prost::{DecodeError, Message};
use std::io::Read;

use crate::proto::tensorboard::Event;
use crate::tf_record::{ChecksumError, ReadRecordError, TfRecordState};

pub struct EventFileReader<R> {
    /// Wall time of the record most recently read from this event file, or `None` if no records
    /// have been read. Used for determining when to consider this file dead and abandon it.
    last_wall_time: Option<f64>,
    /// State buffer for any pending record read.
    record_state: TfRecordState,
    /// Underlying reader owned by this event file.
    reader: R,
}

pub enum ReadEventError {
    /// The record failed its checksum. This may only be detected if the protocol buffer fails to
    /// decode.
    InvalidRecord(ChecksumError),
    /// The record passed its checksum, but the contained protocol buffer is invalid.
    InvalidProto(DecodeError),
    /// The record is a valid `Event` proto, but its `wall_time` is `NaN`.
    NanWallTime { step: i64 },
    /// An error occurred reading the record. May or may not be fatal.
    ReadRecordError(ReadRecordError),
}

impl From<DecodeError> for ReadEventError {
    fn from(e: DecodeError) -> Self {
        ReadEventError::InvalidProto(e)
    }
}

impl From<ChecksumError> for ReadEventError {
    fn from(e: ChecksumError) -> Self {
        ReadEventError::InvalidRecord(e)
    }
}

impl From<ReadRecordError> for ReadEventError {
    fn from(e: ReadRecordError) -> Self {
        ReadEventError::ReadRecordError(e)
    }
}

impl<R: Read> EventFileReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            last_wall_time: None,
            record_state: TfRecordState::new(),
            reader,
        }
    }

    /// Read the next event from the file. `start_time` will be set to the event's wall time if
    /// `start_time` is not yet set or if the event's wall time precedes it.
    pub fn next(&mut self, start_time: &mut f64) -> Result<Event, ReadEventError> {
        let record = self.record_state.read_record(&mut self.reader)?;
        let event = match Event::decode(&record.data[..]) {
            Ok(ev) => ev,
            Err(err) => {
                record.checksum()?;
                return Err(err)?;
            }
        };
        let wall_time = event.wall_time;
        if wall_time.is_nan() {
            return Err(ReadEventError::NanWallTime { step: event.step });
        }
        if wall_time < *start_time {
            *start_time = wall_time;
        }
        self.last_wall_time = Some(wall_time);
        Ok(event)
    }
}
