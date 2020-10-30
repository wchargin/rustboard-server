use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::proto::tensorboard as pb;

pub type Step = i64;
pub type WallTime = f64;

pub type Run = String;
pub type Tag = String;

pub struct Commit {
    pub scalars: Store<ScalarValue>,
    pub tensors: Store<TensorValue>,
    pub blob_sequences: Store<BlobSequenceValue>,
    pub start_times: RwLock<HashMap<Run, WallTime>>,
}

impl Commit {
    pub fn new() -> Self {
        Self {
            scalars: RwLock::new(HashMap::new()),
            tensors: RwLock::new(HashMap::new()),
            blob_sequences: RwLock::new(HashMap::new()),
            start_times: RwLock::new(HashMap::new()),
        }
    }
}

pub type Store<V> = RwLock<HashMap<Run, RwLock<HashMap<Tag, TimeSeries<V>>>>>;
pub type Datum<V> = (WallTime, Result<V, DataLoss>);

pub struct TimeSeries<V> {
    pub metadata: pb::SummaryMetadata,
    pub values: Vec<(Step, Datum<V>)>,
}

impl<V> TimeSeries<V> {
    pub fn new(metadata: pb::SummaryMetadata) -> Self {
        Self {
            metadata,
            values: Vec::new(),
        }
    }
}

pub struct DataLoss;

pub struct ScalarValue(pub f64);
pub struct TensorValue(pub pb::TensorProto);
pub struct BlobSequenceValue(pub Vec<Arc<[u8]>>);
