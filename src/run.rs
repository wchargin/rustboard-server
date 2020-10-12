use byteorder::{ByteOrder, LittleEndian};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use crate::commit::{self, Commit};
use crate::event_file::EventFileReader;
use crate::proto::tensorboard as pb;
use crate::reservoir::StageReservoir;

type Step = i64;

const RUN_GRAPH_NAME: &'static str = "__run_graph__";
const GRAPHS_PLUGIN_NAME: &'static str = "graphs";
const SCALARS_PLUGIN_NAME: &'static str = "scalars";

const COMMIT_DELAY: Duration = Duration::from_secs(5);
const COMMIT_WEIGHT: u64 = 1000; // cf. `TimeSeries::staged_weight()`

pub struct RunLoader {
    /// The earliest event `wall_time` seen in any event file in this run. Initially `INFINITY`,
    /// never `NaN`. May decrease as new events are read, but in practice this is expected to be
    /// the wall time of the first `file_version` event in the first event file.
    start_time: f64,
    /// The last value of `start_time` that was committed, if any. Used to avoid needlessly
    /// grabbing a write-lock on the `Commit::start_times` map, since in practice `start_time` is
    /// expected to be write-once.
    last_committed_start_time: Option<f64>,
    /// The event file loaders that comprise this run.
    files: BTreeMap<PathBuf, EventFileReader<Box<dyn Read>>>,
    time_series: HashMap<String, TimeSeries>,
}

pub struct TimeSeries {
    metadata: pb::SummaryMetadata,
    next_commit: Instant,
    rsv: StageReservoir<StageValue>,
}

struct StageValue {
    wall_time: f64,
    tag: String,
    payload: StagePayload,
}

enum StagePayload {
    GraphDef(Vec<u8>),
    SummaryValue {
        metadata: Option<pb::SummaryMetadata>,
        value: pb::summary::value::Value,
    },
}

impl StagePayload {
    fn take_metadata(&mut self) -> pb::SummaryMetadata {
        match self {
            StagePayload::GraphDef(_) => pb::SummaryMetadata {
                plugin_data: Some(pb::summary_metadata::PluginData {
                    plugin_name: GRAPHS_PLUGIN_NAME.to_string(),
                    content: Vec::new(),
                }),
                display_name: String::new(),
                summary_description: String::new(),
                data_class: pb::DataClass::BlobSequence.into(),
            },
            StagePayload::SummaryValue { metadata, value } => match metadata.take() {
                Some(md) if md.data_class != pb::DataClass::Unknown.into() => md,
                _ if matches!(value, pb::summary::value::Value::SimpleValue(_)) => {
                    pb::SummaryMetadata {
                        plugin_data: Some(pb::summary_metadata::PluginData {
                            plugin_name: SCALARS_PLUGIN_NAME.to_string(),
                            content: Vec::new(),
                        }),
                        display_name: String::new(),
                        summary_description: String::new(),
                        data_class: pb::DataClass::Scalar.into(),
                    }
                }
                Some(mut md) => {
                    if let pb::SummaryMetadata {
                        plugin_data:
                            Some(pb::summary_metadata::PluginData {
                                ref plugin_name, ..
                            }),
                        ..
                    } = md
                    {
                        if plugin_name == SCALARS_PLUGIN_NAME {
                            md.data_class = pb::DataClass::Scalar.into();
                        }
                    };
                    md
                }
                None => pb::SummaryMetadata::default(),
            },
        }
    }
}

impl TimeSeries {
    fn new(metadata: pb::SummaryMetadata) -> Self {
        let capacity =
            match pb::DataClass::from_i32(metadata.data_class).unwrap_or(pb::DataClass::Unknown) {
                pb::DataClass::Unknown => 1,
                pb::DataClass::Scalar => 1000,
                pb::DataClass::Tensor => 100,
                pb::DataClass::BlobSequence => 10,
            };
        Self {
            metadata,
            next_commit: Instant::now() + COMMIT_DELAY,
            rsv: StageReservoir::new(capacity),
        }
    }

    fn data_class(&self) -> pb::DataClass {
        pb::DataClass::from_i32(self.metadata.data_class).unwrap_or(pb::DataClass::Unknown)
    }

    fn staged_weight(&self) -> u64 {
        use pb::DataClass;
        let item_weight = match self.data_class() {
            DataClass::Unknown => return 0,
            DataClass::Scalar => 1,
            DataClass::Tensor => 10,
            DataClass::BlobSequence => 100,
        };
        let preemption_weight = if self.rsv.staged_preemption() { 100 } else { 0 };
        (self.rsv.staged_items().len() as u64) * item_weight + preemption_weight
    }

    fn commit(&mut self, run_name: &str, tag_name: &str, commit: &Commit) {
        use pb::DataClass;
        match self.data_class() {
            DataClass::Unknown => return,
            DataClass::Scalar => {
                self.commit_to(run_name, tag_name, &commit.scalars, Self::commit_scalar)
            }
            DataClass::Tensor => unimplemented!(),
            DataClass::BlobSequence => unimplemented!(),
        }
        self.next_commit = Instant::now() + COMMIT_DELAY;
    }

    fn commit_to<V, F: FnMut(StageValue) -> commit::Datum<V>>(
        &mut self,
        run_name: &str,
        tag_name: &str,
        store: &commit::Store<V>,
        f: F,
    ) {
        let run_map = store.read();
        let mut tag_map = run_map
            .get(run_name)
            .unwrap_or_else(|| panic!("run {:?} missing from commit store", run_name))
            .write();
        let cts: &mut commit::TimeSeries<V> = match tag_map.get_mut(tag_name) {
            Some(cts) => cts,
            None => tag_map
                .entry(tag_name.to_string())
                .or_insert(commit::TimeSeries::new(self.metadata.clone())),
        };
        self.rsv.commit_map(&mut cts.values, f)
    }

    /// Convert a `StageValue` representing a scalar to a `Datum<ScalarValue>`, which holds a data
    /// loss error if this is not possible.
    fn commit_scalar(sv: StageValue) -> commit::Datum<commit::ScalarValue> {
        use commit::{DataLoss, ScalarValue};
        use pb::summary::value::Value;
        let result: Result<ScalarValue, DataLoss> = match sv.payload {
            StagePayload::SummaryValue {
                value: Value::SimpleValue(f),
                ..
            } => Ok(ScalarValue(f64::from(f))),
            StagePayload::SummaryValue {
                value: Value::Tensor(tp),
                ..
            } => {
                let tp: pb::TensorProto = tp; // rust-analyzer has trouble here for some reason
                use pb::DataType;
                match DataType::from_i32(tp.dtype) {
                    Some(DataType::DtFloat) => {
                        if let Some(f) = tp.float_val.first() {
                            Ok(ScalarValue(f64::from(*f)))
                        } else if tp.tensor_content.len() >= 4 {
                            let f: f32 = LittleEndian::read_f32(&tp.tensor_content);
                            Ok(ScalarValue(f64::from(f)))
                        } else {
                            Err(DataLoss)
                        }
                    }
                    Some(DataType::DtDouble) => {
                        if let Some(f) = tp.double_val.first() {
                            Ok(ScalarValue(*f))
                        } else if tp.tensor_content.len() >= 8 {
                            let f: f64 = LittleEndian::read_f64(&tp.tensor_content);
                            Ok(ScalarValue(f))
                        } else {
                            Err(DataLoss)
                        }
                    }
                    _ => Err(DataLoss),
                }
            }
            _ => Err(DataLoss),
        };
        (sv.wall_time, result)
    }
}

impl StageValue {
    fn from_event(e: pb::Event) -> Vec<(Step, StageValue)> {
        let step = e.step;
        let wall_time = e.wall_time;
        match e.what {
            Some(pb::event::What::GraphDef(gd)) => vec![(
                step,
                StageValue {
                    wall_time,
                    tag: RUN_GRAPH_NAME.to_string(),
                    payload: StagePayload::GraphDef(gd),
                },
            )],
            Some(pb::event::What::Summary(sum)) => sum
                .value
                .into_iter()
                .filter_map(|v| {
                    let stage_value = (
                        step,
                        StageValue {
                            wall_time,
                            tag: v.tag,
                            payload: StagePayload::SummaryValue {
                                metadata: v.metadata,
                                value: v.value?,
                            },
                        },
                    );
                    Some(stage_value)
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

impl RunLoader {
    pub fn new() -> Self {
        Self {
            start_time: f64::INFINITY,
            last_committed_start_time: None,
            files: BTreeMap::new(),
            time_series: HashMap::new(),
        }
    }

    pub fn reload(&mut self, filenames: Vec<PathBuf>, run_name: &str, commit: &Commit) {
        self.update_file_set(filenames);
        self.reload_files(run_name, commit);
    }

    fn update_file_set(&mut self, filenames: Vec<PathBuf>) {
        // Store the new set of files so that we can remove discarded files later.
        let mut new_file_set: HashSet<PathBuf> = filenames.iter().cloned().collect();

        // Open readers for any new files.
        for filename in filenames {
            use std::collections::btree_map;
            match self.files.entry(filename) {
                btree_map::Entry::Occupied(_) => {}
                btree_map::Entry::Vacant(v) => {
                    let file = match File::open(v.key()) {
                        Ok(f) => f,
                        // TODO(@wchargin): Improve error handling.
                        Err(e) => {
                            eprintln!("failed to open event file {:?}: {:?}", v.key(), e);
                            continue;
                        }
                    };
                    let boxed_file: Box<dyn Read> = Box::new(BufReader::new(file));
                    let reader = EventFileReader::new(boxed_file);
                    v.insert(reader);
                }
            };
        }

        // Remove any discarded files.
        let to_remove: Vec<_> = self
            .files
            .keys()
            .filter_map(|p| {
                if new_file_set.remove(p) {
                    None
                } else {
                    Some(p.clone())
                }
            })
            .collect();
        for filename in to_remove {
            self.files.remove(&filename);
        }
    }

    fn reload_files(&mut self, run_name: &str, commit: &Commit) {
        for efl in self.files.values_mut() {
            loop {
                use crate::event_file::ReadEventError;
                use crate::tf_record::ReadRecordError;
                let event = match efl.next(&mut self.start_time) {
                    Ok(ev) => ev,
                    Err(ReadEventError::ReadRecordError(ReadRecordError::Truncated)) => {
                        break;
                    }
                    // TODO(@wchargin): Improve error handling.
                    Err(_) => {
                        eprintln!("error reading {:?}", run_name);
                        break;
                    }
                };
                for (step, mut sv) in StageValue::from_event(event).into_iter() {
                    let tag: String = std::mem::take(&mut sv.tag);
                    // Clone the tag so that we can pass it to `ts.commit` below. Could get
                    // cleverer (make `rsv.offer` return a reference to the newly stored item and
                    // maybe futz with cows), but this inefficiency is probably fine.
                    let cloned_tag_name = tag.clone();
                    let ts: &mut TimeSeries = self
                        .time_series
                        .entry(tag)
                        .or_insert_with(|| TimeSeries::new(sv.payload.take_metadata()));
                    if ts.data_class() != pb::DataClass::Unknown {
                        ts.rsv.offer(step, sv);
                    }
                    if ts.staged_weight() > COMMIT_WEIGHT || Instant::now() >= ts.next_commit {
                        Self::commit_start_time(
                            self.start_time,
                            &mut self.last_committed_start_time,
                            run_name,
                            commit,
                        );
                        ts.commit(run_name, &cloned_tag_name, commit)
                    }
                }
            }
        }

        // Commit all dirty reservoirs after reading to end of all files.
        // TODO(@wchargin): Consider batching these locks.
        Self::commit_start_time(
            self.start_time,
            &mut self.last_committed_start_time,
            run_name,
            commit,
        );
        for (tag_name, ts) in self.time_series.iter_mut() {
            if ts.rsv.staged_items().is_empty() {
                continue;
            }
            ts.commit(run_name, tag_name, commit);
        }
    }

    fn commit_start_time(
        start_time: f64,
        last_committed_start_time: &mut Option<f64>,
        run_name: &str,
        commit: &Commit,
    ) {
        if *last_committed_start_time == Some(start_time) {
            return;
        }
        {
            let mut start_times = commit.start_times.write();
            start_times.insert(run_name.to_string(), start_time);
        }
        *last_committed_start_time = Some(start_time);
    }
}
