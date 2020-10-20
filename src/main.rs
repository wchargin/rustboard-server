use async_stream::try_stream;
use futures_core::Stream;
use rustboard_server::logdir::LogdirLoader;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::default::Default;
use std::hash::Hash;
use std::path::PathBuf;
use std::pin::Pin;
use std::thread;
use std::time::Duration;
use tonic::{transport::Server, Request, Response, Status};

use rustboard_server::blob_key::BlobKey;
use rustboard_server::commit::Commit;
use rustboard_server::proto::tensorboard::data;

use data::tensor_board_data_provider_server as tbdps;
use tbdps::{TensorBoardDataProvider, TensorBoardDataProviderServer};

const BLOB_BATCH_SIZE: usize = 1024 * 1024 * 8;
const RELOAD_INTERVAL: Duration = Duration::from_secs(5);

struct DataProvider<'a> {
    head: &'a Commit,
}

#[tonic::async_trait]
// TODO(@wchargin): Figure out if we can make this broader than 'static.
impl TensorBoardDataProvider for DataProvider<'static> {
    async fn list_runs(
        &self,
        _request: Request<data::ListRunsRequest>,
    ) -> Result<Response<data::ListRunsResponse>, Status> {
        let mut res = data::ListRunsResponse::default();
        let start_times = self.head.start_times.read();
        for (run_name, start_time) in &*start_times {
            res.runs.push(data::Run {
                id: run_name.clone(),
                name: run_name.clone(),
                start_time: *start_time,
            });
        }
        Ok(Response::new(res))
    }

    async fn list_scalars(
        &self,
        req: Request<data::ListScalarsRequest>,
    ) -> Result<Response<data::ListScalarsResponse>, Status> {
        let mut req = req.into_inner();
        let mut res: data::ListScalarsResponse = Default::default();
        let (run_filter, tag_filter) = parse_rtf(req.run_tag_filter.take());
        let want_plugin = req
            .plugin_filter
            .map(|pf| pf.plugin_name)
            .unwrap_or_default();
        let store = self.head.scalars.read();
        for (run, tag_map) in &*store {
            if !run_filter.want(run) {
                continue;
            }
            let tags = tag_map.read();
            let mut run_res = data::list_scalars_response::RunEntry::default();
            for (tag, ts) in &*tags {
                if !tag_filter.want(tag) {
                    continue;
                }
                if ts
                    .metadata
                    .plugin_data
                    .as_ref()
                    .map(|pd| pd.plugin_name.as_str())
                    != Some(want_plugin.as_str())
                {
                    continue;
                }
                let mut tag_res = data::list_scalars_response::TagEntry::default();
                tag_res.tag_name = tag.clone();
                tag_res.time_series = Some(data::ScalarTimeSeries {
                    max_step: ts.values.last().map(|(step, _)| *step).unwrap_or(0),
                    max_wall_time: ts
                        .values
                        .iter()
                        .map(|(_, (wt, _))| *wt)
                        .max_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap_or(-f64::INFINITY),
                    summary_metadata: Some(ts.metadata.clone()),
                });
                run_res.tags.push(tag_res);
            }
            if !run_res.tags.is_empty() {
                run_res.run_name = run.clone();
                res.runs.push(run_res);
            }
        }
        Ok(Response::new(res))
    }

    async fn read_scalars(
        &self,
        req: Request<data::ReadScalarsRequest>,
    ) -> Result<Response<data::ReadScalarsResponse>, Status> {
        let mut req = req.into_inner();
        let mut res: data::ReadScalarsResponse = Default::default();
        let (run_filter, tag_filter) = parse_rtf(req.run_tag_filter.take());
        let want_plugin = req
            .plugin_filter
            .map(|pf| pf.plugin_name)
            .unwrap_or_default();
        let store = self.head.scalars.read();
        for (run, tag_map) in &*store {
            if !run_filter.want(run) {
                continue;
            }
            let tags = tag_map.read();
            let mut run_res = data::read_scalars_response::RunEntry::default();
            for (tag, ts) in &*tags {
                if !tag_filter.want(tag) {
                    continue;
                }
                if ts
                    .metadata
                    .plugin_data
                    .as_ref()
                    .map(|pd| pd.plugin_name.as_str())
                    != Some(want_plugin.as_str())
                {
                    continue;
                }
                let mut tag_res: data::read_scalars_response::TagEntry = Default::default();
                tag_res.tag_name = tag.clone();
                let mut data: data::ScalarData = Default::default();
                for (step, (wall_time, maybe_value)) in &ts.values {
                    match maybe_value {
                        Ok(value) => {
                            data.step.push(*step);
                            data.wall_time.push(*wall_time);
                            data.value.push(value.0);
                        }
                        Err(_) => {
                            eprintln!("dropping corrupt datum at step {}", *step);
                        }
                    };
                }
                tag_res.data = Some(data);
                run_res.tags.push(tag_res);
            }
            if !run_res.tags.is_empty() {
                run_res.run_name = run.clone();
                res.runs.push(run_res);
            }
        }
        Ok(Response::new(res))
    }

    async fn list_tensors(
        &self,
        req: Request<data::ListTensorsRequest>,
    ) -> Result<Response<data::ListTensorsResponse>, Status> {
        let mut req = req.into_inner();
        let mut res: data::ListTensorsResponse = Default::default();
        let (run_filter, tag_filter) = parse_rtf(req.run_tag_filter.take());
        let want_plugin = req
            .plugin_filter
            .map(|pf| pf.plugin_name)
            .unwrap_or_default();
        let store = self.head.tensors.read();
        for (run, tag_map) in &*store {
            if !run_filter.want(run) {
                continue;
            }
            let tags = tag_map.read();
            let mut run_res = data::list_tensors_response::RunEntry::default();
            for (tag, ts) in &*tags {
                if !tag_filter.want(tag) {
                    continue;
                }
                if ts
                    .metadata
                    .plugin_data
                    .as_ref()
                    .map(|pd| pd.plugin_name.as_str())
                    != Some(want_plugin.as_str())
                {
                    continue;
                }
                let mut tag_res = data::list_tensors_response::TagEntry::default();
                tag_res.tag_name = tag.clone();
                tag_res.time_series = Some(data::TensorTimeSeries {
                    max_step: ts.values.last().map(|(step, _)| *step).unwrap_or(0),
                    max_wall_time: ts
                        .values
                        .iter()
                        .map(|(_, (wt, _))| *wt)
                        .max_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap_or(-f64::INFINITY),
                    summary_metadata: Some(ts.metadata.clone()),
                });
                run_res.tags.push(tag_res);
            }
            if !run_res.tags.is_empty() {
                run_res.run_name = run.clone();
                res.runs.push(run_res);
            }
        }
        Ok(Response::new(res))
    }

    async fn read_tensors(
        &self,
        req: Request<data::ReadTensorsRequest>,
    ) -> Result<Response<data::ReadTensorsResponse>, Status> {
        let mut req = req.into_inner();
        let mut res: data::ReadTensorsResponse = Default::default();
        let (run_filter, tag_filter) = parse_rtf(req.run_tag_filter.take());
        let want_plugin = req
            .plugin_filter
            .map(|pf| pf.plugin_name)
            .unwrap_or_default();
        let store = self.head.tensors.read();
        for (run, tag_map) in &*store {
            if !run_filter.want(run) {
                continue;
            }
            let tags = tag_map.read();
            let mut run_res = data::read_tensors_response::RunEntry::default();
            for (tag, ts) in &*tags {
                if !tag_filter.want(tag) {
                    continue;
                }
                if ts
                    .metadata
                    .plugin_data
                    .as_ref()
                    .map(|pd| pd.plugin_name.as_str())
                    != Some(want_plugin.as_str())
                {
                    continue;
                }
                let mut tag_res: data::read_tensors_response::TagEntry = Default::default();
                tag_res.tag_name = tag.clone();
                let mut data: data::TensorData = Default::default();
                for (step, (wall_time, maybe_value)) in &ts.values {
                    match maybe_value {
                        Ok(value) => {
                            data.step.push(*step);
                            data.wall_time.push(*wall_time);
                            data.value.push(value.0.clone());
                        }
                        Err(_) => {
                            eprintln!("dropping corrupt datum at step {}", *step);
                        }
                    };
                }
                tag_res.data = Some(data);
                run_res.tags.push(tag_res);
            }
            if !run_res.tags.is_empty() {
                run_res.run_name = run.clone();
                res.runs.push(run_res);
            }
        }
        Ok(Response::new(res))
    }

    async fn list_blob_sequences(
        &self,
        req: Request<data::ListBlobSequencesRequest>,
    ) -> Result<Response<data::ListBlobSequencesResponse>, Status> {
        let mut req = req.into_inner();
        let mut res: data::ListBlobSequencesResponse = Default::default();
        let (run_filter, tag_filter) = parse_rtf(req.run_tag_filter.take());
        let want_plugin = req
            .plugin_filter
            .map(|pf| pf.plugin_name)
            .unwrap_or_default();
        let store = self.head.blob_sequences.read();
        for (run, tag_map) in &*store {
            if !run_filter.want(run) {
                continue;
            }
            let tags = tag_map.read();
            let mut run_res = data::list_blob_sequences_response::RunEntry::default();
            for (tag, ts) in &*tags {
                if !tag_filter.want(tag) {
                    continue;
                }
                if ts
                    .metadata
                    .plugin_data
                    .as_ref()
                    .map(|pd| pd.plugin_name.as_str())
                    != Some(want_plugin.as_str())
                {
                    continue;
                }
                let mut tag_res = data::list_blob_sequences_response::TagEntry::default();
                tag_res.tag_name = tag.clone();
                tag_res.time_series = Some(data::BlobSequenceTimeSeries {
                    max_step: ts.values.last().map(|(step, _)| *step).unwrap_or(0),
                    max_wall_time: ts
                        .values
                        .iter()
                        .map(|(_, (wt, _))| *wt)
                        .max_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap_or(-f64::INFINITY),
                    max_length: ts
                        .values
                        .iter()
                        .filter_map(|(_, (_, blobs))| {
                            blobs.as_ref().ok().map(|bsv| bsv.0.len() as i64)
                        })
                        .max()
                        .unwrap_or(-1),
                    summary_metadata: Some(ts.metadata.clone()),
                });
                run_res.tags.push(tag_res);
            }
            if !run_res.tags.is_empty() {
                run_res.run_name = run.clone();
                res.runs.push(run_res);
            }
        }
        Ok(Response::new(res))
    }

    async fn read_blob_sequences(
        &self,
        req: Request<data::ReadBlobSequencesRequest>,
    ) -> Result<Response<data::ReadBlobSequencesResponse>, Status> {
        let mut req = req.into_inner();
        let mut res: data::ReadBlobSequencesResponse = Default::default();
        let (run_filter, tag_filter) = parse_rtf(req.run_tag_filter.take());
        let want_plugin = req
            .plugin_filter
            .map(|pf| pf.plugin_name)
            .unwrap_or_default();
        let store = self.head.blob_sequences.read();
        for (run, tag_map) in &*store {
            if !run_filter.want(run) {
                continue;
            }
            let tags = tag_map.read();
            let mut run_res = data::read_blob_sequences_response::RunEntry::default();
            for (tag, ts) in &*tags {
                if !tag_filter.want(tag) {
                    continue;
                }
                if ts
                    .metadata
                    .plugin_data
                    .as_ref()
                    .map(|pd| pd.plugin_name.as_str())
                    != Some(want_plugin.as_str())
                {
                    continue;
                }
                let mut tag_res: data::read_blob_sequences_response::TagEntry = Default::default();
                tag_res.tag_name = tag.clone();
                let mut data: data::BlobSequenceData = Default::default();
                for (step, (wall_time, maybe_value)) in &ts.values {
                    match maybe_value {
                        Ok(value) => {
                            data.step.push(*step);
                            data.wall_time.push(*wall_time);
                            let experiment_id = req.experiment_id.as_ref();
                            let sample_count = value.0.len();
                            use std::borrow::Cow;
                            let blob_refs = (0..sample_count)
                                .map(|index| {
                                    let bk = BlobKey {
                                        experiment_id: Cow::Borrowed(experiment_id),
                                        run: Cow::Borrowed(run.as_ref()),
                                        tag: Cow::Borrowed(tag.as_ref()),
                                        step: *step,
                                        index,
                                    };
                                    data::BlobReference {
                                        blob_key: bk.to_string(),
                                        url: String::new(),
                                    }
                                })
                                .collect();
                            data.values.push(data::BlobReferenceSequence { blob_refs });
                        }
                        Err(_) => {
                            eprintln!("dropping corrupt datum at step {}", *step);
                        }
                    };
                }
                tag_res.data = Some(data);
                run_res.tags.push(tag_res);
            }
            if !run_res.tags.is_empty() {
                run_res.run_name = run.clone();
                res.runs.push(run_res);
            }
        }
        Ok(Response::new(res))
    }

    type ReadBlobStream =
        Pin<Box<dyn Stream<Item = Result<data::ReadBlobResponse, Status>> + Send + Sync + 'static>>;

    async fn read_blob(
        &self,
        req: Request<data::ReadBlobRequest>,
    ) -> Result<Response<Self::ReadBlobStream>, Status> {
        let req = req.into_inner();
        let bk: BlobKey = match req.blob_key.parse() {
            Err(e) => {
                return Err(Status::invalid_argument(format!(
                    "failed to parse blob key: {:?}",
                    e,
                )))
            }
            Ok(bk) => bk,
        };

        let store = self.head.blob_sequences.read();
        let tag_map = store
            .get(bk.run.as_ref())
            .ok_or_else(|| Status::not_found(format!("no such run: {:?}", bk.run)))?
            .read();
        let ts = tag_map.get(bk.tag.as_ref()).ok_or_else(|| {
            Status::not_found(format!("run {:?} has no such tag: {:?}", bk.run, bk.tag))
        })?;
        let datum = ts
            .values
            .iter()
            .find_map(
                |(step, (_, value))| {
                    if *step == bk.step {
                        Some(value)
                    } else {
                        None
                    }
                },
            )
            .ok_or_else(|| {
                Status::not_found(format!(
                    "run {:?}, tag {:?} has no step {}; may have been evicted",
                    bk.run, bk.tag, bk.step
                ))
            })?
            .as_ref()
            .map_err(|_| {
                Status::data_loss(format!(
                    "blob sequence for run {:?}, tag {:?}, step {} has invalid data",
                    bk.run, bk.tag, bk.step
                ))
            })?;
        let blob = datum.0.get(bk.index).ok_or_else(|| {
            Status::not_found(format!(
                "blob sequence at run {:?}, tag {:?}, step {:?} has no index {} (length: {})",
                bk.run,
                bk.tag,
                bk.step,
                bk.index,
                datum.0.len()
            ))
        })?;
        let blob = blob.clone();
        drop(tag_map);
        drop(store);

        let stream = try_stream! {
            for chunk in blob.chunks(BLOB_BATCH_SIZE) {
                yield data::ReadBlobResponse {data: chunk.to_vec()};
            }
        };

        Ok(Response::new(Box::pin(stream) as Self::ReadBlobStream))
    }
}

fn parse_rtf(rtf: Option<data::RunTagFilter>) -> (StringFilter, StringFilter) {
    let rtf = rtf.unwrap_or_default();
    let run_filter = match rtf.runs {
        None => StringFilter::All,
        Some(data::RunFilter { runs }) => StringFilter::Just(runs.into_iter().collect()),
    };
    let tag_filter = match rtf.tags {
        None => StringFilter::All,
        Some(data::TagFilter { tags }) => StringFilter::Just(tags.into_iter().collect()),
    };
    (run_filter, tag_filter)
}

enum StringFilter {
    All,
    Just(HashSet<String>),
}

impl StringFilter {
    pub fn want<Q: ?Sized>(&self, value: &Q) -> bool
    where
        String: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self {
            StringFilter::All => true,
            StringFilter::Just(some) => some.contains(value),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logdir = std::env::args_os()
        .nth(1)
        .expect("give logdir as first argument");

    // leak commit because gRPC server requires static lifetime
    let commit: &'static Commit = Box::leak(Box::new(Commit::new()));

    thread::Builder::new()
        .name("Reloader".into())
        .spawn(move || {
            let mut loader = LogdirLoader::new(commit, PathBuf::from(logdir));
            loop {
                loader.reload();
                thread::sleep(RELOAD_INTERVAL);
            }
        })?;

    let addr = "[::1]:6206".parse().unwrap();
    let dp = DataProvider { head: commit };
    let svc = TensorBoardDataProviderServer::new(dp);
    Server::builder().add_service(svc).serve(addr).await?;
    Ok(())
}
