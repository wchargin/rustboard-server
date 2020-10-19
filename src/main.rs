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

use rustboard_server::commit::Commit;
use rustboard_server::proto::tensorboard::data;

use data::tensor_board_data_provider_server as tbdps;
use tbdps::{TensorBoardDataProvider, TensorBoardDataProviderServer};

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
        _request: Request<data::ListBlobSequencesRequest>,
    ) -> Result<Response<data::ListBlobSequencesResponse>, Status> {
        todo!()
    }

    async fn read_blob_sequences(
        &self,
        _request: Request<data::ReadBlobSequencesRequest>,
    ) -> Result<Response<data::ReadBlobSequencesResponse>, Status> {
        todo!()
    }

    type ReadBlobStream =
        Pin<Box<dyn Stream<Item = Result<data::ReadBlobResponse, Status>> + Send + Sync + 'static>>;

    async fn read_blob(
        &self,
        _request: Request<data::ReadBlobRequest>,
    ) -> Result<Response<Self::ReadBlobStream>, Status> {
        todo!()
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
