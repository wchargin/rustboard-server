use futures_core::Stream;
use rustboard_server::logdir::LogdirLoader;
use std::default::Default;
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
        _request: Request<data::ListScalarsRequest>,
    ) -> Result<Response<data::ListScalarsResponse>, Status> {
        todo!()
    }

    async fn read_scalars(
        &self,
        _request: Request<data::ReadScalarsRequest>,
    ) -> Result<Response<data::ReadScalarsResponse>, Status> {
        todo!()
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
