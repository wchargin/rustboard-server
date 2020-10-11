pub mod tensorboard {
    tonic::include_proto!("tensorboard");
    pub mod data {
        tonic::include_proto!("tensorboard.data");
    }
}
