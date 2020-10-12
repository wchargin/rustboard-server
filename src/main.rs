use std::path::PathBuf;

use rustboard_server::commit::{self, Commit};
use rustboard_server::logdir::LogdirLoader;

fn main() {
    let logdir = std::env::args_os()
        .nth(1)
        .expect("give logdir as first argument");
    let commit = Commit::new();
    let mut loader = LogdirLoader::new(&commit, PathBuf::from(logdir));
    eprintln!("initialized loader");
    loader.reload();

    fn describe_store<V>(name: &str, store: &commit::Store<V>) {
        for (run, tag_map) in &*store.read() {
            eprintln!("have {} for {:?}", name, run);
            for (tag, ts) in &*tag_map.read() {
                eprintln!(
                    "run {:?} tag {:?} (plugin={}) has {} points up to step {}",
                    run,
                    tag,
                    ts.metadata
                        .plugin_data
                        .as_ref()
                        .map(|pd| pd.plugin_name.as_str())
                        .unwrap_or("???"),
                    ts.values.len(),
                    ts.values.last().unwrap().0
                );
            }
        }
    }

    describe_store("scalars", &commit.scalars);
    describe_store("tensors", &commit.tensors);
    describe_store("blobs", &commit.blob_sequences);

    eprintln!("terminating");
}
