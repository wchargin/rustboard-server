use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use crossbeam::queue::SegQueue;
use walkdir::WalkDir;

use crate::commit::{self, Commit};
use crate::run::RunLoader;

const EVENT_FILE_BASENAME_INFIX: &str = "tfevents";

const LOADER_THREAD_COUNT: usize = 8;

pub struct LogdirLoader<'a> {
    commit: &'a Commit,
    root: PathBuf,
    runs: HashMap<String, Run>,
}

struct Run {
    relpath: PathBuf,
    loader: RunLoader,
    merged_relpaths: HashSet<PathBuf>,
}

struct EventFileDiscovery {
    run_relpath: PathBuf,
    event_file: PathBuf,
}

impl<'a> LogdirLoader<'a> {
    pub fn new(commit: &'a Commit, root: PathBuf) -> Self {
        Self {
            commit,
            root,
            runs: HashMap::new(),
        }
    }

    pub fn reload(&mut self) {
        let mut discoveries = self.discover_runs();

        let discovered_runs: HashSet<String> = discoveries.keys().cloned().collect();
        let to_remove = self
            .runs
            .keys()
            .filter_map(|k| {
                if discovered_runs.contains(k) {
                    None
                } else {
                    Some(k.to_string())
                }
            })
            .collect::<Vec<_>>();
        for run_name in &to_remove {
            self.runs.remove(run_name);
        }
        if !to_remove.is_empty() {
            {
                let mut start_times = self.commit.start_times.write();
                for run_name in &to_remove {
                    start_times.remove(run_name);
                }
            }
            fn remove_from<V>(store_lock: &commit::Store<V>, to_remove: &[String]) {
                let mut store = store_lock.write();
                for run_name in to_remove {
                    store.remove(run_name);
                }
            }
            remove_from(&self.commit.scalars, &to_remove);
            remove_from(&self.commit.tensors, &to_remove);
            remove_from(&self.commit.blob_sequences, &to_remove);
        }

        let to_add = discoveries
            .keys()
            .filter(|k| !self.runs.contains_key(*k))
            .collect::<Vec<_>>();
        if !to_add.is_empty() {
            // Don't add start times; runs will do that upon loading. Do populate stores.
            fn add_to<V>(store_lock: &commit::Store<V>, to_add: &[&String]) {
                let mut store = store_lock.write();
                for run_name in to_add {
                    store.entry((*run_name).clone()).or_default();
                }
            }
            add_to(&self.commit.scalars, &to_add);
            add_to(&self.commit.tensors, &to_add);
            add_to(&self.commit.blob_sequences, &to_add);
        }

        struct WorkUnit<'a> {
            loader: &'a mut RunLoader,
            filenames: Vec<PathBuf>,
            run_name: &'a String,
        };

        let work_units = SegQueue::new();
        for (run_name, event_files) in discoveries.iter_mut() {
            let run = self
                .runs
                .entry(run_name.to_string())
                .or_insert_with(|| Run {
                    // `event_files` non-empty by construction, so we can take the first relpath
                    relpath: event_files[0].run_relpath.clone(),
                    loader: RunLoader::new(),
                    merged_relpaths: HashSet::new(),
                });
            for ef in &*event_files {
                if run.merged_relpaths.insert(ef.run_relpath.clone())
                    && ef.run_relpath != run.relpath
                {
                    eprintln!(
                        "merging directories {:?} and {:?}, which both normalize to run {:?}",
                        run.relpath, ef.run_relpath, run_name
                    );
                }
            }
        }
        for (run_name, run) in self.runs.iter_mut() {
            let event_files = discoveries.get_mut(run_name).unwrap();
            let filenames = std::mem::take(event_files)
                .into_iter()
                .map(|d| d.event_file)
                .collect::<Vec<_>>();
            eprintln!("enqueuing run {:?}...", run_name);
            work_units.push(WorkUnit {
                loader: &mut run.loader,
                filenames,
                run_name,
            });
        }

        let commit = self.commit;
        let q = &work_units;
        let scope_result = crossbeam::scope(move |scope| {
            let handles = (0..LOADER_THREAD_COUNT)
                .map(|i| {
                    let builder = scope.builder().name(format!("Reloader-{:03}", i));
                    let handle = builder.spawn(move |_| {
                        while let Some(wu) = q.pop() {
                            eprintln!("loading run {:?}...", wu.run_name);
                            wu.loader.reload(wu.filenames, wu.run_name, commit);
                            eprintln!("loaded run {:?}...", wu.run_name);
                        }
                    });
                    handle.expect("failed to spawn reloader thread")
                })
                .collect::<Vec<_>>();
            drop(handles)
        });
        if let Err(e) = scope_result {
            eprintln!("error in crossbeam scope: {:?}", e);
        }
        eprintln!("finished load cycle");
    }

    fn discover_runs(&self) -> HashMap<String, Vec<EventFileDiscovery>> {
        let mut run_map = HashMap::<String, Vec<_>>::new();
        for dirent in WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|r| match r {
                Ok(dirent) => Some(dirent),
                Err(e) => {
                    eprintln!("error walking log directory: {}", e);
                    None
                }
            })
        {
            if !dirent.file_type().is_file() {
                continue;
            }
            if !dirent
                .file_name()
                .to_string_lossy()
                .contains(EVENT_FILE_BASENAME_INFIX)
            {
                continue;
            }
            let run_dir = match dirent.path().parent() {
                None => {
                    eprintln!(
                        "path {} is a file but has no parent",
                        dirent.path().display()
                    );
                    continue;
                }
                Some(parent) => parent,
            };
            let mut run_relpath = match run_dir.strip_prefix(&self.root) {
                Err(_) => {
                    eprintln!(
                        "log directory {} is not a prefix of run directory {}",
                        &self.root.display(),
                        &run_dir.display(),
                    );
                    continue;
                }
                Ok(rp) => rp.to_path_buf(),
            };
            if run_relpath == Path::new("") {
                run_relpath.push(".");
            }
            let run_name = format!("{}", run_relpath.display());
            run_map
                .entry(run_name)
                .or_default()
                .push(EventFileDiscovery {
                    run_relpath,
                    event_file: dirent.into_path(),
                });
        }
        run_map
    }
}
