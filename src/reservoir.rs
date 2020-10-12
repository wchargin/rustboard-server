use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type Step = i64;

/// A [reservoir sampling] data structure, with support for preemption and deferred "commits" of
/// records to a separate destination for better concurrency. This structure always keeps the
/// latest record in the reservoir, and therefore must inspect every record in the stream.
///
/// # Preemption
///
/// All records stored in this reservoir have a *step* and a *payload*. The step, a non-negative
/// integer, is expected to be monotonically (strictly) increasing over time. Whenever a new record
/// arrives with step `s`, it preempts any records with steps not smaller than `s`.
///
/// This is motivated by preemptions of nondeterministic training jobs. If a job is checkpointed at
/// step 80 and preempted at step 100, any metrics written between steps 81 and 100 will be
/// recomputed when the job restarts. Since the outputs may be nondeterministic, the old values
/// must be discarded to avoid a misleading discontinuity between steps 100 and 101.
///
/// For the purpose of this reservoir, preemption is considered a normal mode of operation, not an
/// error. However, jobs incorrectly configured to emit non-increasing steps without being
/// preempted may find that this reservoir does not behave as they expect. For instance, a job that
/// emits many records all at step 0 will find that only one record is retained in the reservoir.
///
/// # Deferred commits
///
/// This reservoir is designed to maximize throughput of reading records from disk while still
/// providing a live-updating view to clients. To do so, we separate the data structures that the
/// reading worker must modify (the *stage*: i.e., this reservoir) from the clients' view (the
/// *commit*). The worker may own the stage exclusively, without any locks or synchronization. At
/// any time, the worker may take a write-lock of the commit and update it with the changes from
/// the stage. For instance, the worker might have a policy of committing "every 1000 records read,
/// or every 10 large records read, or every 5 seconds, whichever comes first".
///
/// The commit operation is quite fast: not only does it not do much work per record being
/// committed, it only commits records that have made it through the sampling process. Thus, if
/// 1000 records are read and 900 of them are discarded, deferring the commit will have saved 900
/// records' worth of wasted copies, compared to if the commit were always kept exactly up to date.
/// Assuming that the commit is shared under a [`std::sync::RwLock`] or similar, the critical
/// section in which the worker needs a write-lock should be short, and thus clients may normally
/// enjoy an uncontended view of the commit.
///
/// [reservoir sampling]: https://en.wikipedia.org/wiki/Reservoir_sampling
#[derive(Debug)]
pub struct StageReservoir<T, C = ChaCha20Rng> {
    /// Steps of items currently in the reservoir whose values have already been committed. Stored
    /// in step-sorted order, and all steps in `committed_steps` precede all steps in
    /// `staged_items`.
    committed_steps: Vec<Step>,
    /// Items currently in the reservoir but not yet committed. Stored in step-sorted order, and
    /// all steps in `staged_items` succeed all steps in `committed_steps`.
    staged_items: Vec<(Step, T)>,
    /// Total capacity of this reservoir. The combined physical capacities of `committed_steps` and
    /// `staged_items` may exceed this, but their combined lengths will not. Behavior is undefined
    /// if `capacity == 0`.
    capacity: usize,
    /// Whether there has been a preemption since the last commit.
    staged_preemption: bool,
    /// Reservoir control, to determine whether and whither a given new record should be included.
    ctl: C,
    /// Total number of records passed in the stream so far, regardless of whether they were ever
    /// added to the reservoir. Usually called `N` in the literature.
    seen: usize,
}

/// A `ReservoirControl` determines which records from a stream should be included into a
/// reservoir, and which records they should evict.
pub trait ReservoirControl {
    /// Upon seeing the latest record in a stream of `n` so far, roll for the index into the
    /// reservoir that should be evicted to make room for this record. If the result is greater
    /// than or equal to the capacity of the reservoir, the record will be skipped; otherwise, it
    /// will evict the record at the given index. The input `n` will be positive, and the result
    /// must be in `0..n`.
    ///
    /// A useful implementation is to simply roll a random integer in `0..n`. Test code may wish to
    /// provide an alternate, deterministic implementation.
    fn destination(&mut self, n: usize) -> usize;
}

impl<R: Rng> ReservoirControl for R {
    fn destination(&mut self, n: usize) -> usize {
        Uniform::from(0..n).sample(self)
    }
}

impl<T> StageReservoir<T, ChaCha20Rng> {
    /// Create a new reservoir with the specified capacity, using a fixed-seed random number
    /// generator for reservoir control. All reservoirs created by this function will use the same
    /// sequence of random numbers.
    ///
    /// This function does not allocate. Reservoir capacity is allocated as records are offered.
    pub fn new(capacity: usize) -> Self {
        Self::with_control(capacity, ChaCha20Rng::seed_from_u64(0))
    }
}

impl<T, C: ReservoirControl> StageReservoir<T, C> {
    /// Offer a record to the reservoir. The reservoir will always include the latest record. Other
    /// than the latest record, the records kept form a simple random sample of the stream (or at
    /// least approximately so in the case of preemptions).
    pub fn offer(&mut self, step: Step, v: T) {
        self.preempt(step);
        self.seen += 1;
        let dst = self.ctl.destination(self.seen);

        // Didn't make the cut? Keep-last only.
        if dst >= self.capacity {
            if self.staged_items.pop().is_none() {
                self.committed_steps.pop();
            }
            self.staged_items.push((step, v));
            return;
        }

        // Room to spare? Add to end.
        let num_stored = self.committed_steps.len() + self.staged_items.len();
        if num_stored < self.capacity {
            self.staged_items.push((step, v));
            return;
        }

        // Otherwise, we're evicting.
        if dst < self.committed_steps.len() {
            self.committed_steps.remove(dst);
        } else {
            self.staged_items.remove(dst - self.committed_steps.len());
        }
        self.staged_items.push((step, v))
    }
}

impl<T, C> StageReservoir<T, C> {
    /// Create a new reservoir with the specified capacity and reservoir control.
    ///
    /// This function does not allocate. Reservoir capacity is allocated as records are offered.
    pub fn with_control(capacity: usize, ctl: C) -> Self {
        Self {
            committed_steps: Vec::new(),
            staged_items: Vec::new(),
            capacity,
            staged_preemption: false,
            ctl,
            seen: 0,
        }
    }

    /// Access a view of the currently staged items.
    pub fn staged_items(&self) -> &Vec<(Step, T)> {
        &self.staged_items
    }

    /// Check whether a preemption has been staged.
    pub fn staged_preemption(&self) -> bool {
        self.staged_preemption
    }

    fn preempt(&mut self, step: Step) {
        let staged_nonpreempted = self
            .staged_items
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, (s, _))| if *s < step { Some(i + 1) } else { None })
            .unwrap_or(0);
        let staged_preempted = self.staged_items.len() - staged_nonpreempted;
        if staged_preempted > 0 {
            self.staged_items.truncate(staged_nonpreempted);
            self.staged_preemption = true;
        }
        let mut preempted = staged_preempted;
        if self.staged_items.is_empty() {
            // Committed steps may have been preempted as well.
            let committed_nonpreempted = self
                .committed_steps
                .iter()
                .enumerate()
                .rev()
                .find_map(|(i, s)| if *s < step { Some(i + 1) } else { None })
                .unwrap_or(0);
            let committed_preempted = self.committed_steps.len() - committed_nonpreempted;
            if committed_preempted > 0 {
                self.committed_steps.truncate(committed_nonpreempted);
                self.staged_preemption = true;
                preempted += committed_preempted;
            }
        }
        if preempted == 0 {
            return; // No need to adjust `seen`.
        }
        let old_stored = self.committed_steps.len() + self.staged_items.len() + preempted;
        let fac_preempted = (preempted as f64) / (old_stored as f64);
        self.seen = (self.seen as f64 * (1.0 - fac_preempted)).ceil() as usize;
    }

    /// Commit pending changes from this reservoir into a commit view. The commit should be a
    /// vector that starts empty and is modified only by calls to `commit`/`commit_map` on this
    /// reservoir value.
    pub fn commit(&mut self, head: &mut Vec<(Step, T)>) {
        self.commit_map(head, |t| t)
    }

    /// Commit pending changes from this reservoir into a commit view, applying a mapping function
    /// to each new value. This can be used to perform relatively expensive conversions or
    /// enrichments only for records that are actually committed. The commit should be a vector
    /// that starts empty and is modified only by calls to `commit`/`commit_map` on this reservoir
    /// value.
    pub fn commit_map<S, F: FnMut(T) -> S>(&mut self, head: &mut Vec<(Step, S)>, mut f: F) {
        let mut keep_steps = self.committed_steps.iter().peekable();
        head.retain(move |(s, _)| match keep_steps.peek() {
            Some(t) if *s == **t => {
                keep_steps.next();
                true
            }
            _ => false,
        });
        self.committed_steps
            .extend(self.staged_items.iter().map(|(step, _)| *step));
        head.extend(self.staged_items.drain(..).map(|(step, t)| (step, f(t))));
        self.staged_preemption = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Debug)]
    struct ScriptedControl(VecDeque<usize>);

    impl ScriptedControl {
        fn new() -> Self {
            ScriptedControl(VecDeque::new())
        }
    }

    impl ReservoirControl for ScriptedControl {
        fn destination(&mut self, n: usize) -> usize {
            let result = self.0.pop_front().expect("overran script");
            assert!(result < n, "bad script: {} >= {}", result, n);
            result
        }
    }

    impl Extend<usize> for ScriptedControl {
        fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
            self.0.extend(iter)
        }
    }

    #[test]
    fn test() {
        let mut rsv = StageReservoir::with_control(7, ScriptedControl::new());
        let mut head = Vec::new();
        fn mapper(s: &str) -> &str {
            // leak, for test convenience
            Box::leak(format!(":{}:", s).into_boxed_str())
        }

        rsv.ctl.extend(vec![0, 1, 1, 2]);
        rsv.offer(0, "zero");
        rsv.offer(1, "one");
        rsv.offer(2, "two");
        rsv.offer(3, "three???");
        rsv.commit_map(&mut head, mapper);
        assert_eq!(
            head,
            vec![(0, ":zero:"), (1, ":one:"), (2, ":two:"), (3, ":three???:")],
        );

        rsv.ctl.extend(vec![1, 2, 1, 3]);
        rsv.offer(4, "four???");
        rsv.offer(3, "three");
        rsv.offer(4, "four");
        rsv.offer(5, "five");
        rsv.commit_map(&mut head, mapper);
        assert_eq!(
            head,
            vec![
                (0, ":zero:"),
                (1, ":one:"),
                (2, ":two:"),
                (3, ":three:"),
                (4, ":four:"),
                (5, ":five:"),
            ],
        );

        rsv.ctl.extend(vec![4]);
        rsv.offer(6, "six");
        rsv.ctl.extend(vec![1]); // the first that matters: evict step 1
        rsv.offer(7, "seven");
        rsv.ctl.extend(vec![2, 3]); // these don't matter: preempt two, store two
        rsv.offer(6, "SIX");
        rsv.offer(7, "SEVEN");
        rsv.ctl.extend(vec![4]); // evict step 5 (at index 4, since 1 was evicted already)
        rsv.offer(8, "EIGHT");
        rsv.ctl.extend(vec![8, 9]); // drop steps 9 and 10, modulo keep-last
        rsv.offer(9, "NINE");
        rsv.offer(10, "TEN");
        rsv.commit_map(&mut head, mapper);
        assert_eq!(
            head,
            vec![
                (0, ":zero:"),
                (2, ":two:"),
                (3, ":three:"),
                (4, ":four:"),
                (6, ":SIX:"),
                (7, ":SEVEN:"),
                (10, ":TEN:"),
            ],
        );

        // Preempt again, then drop records even while there's space.
        rsv.ctl.extend(vec![0, 7, 7]);
        rsv.offer(5, "!five!");
        rsv.offer(6, "!six!");
        rsv.offer(7, "!seven!");
        rsv.commit_map(&mut head, mapper);
        assert_eq!(
            head,
            vec![
                (0, ":zero:"),
                (2, ":two:"),
                (3, ":three:"),
                (4, ":four:"),
                (7, ":!seven!:"),
            ],
        );
    }

    #[test]
    fn test_random() {
        // Seeded RNG, with tests for some invariants.
        let mut rsv = StageReservoir::new(10);
        let mut head = Vec::new();

        // Fill with `[i * i for i in range(1, 11)]`, exactly filling the reservoir.
        for i in 1..=10 {
            rsv.offer(i * i, ());
            if i % 5 == 0 {
                rsv.commit(&mut head);
                assert_eq!(head, (1..=i).map(|j| (j * j, ())).collect::<Vec<_>>());
            }
        }

        // Fill with more square numbers, keeping last but not overflowing.
        for i in 11..=16 {
            rsv.offer(i * i, ());
            rsv.commit(&mut head);
            assert_eq!(head.len(), 10);
            assert_eq!(head.last(), Some(&(i * i, ())));
        }

        // Seen 16 records, keeping 10. Preempt to invalidate records 9..=16, that the reservoir
        // must have between 2 and 8 old records before the new one is added.
        rsv.offer(70, ()); // 8 * 8 < 70 < 9 * 9
        rsv.commit(&mut head);
        assert!(
            (2..=9).contains(&head.len()),
            "want 2 <= {} <= 9: {:?}",
            head.len(),
            head
        );
        assert!(
            head.iter().all(|(s, _)| *s <= 70),
            "want all <= 70: {:?}",
            head
        );
        assert_eq!(head.last(), Some(&(70, ())));

        // One more sanity check: add another record. The "70" preemption may or may not be
        // evicted, but this new record should be the last.
        rsv.offer(71, ());
        rsv.commit(&mut head);
        assert_eq!(head.last(), Some(&(71, ())));
    }

    #[test]
    fn test_deterministic_and_commit_independent() {
        let mut r1 = StageReservoir::new(10);
        let mut r2 = StageReservoir::new(10);
        let mut h1 = Vec::new();
        let mut h2 = Vec::new();
        for i in 0..100 {
            r1.offer(i, ());
            r2.offer(i, ());
            match i % 10 {
                2 => r1.commit(&mut h1),
                7 => r2.commit(&mut h2),
                9 => {
                    r1.commit(&mut h1);
                    r2.commit(&mut h2);
                    assert_eq!(h1, h2);
                }
                _ => (),
            }
        }
    }
}
