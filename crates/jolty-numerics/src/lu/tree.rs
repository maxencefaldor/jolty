//! Multifrontal factorization and triangular solves over a dissected diagonal block (Duff and
//! Reid, "The multifrontal solution of indefinite sparse symmetric linear equations", ACM
//! Transactions on Mathematical Software 9(3), 1983).
//!
//! Each interior factors its local columns by left-looking elimination, choosing pivots from
//! its local rows and producing a dense Schur complement update on its border. Each separator
//! assembles a dense frontal matrix from its local entries and its children's updates in order.
//! The separator eliminates its local columns and any delayed child columns, producing an update
//! for its parent. A column whose best local pivot falls below threshold is delayed with an
//! unpivoted local row to the parent front so that factors remain exact; the root pivots on any
//! remaining row.
//!
//! Child subtrees share no matrix entries and factor concurrently as rayon tasks when two or
//! more threads are requested. Frontal columns are distributed across a subtree's thread share
//! proportional to its position count. Every floating-point value undergoes the exact same
//! operation sequence regardless of scheduling, yielding bit-identical factors and solutions
//! across thread counts.

use rayon::prelude::*;

use super::dissection::{Dissection, Node};
use super::numeric::{ColumnFactors, PIVOT_TOLERANCE, Triangle};
use super::{LinearError, NONE, Scalar};

/// Multifrontal factors for a dissected diagonal block, stored in dissection node order.
#[derive(Clone)]
pub(super) struct Tree<T> {
    parts: Vec<Part<T>>,
}

/// Factors for a single node of a dissection tree.
#[derive(Clone)]
enum Part<T> {
    Interior(Box<Interior<T>>),
    Front(Box<Front<T>>),
}

/// Borrowed view of the dense column-major Schur complement update passed from a node to its
/// parent, ordered with delayed rows and columns before border entries.
struct Update<'a, T> {
    /// Number of delayed rows and delayed columns.
    delayed: usize,
    values: &'a [T],
    /// Starting offset in `values` of entry `(0, 0)`.
    first: usize,
    /// Stride in `values` between consecutive columns.
    leading: usize,
}

impl<T: Scalar> Update<'_, T> {
    /// Returns the update entry at `(row, column)`.
    fn get(&self, row: usize, column: usize) -> T {
        self.values[self.first + row + column * self.leading]
    }
}

impl<T: Scalar> Tree<T> {
    /// Allocates an unfactored multifrontal tree matching `dissection`.
    pub(super) fn new(dissection: &Dissection) -> Self {
        let parts = dissection
            .nodes
            .iter()
            .map(|node| {
                if node.is_interior() {
                    Part::Interior(Box::default())
                } else {
                    Part::Front(Box::default())
                }
            })
            .collect();
        Self { parts }
    }

    /// Invalidates every cached pivot so the next factorization searches afresh.
    pub(super) fn forget(&mut self) {
        for part in &mut self.parts {
            match part {
                Part::Interior(interior) => interior.complete = false,
                Part::Front(front) => front.complete = false,
            }
        }
    }

    /// Returns `true` when every node retains valid pivots for refactorization.
    pub(super) fn is_complete(&self) -> bool {
        self.parts.iter().all(|part| match part {
            Part::Interior(interior) => interior.complete,
            Part::Front(front) => front.complete,
        })
    }

    /// Factors the dissected block from matrix `values` indexed by slot across up to `threads`
    /// threads (or sequentially on the calling thread when `threads <= 1`).
    ///
    /// Each node reuses cached pivots while every pivot meets the threshold and all children
    /// reused theirs; otherwise the node searches afresh. Returns `true` if any node searched.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] naming the first column with no nonzero pivot, or
    /// [`LinearError::TooLarge`] when factor buffers exceed available memory.
    pub(super) fn factor(
        &mut self,
        dissection: &Dissection,
        values: &[T],
        threads: usize,
    ) -> Result<bool, LinearError> {
        let nodes = &dissection.nodes;
        let root = dissection.root();
        upward(
            nodes,
            (root, Visits::on(threads)),
            &mut self.parts,
            &|(index, threads), part, below, searched| {
                let node = &nodes[index];
                let below = (below, node.earliest);
                let searched = searched
                    || !match part {
                        Part::Interior(interior) => {
                            interior.complete && interior.refactor(node, values)
                        }
                        Part::Front(front) => {
                            front.complete && front.refactor(node, values, below, threads)?
                        }
                    };
                if searched {
                    match part {
                        Part::Interior(interior) => interior.search(node, values),
                        Part::Front(front) => {
                            front.search(node, values, below, (threads, index == root))
                        }
                    }?;
                }
                Ok(searched)
            },
        )
    }

    /// Returns `true` when subsequent factorizations reuse the exact same pivots as `other`,
    /// matching [`super::numeric::Numeric::pivots_alike`].
    pub(super) fn pivots_alike(&self, other: &Self) -> bool {
        self.parts.iter().zip(&other.parts).all(|pair| match pair {
            (Part::Interior(one), Part::Interior(another)) => one.pivots_alike(another),
            (Part::Front(one), Part::Front(another)) => {
                one.complete == another.complete
                    && (!one.complete || one.operations == another.operations)
            }
            _ => false,
        })
    }

    /// Returns the total number of columns delayed from child nodes to parents in the last
    /// factorization.
    pub(super) fn delays(&self) -> usize {
        self.parts
            .iter()
            .map(|part| match part {
                Part::Interior(interior) => interior.delayed.len(),
                Part::Front(front) => front.fully - front.steps,
            })
            .sum()
    }

    /// Returns the total number of stored entries across all node factors.
    pub(super) fn entries(&self) -> usize {
        self.parts
            .iter()
            .map(|part| match part {
                Part::Interior(interior) => interior.factors.entries() + interior.outer.entries(),
                Part::Front(front) => {
                    let size = front.rows.len();
                    front.steps * (2 * size - front.steps)
                }
            })
            .sum()
    }

    /// Solves the block system (or its transpose when `transposed` is `true`) using `scratch`
    /// across up to `threads` threads.
    ///
    /// Reads the ordered right-hand side from `given` and writes the block solution into
    /// `solution` (indexed by ordered column, or by ordered row when transposed).
    pub(super) fn solve(
        &self,
        dissection: &Dissection,
        (given, solution): (&[T], &mut [T]),
        scratch: &mut Vec<Scratch<T>>,
        (transposed, threads): (bool, usize),
    ) {
        let nodes = &dissection.nodes;
        scratch.resize_with(nodes.len(), Scratch::default);
        let root = dissection.root();
        let Ok(_) = upward(
            nodes,
            (root, Visits::on(threads)),
            scratch,
            &|(index, _), space, below, _| {
                let node = &nodes[index];
                let below = (self.parts.as_slice(), below, node.earliest);
                match &self.parts[index] {
                    Part::Interior(interior) => interior.forward(node, given, space, transposed),
                    Part::Front(front) => front.forward(node, given, space, below, transposed),
                }
                Ok::<bool, std::convert::Infallible>(false)
            },
        );
        downward(
            nodes,
            (root, Visits::on(threads)),
            scratch,
            solution,
            &|index, space, below, solution| {
                let node = &nodes[index];
                let parts = self.parts.as_slice();
                match &parts[index] {
                    Part::Interior(interior) => {
                        interior.backward(node, space, solution, transposed);
                    }
                    Part::Front(front) => {
                        front.backward(node, space, (parts, below), solution, transposed);
                    }
                }
            },
        );
    }
}

impl<T: Scalar> Part<T> {
    /// Returns the ordered positions of the rows delayed by the node.
    fn delayed_rows(&self) -> &[usize] {
        match self {
            Self::Interior(interior) => &interior.delayed_rows,
            Self::Front(front) => &front.rows[front.steps..front.fully],
        }
    }

    /// Returns the ordered positions of the columns delayed by the node.
    fn delayed_columns(&self) -> &[usize] {
        match self {
            Self::Interior(interior) => &interior.delayed_columns,
            Self::Front(front) => &front.columns[front.steps..front.fully],
        }
    }

    /// Returns a view of the Schur complement update for the parent front.
    fn update(&self) -> Update<'_, T> {
        match self {
            Self::Interior(interior) => {
                let size = interior.update_rows.len();
                Update {
                    delayed: interior.delayed.len(),
                    values: &interior.update,
                    first: 0,
                    leading: size,
                }
            }
            Self::Front(front) => {
                let leading = front.rows.len();
                Update {
                    delayed: front.fully - front.steps,
                    values: &front.frame,
                    first: front.steps * (leading + 1),
                    leading,
                }
            }
        }
    }
}

/// Per-node scratch buffers reused across multifrontal solves.
#[derive(Debug, Clone)]
pub(super) struct Scratch<T> {
    /// Node frontal vector: one entry per row (or per column when transposed).
    frame: Vec<T>,
    /// Eliminated step values (or frontal column entries during backward substitution).
    steps: Vec<T>,
    /// Forward-pass contribution passed upward to the parent.
    up: Vec<T>,
    /// Backward-pass solution values received from the parent.
    down: Vec<T>,
}

impl<T> Default for Scratch<T> {
    fn default() -> Self {
        Self {
            frame: Vec::new(),
            steps: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
        }
    }
}

/// Splits consecutive `children` into two balanced groups for parallel execution across
/// `threads` threads.
///
/// Returns `(split_index, subtree_nodes, first_thread_share)`, partitioning where cumulative
/// position counts balance and assigning threads proportionally.
fn split(nodes: &[Node], children: &[usize], threads: usize) -> (usize, usize, usize) {
    let positions = |child: usize| nodes[child].end - nodes[child].first;
    let total: usize = children.iter().map(|&child| positions(child)).sum();
    let (mut before, mut middle) = (0, 1);
    for (index, &child) in children.iter().enumerate().take(children.len() - 1) {
        before += positions(child);
        middle = index + 1;
        if 2 * before >= total {
            break;
        }
    }
    let first = children[0];
    let last = children[middle - 1];
    let subtrees = last + 1 - nodes[first].earliest;
    let share = (threads * before + total / 2) / total.max(1);
    (middle, subtrees, share)
}

/// Traversal policy specifying a subtree's frontal thread share and whether child subtrees
/// spawn rayon tasks.
#[derive(Clone, Copy)]
struct Visits {
    threads: usize,
    tasks: bool,
}

impl Visits {
    /// Creates a traversal policy for `threads` threads, enabling rayon tasks when `threads >= 2`.
    fn on(threads: usize) -> Self {
        Self {
            threads,
            tasks: threads >= 2,
        }
    }

    /// Returns a copy of the traversal policy with `share` threads assigned to frontal work.
    fn sharing(self, share: usize) -> Self {
        Self {
            threads: share,
            ..self
        }
    }
}

/// Runs `visit` in postorder over the subtree rooted at `index`, passing each node's index,
/// thread share, mutable state, descendant states from `earliest`, and whether any descendant
/// returned `true`.
///
/// Returns `true` if any visit returned `true`, or the first error in node order.
fn upward<S: Send, E: Send>(
    nodes: &[Node],
    (index, visits): (usize, Visits),
    states: &mut [S],
    visit: &(impl Fn((usize, usize), &mut S, &[S], bool) -> Result<bool, E> + Sync),
) -> Result<bool, E> {
    let Some((state, below)) = states.split_last_mut() else {
        return Ok(false);
    };
    let any = upward_all(nodes, (&nodes[index].children, visits), below, visit)?;
    Ok(visit((index, visits.threads), state, below, any)? || any)
}

/// Runs [`upward`] across `children`, whose states partition `states` in order.
///
/// When tasks are enabled, recursively bisects `children` by position count via `rayon::join`
/// down to interiors; otherwise visits children sequentially.
fn upward_all<S: Send, E: Send>(
    nodes: &[Node],
    (children, visits): (&[usize], Visits),
    states: &mut [S],
    visit: &(impl Fn((usize, usize), &mut S, &[S], bool) -> Result<bool, E> + Sync),
) -> Result<bool, E> {
    match children {
        [] => return Ok(false),
        [only] => return upward(nodes, (*only, visits), states, visit),
        _ => {}
    }
    if !visits.tasks {
        let mut any = false;
        let mut rest = states;
        for &child in children {
            let (subtree, after) = rest.split_at_mut(child + 1 - nodes[child].earliest);
            any |= upward(nodes, (child, visits), subtree, visit)?;
            rest = after;
        }
        return Ok(any);
    }
    let (middle, subtrees, share) = split(nodes, children, visits.threads);
    let (first, later) = children.split_at(middle);
    let (before, after) = states.split_at_mut(subtrees);
    let rest = visits.threads.saturating_sub(share).max(1);
    let (one, other) = (visits.sharing(share.max(1)), visits.sharing(rest));
    let (one, other) = rayon::join(
        || upward_all(nodes, (first, one), before, visit),
        || upward_all(nodes, (later, other), after, visit),
    );
    Ok(one? | other?)
}

/// Runs `visit` in preorder over the subtree rooted at `index`, passing each node's state,
/// mutable descendant states, and the corresponding slice of `solution`.
fn downward<S: Send, T: Send>(
    nodes: &[Node],
    (index, visits): (usize, Visits),
    states: &mut [S],
    solution: &mut [T],
    visit: &(impl Fn(usize, &mut S, &mut [S], &mut [T]) + Sync),
) {
    let Some((state, below)) = states.split_last_mut() else {
        return;
    };
    visit(index, state, below, solution);
    downward_all(
        nodes,
        (&nodes[index].children, visits),
        below,
        solution,
        visit,
    );
}

/// Runs [`downward`] across `children`, following [`upward_all`] scheduling with child states
/// tiling `states` and child positions tiling the prefix of `solution`.
fn downward_all<S: Send, T: Send>(
    nodes: &[Node],
    (children, visits): (&[usize], Visits),
    states: &mut [S],
    solution: &mut [T],
    visit: &(impl Fn(usize, &mut S, &mut [S], &mut [T]) + Sync),
) {
    let (Some(&first_child), Some(&last_child)) = (children.first(), children.last()) else {
        return;
    };
    let solution = &mut solution[..nodes[last_child].end - nodes[first_child].first];
    if let [only] = children {
        downward(nodes, (*only, visits), states, solution, visit);
        return;
    }
    if !visits.tasks {
        let (mut rest, mut written) = (states, solution);
        for &child in children {
            let node = &nodes[child];
            let (subtree, after) = rest.split_at_mut(child + 1 - node.earliest);
            let (part, beyond) = written.split_at_mut(node.end - node.first);
            downward(nodes, (child, visits), subtree, part, visit);
            (rest, written) = (after, beyond);
        }
        return;
    }
    let (middle, subtrees, share) = split(nodes, children, visits.threads);
    let (first, later) = children.split_at(middle);
    let (before, after) = states.split_at_mut(subtrees);
    let boundary = nodes[first[first.len() - 1]].end - nodes[first_child].first;
    let (written, beyond) = solution.split_at_mut(boundary);
    let rest = visits.threads.saturating_sub(share).max(1);
    let (one, other) = (visits.sharing(share.max(1)), visits.sharing(rest));
    rayon::join(
        || downward_all(nodes, (first, one), before, written, visit),
        || downward_all(nodes, (later, other), after, beyond, visit),
    );
}

/// Sparse left-looking factors and dense Schur update for an interior node.
///
/// The interior frame orders local rows and columns before border entries. Any local column
/// whose best local pivot falls below threshold is delayed alongside an equal number of
/// unpivoted local rows.
#[derive(Clone)]
struct Interior<T> {
    /// Flag indicating whether cached pivots from the last factorization remain valid.
    complete: bool,
    /// Column-by-column sparse `L U` factors and elimination workspace.
    factors: ColumnFactors<T>,
    /// Frame column index eliminated at each step.
    pivot_columns: Vec<usize>,
    /// Delayed local column indices, in order.
    delayed: Vec<usize>,
    /// Row indices of the update: unpivoted local rows followed by border rows.
    update_rows: Vec<usize>,
    /// Upper-triangular `U` entries for each update column, indexed by elimination step.
    outer: Triangle<T>,
    /// Dense column-major Schur complement update.
    update: Vec<T>,
    /// Ordered matrix positions of the delayed rows and delayed columns.
    delayed_rows: Vec<usize>,
    delayed_columns: Vec<usize>,
}

impl<T: Scalar> Default for Interior<T> {
    fn default() -> Self {
        Self {
            complete: false,
            factors: ColumnFactors::new(0),
            pivot_columns: Vec::new(),
            delayed: Vec::new(),
            update_rows: Vec::new(),
            outer: Triangle::new(0),
            update: Vec::new(),
            delayed_rows: Vec::new(),
            delayed_columns: Vec::new(),
        }
    }
}

/// Returns the sparse entries assembled by `node` for frame `column`, covering local columns
/// and border-column entries in local rows.
fn frame_column(node: &Node, column: usize) -> &[(usize, usize)] {
    let own = node.size();
    if column < own {
        node.columns.of(column)
    } else {
        node.border_columns.of(column - own)
    }
}

impl<T: Scalar> Interior<T> {
    /// Returns `true` when both interiors hold matching cached pivot patterns or neither holds
    /// cached pivots.
    fn pivots_alike(&self, other: &Self) -> bool {
        match (self.complete, other.complete) {
            (false, false) => true,
            (true, true) => {
                self.factors.patterns_alike(&other.factors)
                    && self.pivot_columns == other.pivot_columns
                    && self.delayed == other.delayed
                    && self.outer.pattern_eq(&other.outer)
            }
            _ => false,
        }
    }

    /// Factors the local columns of `node` from `values`, searching afresh for every pivot.
    fn search(&mut self, node: &Node, values: &[T]) -> Result<(), LinearError> {
        let own = node.size();
        let size = own + node.border.len();
        self.complete = false;
        self.factors.reset(size);
        self.pivot_columns.clear();
        self.delayed.clear();
        self.outer.truncate(0);
        for column in 0..own {
            self.eliminate(node, values, column)?;
        }
        self.update_rows.clear();
        self.update_rows
            .extend((0..own).filter(|&row| self.factors.positions[row] == NONE));
        debug_assert_eq!(
            self.update_rows.len(),
            self.delayed.len(),
            "a row left for each column"
        );
        self.update_rows.extend(own..size);
        let width = self.update_rows.len();
        let entries = width.checked_mul(width).ok_or(LinearError::TooLarge)?;
        self.update.clear();
        self.update.try_reserve(entries)?;
        self.update.resize(entries, T::ZERO);
        for index in 0..width {
            let column = self.update_column(index, own);
            let entries = frame_column(node, column);
            self.factors.scatter_and_solve(entries, values, own + index);
            self.factors.store_upper_into(&mut self.outer)?;
            self.gather(index);
        }
        self.delayed_rows.clear();
        self.delayed_rows.extend(
            self.update_rows[..self.delayed.len()]
                .iter()
                .map(|&row| node.own + row),
        );
        self.delayed_columns.clear();
        self.delayed_columns
            .extend(self.delayed.iter().map(|&column| node.own + column));
        self.complete = true;
        Ok(())
    }

    /// Eliminates local `column` of `node` from `values`, choosing the matched row or largest
    /// local pivot above threshold and delaying the column otherwise.
    fn eliminate(&mut self, node: &Node, values: &[T], column: usize) -> Result<(), LinearError> {
        let entries = node.columns.of(column);
        let step = self.factors.pivots.len();
        let found = self
            .factors
            .eliminate_column(entries, values, (column, step, node.size()))?;
        let Some((chosen, pivot)) = found else {
            self.delayed.push(column);
            return Ok(());
        };
        self.factors.pivots.push(pivot);
        self.factors.pivot_rows.push(chosen);
        self.pivot_columns.push(column);
        Ok(())
    }

    /// Returns the frame column index corresponding to column `index` of the update.
    fn update_column(&self, index: usize, own: usize) -> usize {
        match self.delayed.get(index) {
            Some(&column) => column,
            None => own + index - self.delayed.len(),
        }
    }

    /// Copies the active work vector at update rows into column `index` of `update` and clears
    /// those work slots.
    fn gather(&mut self, index: usize) {
        let width = self.update_rows.len();
        let column = &mut self.update[index * width..(index + 1) * width];
        for (value, &row) in column.iter_mut().zip(&self.update_rows) {
            *value = self.factors.work[row];
            self.factors.work[row] = T::ZERO;
        }
    }

    /// Refactors the local columns of `node` from `values` using cached pivots, returning `true`
    /// when every pivot meets the threshold.
    fn refactor(&mut self, node: &Node, values: &[T]) -> bool {
        for step in 0..self.factors.pivots.len() {
            let entries = node.columns.of(self.pivot_columns[step]);
            let Some(pivot) = self.factors.refactor_column(entries, values, step) else {
                return false;
            };
            self.factors.pivots[step] = pivot;
        }
        let own = node.size();
        for index in 0..self.update_rows.len() {
            let entries = frame_column(node, self.update_column(index, own));
            self.outer.replay(
                (entries, values),
                (&self.factors.pivot_rows, &self.factors.lower),
                &mut self.factors.work,
                index,
            );
            self.gather(index);
        }
        true
    }

    /// Executes the upward solve pass (or transposed pass when `transposed` is `true`) from
    /// `given`, writing step values and upward border contributions into `space`.
    fn forward(&self, node: &Node, given: &[T], space: &mut Scratch<T>, transposed: bool) {
        let own = node.size();
        let Scratch {
            frame, steps, up, ..
        } = space;
        frame.clear();
        frame.extend_from_slice(&given[node.own..node.end]);
        frame.resize(own + node.border.len(), T::ZERO);
        steps.clear();
        up.clear();
        if transposed {
            for step in 0..self.factors.pivots.len() {
                let right_hand_side = frame[self.pivot_columns[step]];
                let value = self
                    .factors
                    .upper
                    .subtract_dot(step, right_hand_side, steps);
                steps.push(value / self.factors.pivots[step]);
            }
            for index in 0..self.update_rows.len() {
                let right_hand_side = frame[self.update_column(index, own)];
                up.push(self.outer.subtract_dot(index, right_hand_side, steps));
            }
        } else {
            for (step, &row) in self.factors.pivot_rows.iter().enumerate() {
                let value = frame[row];
                steps.push(value);
                self.factors.lower.subtract(step, value, frame);
            }
            up.extend(self.update_rows.iter().map(|&row| frame[row]));
        }
    }

    /// Executes the downward solve pass (or transposed pass when `transposed` is `true`),
    /// writing local solution components into `solution`.
    fn backward(&self, node: &Node, space: &mut Scratch<T>, solution: &mut [T], transposed: bool) {
        let offset = node.own - node.first;
        if transposed {
            let Scratch {
                frame, steps, down, ..
            } = space;
            for (&row, &value) in self.update_rows.iter().zip(down.iter()) {
                frame[row] = value;
            }
            for step in (0..self.factors.pivots.len()).rev() {
                let value = self.factors.lower.subtract_dot(step, steps[step], frame);
                let row = self.factors.pivot_rows[step];
                frame[row] = value;
                solution[offset + row] = value;
            }
        } else {
            let Scratch { steps, down, .. } = space;
            for index in (0..down.len()).rev() {
                self.outer.subtract(index, down[index], steps);
            }
            for step in (0..self.factors.pivots.len()).rev() {
                let value = steps[step] / self.factors.pivots[step];
                solution[offset + self.pivot_columns[step]] = value;
                self.factors.upper.subtract(step, value, steps);
            }
        }
    }
}

/// Applies elimination steps in `steps` from dense factor columns `earlier` to dense `column`,
/// subtracting each step's subdiagonal multipliers scaled by `column[step]`.
fn take_in<T: Scalar>(earlier: &[T], column: &mut [T], steps: std::ops::Range<usize>) {
    let size = column.len();
    for step in steps {
        let multiplier = column[step];
        if multiplier != T::ZERO {
            let factors = &earlier[step * size + step + 1..(step + 1) * size];
            for (value, &factor) in column[step + 1..].iter_mut().zip(factors) {
                *value -= factor * multiplier;
            }
        }
    }
}

/// Elimination action recorded for a single frontal column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    /// Pivot chosen at the recorded row slot and swapped into the current step.
    Pivot(usize),
    /// Column delayed and swapped with the final uneliminated slot.
    Delay(usize),
}

/// Replays the row (`rows = true`) or column (`rows = false`) swaps recorded in
/// `operations` onto `values`.
fn replay_swaps<U>(operations: &[Operation], values: &mut [U], rows: bool) {
    let mut step = 0;
    for &operation in operations {
        match operation {
            Operation::Pivot(row) => {
                if rows {
                    values.swap(step, row);
                }
                step += 1;
            }
            Operation::Delay(other) => {
                if !rows {
                    values.swap(step, other);
                }
            }
        }
    }
}

/// Dense in-place frontal factors for a separator node, ordering local and delayed entries
/// before border entries.
#[derive(Clone)]
struct Front<T> {
    /// Flag indicating whether cached pivots from the last factorization remain valid.
    complete: bool,
    /// Ordered matrix positions of the frontal rows and columns in eliminated slot order.
    rows: Vec<usize>,
    columns: Vec<usize>,
    /// Mapping from assembled frame slot to eliminated slot for rows and columns.
    row_slots: Vec<usize>,
    column_slots: Vec<usize>,
    /// Number of fully summed rows and columns (local plus delayed from children).
    fully: usize,
    /// Number of elimination steps completed.
    steps: usize,
    /// Dense column-major front storing unit-diagonal `L` below the diagonal, `U` on and above
    /// it, and the trailing Schur update.
    frame: Vec<T>,
    /// Recorded pivot and delay sequence in elimination order.
    operations: Vec<Operation>,
    /// Count of earlier elimination steps already applied to each column slot.
    applied: Vec<usize>,
    /// Scratch buffers for replaying slot permutations and mapping child updates into the front.
    origins: Vec<usize>,
    slots: Vec<usize>,
}

impl<T> Default for Front<T> {
    fn default() -> Self {
        Self {
            complete: false,
            rows: Vec::new(),
            columns: Vec::new(),
            row_slots: Vec::new(),
            column_slots: Vec::new(),
            fully: 0,
            steps: 0,
            frame: Vec::new(),
            operations: Vec::new(),
            applied: Vec::new(),
            origins: Vec::new(),
            slots: Vec::new(),
        }
    }
}

impl<T: Scalar> Front<T> {
    /// Returns the assembled frame slot of each row or column in child `which`'s update, placing
    /// delayed entries at `offset`.
    fn child_slots(
        node: &Node,
        (child, which): (&Part<T>, usize),
        offset: usize,
        delayed_in: usize,
    ) -> impl Iterator<Item = usize> {
        let own = node.size();
        let delayed = child.delayed_rows().len();
        (offset..offset + delayed).chain(node.child_borders[which].iter().map(move |&index| {
            if index < own {
                index
            } else {
                index + delayed_in
            }
        }))
    }

    /// Assembles the dense front of `node` from matrix `values` and child updates in `below`
    /// (offset by `base`).
    fn assemble(
        &mut self,
        node: &Node,
        values: &[T],
        below: &[Part<T>],
        base: usize,
    ) -> Result<(), LinearError> {
        let own = node.size();
        let children = || node.children.iter().map(|&child| &below[child - base]);
        self.rows.clear();
        self.rows.extend(node.own..node.end);
        self.columns.clear();
        self.columns.extend(node.own..node.end);
        for child in children() {
            self.rows.extend_from_slice(child.delayed_rows());
            self.columns.extend_from_slice(child.delayed_columns());
        }
        self.fully = self.rows.len();
        debug_assert_eq!(
            self.fully,
            self.columns.len(),
            "a row delayed for each column"
        );
        self.rows.extend_from_slice(&node.border);
        self.columns.extend_from_slice(&node.border);
        let size = self.rows.len();
        let delayed_in = self.fully - own;
        let entries = size.checked_mul(size).ok_or(LinearError::TooLarge)?;
        self.frame.clear();
        self.frame.try_reserve(entries)?;
        self.frame.resize(entries, T::ZERO);
        let slot = |index: usize| {
            if index < own {
                index
            } else {
                index + delayed_in
            }
        };
        for column in 0..own {
            for &(row, value) in node.columns.of(column) {
                self.frame[slot(row) + column * size] += values[value];
            }
        }
        for index in 0..node.border.len() {
            let column = self.fully + index;
            for &(row, value) in node.border_columns.of(index) {
                self.frame[row + column * size] += values[value];
            }
        }
        let mut offset = own;
        for (which, child) in children().enumerate() {
            let update = child.update();
            self.slots.clear();
            self.slots
                .extend(Self::child_slots(node, (child, which), offset, delayed_in));
            for (column, &into) in self.slots.iter().enumerate() {
                let target = &mut self.frame[into * size..(into + 1) * size];
                for (row, &at) in self.slots.iter().enumerate() {
                    target[at] += update.get(row, column);
                }
            }
            offset += update.delayed;
        }
        Ok(())
    }

    /// Factors the front of `node` across up to `threads` threads, searching for pivots and
    /// reporting [`LinearError::Singular`] if any column remains delayed at the root.
    fn search(
        &mut self,
        node: &Node,
        values: &[T],
        (below, base): (&[Part<T>], usize),
        (threads, root): (usize, bool),
    ) -> Result<(), LinearError> {
        self.complete = false;
        self.assemble(node, values, below, base)?;
        self.operations.clear();
        let (steps, _) = self.eliminate(false, threads);
        if root && steps < self.fully {
            return Err(LinearError::Singular {
                columns: vec![self.columns[steps]],
            });
        }
        self.complete = true;
        Ok(())
    }

    /// Refactors the front of `node` across up to `threads` threads by replaying cached
    /// operations, returning `true` when every pivot meets the threshold.
    fn refactor(
        &mut self,
        node: &Node,
        values: &[T],
        (below, base): (&[Part<T>], usize),
        threads: usize,
    ) -> Result<bool, LinearError> {
        self.assemble(node, values, below, base)?;
        let (_, held) = self.eliminate(true, threads);
        self.complete = held;
        Ok(held)
    }

    /// Eliminates fully summed columns of the front across up to `threads` threads, searching
    /// for pivots or replaying cached operations when `replay` is `true`.
    ///
    /// Returns `(steps, held)` where `held` indicates whether every replayed pivot met the
    /// threshold, stopping at the first failure. Each column absorbs pending earlier steps in
    /// order right before elimination, and trailing columns absorb panels in parallel across
    /// `threads`. Row swaps apply across the full width immediately so every entry receives
    /// the exact same operation sequence regardless of panel size.
    fn eliminate(&mut self, replay: bool, threads: usize) -> (usize, bool) {
        let size = self.rows.len();
        let panel = if threads < 2 {
            self.fully
        } else {
            self.fully.div_ceil(threads)
        }
        .max(1);
        self.applied.clear();
        self.applied.resize(size, 0);
        let (mut step, mut left, mut done, mut taken) = (0, self.fully, 0, 0);
        while step < left {
            self.catch_up(step, step);
            let operation = if replay {
                self.operations[done]
            } else {
                let chosen = self.choose(step, left);
                self.operations.push(chosen);
                chosen
            };
            done += 1;
            match operation {
                Operation::Pivot(row) => {
                    self.swap_rows(step, row);
                    if replay && !self.holds(step) {
                        return (step, false);
                    }
                    self.divide(step);
                    step += 1;
                    taken += 1;
                    if taken == panel {
                        self.catch_up_after(step, threads);
                        taken = 0;
                    }
                }
                Operation::Delay(other) => {
                    left -= 1;
                    self.swap_columns(step, other);
                    self.applied.swap(step, other);
                }
            }
        }
        self.catch_up_after(step, threads);
        self.finish(step);
        (step, true)
    }

    /// Returns frontal column `step` together with its minimum acceptable pivot magnitude.
    fn column_threshold(&self, step: usize) -> (&[T], f64) {
        let size = self.rows.len();
        let column = &self.frame[step * size..(step + 1) * size];
        let largest = column[step..]
            .iter()
            .map(|value| value.magnitude())
            .fold(0.0, f64::max);
        (column, PIVOT_TOLERANCE * largest)
    }

    /// Chooses the elimination [`Operation`] for frontal column `step`, preferring the matched
    /// row or largest eligible pivot above threshold and delaying to slot `left - 1` otherwise.
    fn choose(&self, step: usize, left: usize) -> Operation {
        let (column, least) = self.column_threshold(step);
        let (mut best, mut chosen, mut diagonal) = (0.0, NONE, NONE);
        for (row, value) in column.iter().enumerate().take(self.fully).skip(step) {
            let magnitude = value.magnitude();
            if magnitude > best {
                (best, chosen) = (magnitude, row);
            }
            if self.rows[row] == self.columns[step] {
                diagonal = row;
            }
        }
        if diagonal != NONE
            && column[diagonal].magnitude() > 0.0
            && column[diagonal].magnitude() >= least
        {
            Operation::Pivot(diagonal)
        } else if chosen != NONE && best >= least {
            Operation::Pivot(chosen)
        } else {
            Operation::Delay(left - 1)
        }
    }

    /// Returns `true` when the pivot at `step` is nonzero and meets the column threshold.
    fn holds(&self, step: usize) -> bool {
        let (column, least) = self.column_threshold(step);
        let magnitude = column[step].magnitude();
        magnitude > 0.0 && magnitude >= least
    }

    /// Scales the subdiagonal entries of frontal column `step` by its diagonal pivot to form `L`.
    fn divide(&mut self, step: usize) {
        let size = self.rows.len();
        let column = &mut self.frame[step * size..(step + 1) * size];
        let pivot = column[step];
        for value in &mut column[step + 1..] {
            *value = *value / pivot;
        }
    }

    /// Applies all pending earlier elimination steps up to `until` to frontal `column`.
    fn catch_up(&mut self, column: usize, until: usize) {
        let size = self.rows.len();
        let (earlier, rest) = self.frame.split_at_mut(column * size);
        take_in(earlier, &mut rest[..size], self.applied[column]..until);
        self.applied[column] = until;
    }

    /// Applies all pending earlier elimination steps up to `step` across trailing columns from
    /// `step` onward using up to `threads` threads.
    fn catch_up_after(&mut self, step: usize, threads: usize) {
        let size = self.rows.len();
        let (earlier, rest) = self.frame.split_at_mut(step * size);
        let applied = &mut self.applied[step..];
        let columns = applied.len();
        if columns == 0 {
            return;
        }
        if threads < 2 || columns == 1 {
            for (column, taken) in rest.chunks_exact_mut(size).zip(applied) {
                take_in(earlier, column, *taken..step);
                *taken = step;
            }
            return;
        }
        let share = columns.div_ceil(threads);
        rest.par_chunks_mut(share * size)
            .zip(applied.par_chunks_mut(share))
            .for_each(|(columns, applied)| {
                for (column, taken) in columns.chunks_exact_mut(size).zip(applied) {
                    take_in(earlier, column, *taken..step);
                    *taken = step;
                }
            });
    }

    /// Records final eliminated row and column slot mappings after `steps` eliminations.
    fn finish(&mut self, steps: usize) {
        self.steps = steps;
        let size = self.rows.len();
        // Replay swaps on the identity permutation to map assembled slots to eliminated slots.
        for (slots, rows) in [(&mut self.row_slots, true), (&mut self.column_slots, false)] {
            self.origins.clear();
            self.origins.extend(0..size);
            replay_swaps(&self.operations, &mut self.origins, rows);
            slots.clear();
            slots.resize(size, 0);
            for (slot, &assembled) in self.origins.iter().enumerate() {
                slots[assembled] = slot;
            }
        }
    }

    /// Swaps rows `one` and `other` across the entire frontal matrix.
    fn swap_rows(&mut self, one: usize, other: usize) {
        if one == other {
            return;
        }
        let size = self.rows.len();
        for column in self.frame.chunks_exact_mut(size) {
            column.swap(one, other);
        }
        self.rows.swap(one, other);
    }

    /// Swaps columns `one` and `other` (`one < other`) across the entire frontal matrix.
    fn swap_columns(&mut self, one: usize, other: usize) {
        if one == other {
            return;
        }
        let size = self.rows.len();
        let (left, right) = self.frame.split_at_mut(other * size);
        left[one * size..(one + 1) * size].swap_with_slice(&mut right[..size]);
        self.columns.swap(one, other);
    }

    /// Executes the upward solve pass (or transposed pass when `transposed` is `true`),
    /// assembling `given` and child updates and writing the upward contribution into `space`.
    fn forward(
        &self,
        node: &Node,
        given: &[T],
        space: &mut Scratch<T>,
        (parts, below, base): (&[Part<T>], &[Scratch<T>], usize),
        transposed: bool,
    ) {
        let size = self.rows.len();
        let own = node.size();
        let delayed_in = self.fully - own;
        let Scratch { frame, up, .. } = space;
        frame.clear();
        frame.extend_from_slice(&given[node.own..node.end]);
        frame.resize(size, T::ZERO);
        let mut offset = own;
        for (which, &child) in node.children.iter().enumerate() {
            let part = &parts[child];
            let values = &below[child - base].up;
            for (slot, &value) in
                Self::child_slots(node, (part, which), offset, delayed_in).zip(values.iter())
            {
                frame[slot] += value;
            }
            offset += part.delayed_rows().len();
        }
        // Each step swapped rows (or delayed columns) across the whole frame, so apply all
        // recorded swaps to the right-hand side first.
        replay_swaps(&self.operations, frame, !transposed);
        if transposed {
            for column in 0..size {
                let entries = &self.frame[column * size..column * size + column.min(self.steps)];
                let mut value = frame[column];
                for (&factor, &earlier) in entries.iter().zip(frame.iter()) {
                    value -= factor * earlier;
                }
                frame[column] = if column < self.steps {
                    value / self.frame[column * size + column]
                } else {
                    value
                };
            }
        } else {
            for step in 0..self.steps {
                let value = frame[step];
                let multipliers = &self.frame[step * size + step + 1..(step + 1) * size];
                for (target, &factor) in frame[step + 1..].iter_mut().zip(multipliers) {
                    *target -= factor * value;
                }
            }
        }
        up.clear();
        up.extend_from_slice(&frame[self.steps..]);
    }

    /// Executes the downward solve pass (or transposed pass when `transposed` is `true`),
    /// writing eliminated solution entries into `solution` and passing updates to children.
    fn backward(
        &self,
        node: &Node,
        space: &mut Scratch<T>,
        (parts, below): (&[Part<T>], &mut [Scratch<T>]),
        solution: &mut [T],
        transposed: bool,
    ) {
        let size = self.rows.len();
        let Scratch {
            frame, steps, down, ..
        } = space;
        steps.clear();
        steps.resize(self.steps, T::ZERO);
        steps.extend_from_slice(down);
        if transposed {
            for step in (0..self.steps).rev() {
                let mut value = frame[step];
                let entries = &self.frame[step * size + step + 1..(step + 1) * size];
                for (&factor, &later) in entries.iter().zip(&steps[step + 1..]) {
                    value -= factor * later;
                }
                steps[step] = value;
            }
        } else {
            for column in (self.steps..size).rev() {
                let value = steps[column];
                let entries = &self.frame[column * size..column * size + self.steps];
                for (target, &factor) in frame.iter_mut().zip(entries) {
                    *target -= factor * value;
                }
            }
            for step in (0..self.steps).rev() {
                let value = frame[step] / self.frame[step * size + step];
                steps[step] = value;
                let entries = &self.frame[step * size..step * size + step];
                for (target, &factor) in frame.iter_mut().zip(entries) {
                    *target -= factor * value;
                }
            }
        }
        let (positions, slots) = if transposed {
            (&self.rows, &self.row_slots)
        } else {
            (&self.columns, &self.column_slots)
        };
        for (&position, &value) in positions.iter().zip(steps.iter()).take(self.steps) {
            solution[position - node.first] = value;
        }
        self.pass_down(node, (parts, below), steps, slots);
    }

    /// Extracts each child's backward-pass update values from `values` via `slots` and writes
    /// them into child scratch buffers.
    fn pass_down(
        &self,
        node: &Node,
        (parts, below): (&[Part<T>], &mut [Scratch<T>]),
        values: &[T],
        slots: &[usize],
    ) {
        let own = node.size();
        let delayed_in = self.fully - own;
        let base = node.earliest;
        let mut offset = own;
        for (which, &child) in node.children.iter().enumerate() {
            let part = &parts[child];
            let down = &mut below[child - base].down;
            down.clear();
            down.extend(
                Self::child_slots(node, (part, which), offset, delayed_in)
                    .map(|slot| values[slots[slot]]),
            );
            offset += part.delayed_rows().len();
        }
    }
}
