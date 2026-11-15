//! The factorization of a dissected block, by the multifrontal method over its dissection's
//! tree (Duff and Reid, "The multifrontal solution of indefinite sparse symmetric linear
//! equations", ACM Transactions on Mathematical Software 9(3), 1983), and the solves with
//! its factors.
//!
//! Each interior factors its own columns by the left-looking elimination of the block's
//! other columns, choosing each pivot among its own rows, and leaves its update: the
//! Schur complement of its rows and columns on its border, dense. Each separator assembles
//! its frame, dense: the entries of the matrix it owns, then its children's updates in
//! their order, and eliminates its own columns, choosing each pivot among its own rows and
//! those its children could not pivot, and leaves the update of the rest. A column whose
//! best pivot within its node falls short of the threshold moves, with a row its node did
//! not pivot, to its parent's frame, delayed rather than perturbed, so the factors stay
//! exact; the root pivots on any row left.
//!
//! The subtrees of a node's children share no entry, so given two threads or more they
//! factor at once, as rayon tasks split down to the interiors, which the pool's threads
//! take as they free, and a front's columns spread over its share of the threads given, in
//! proportion to its subtree's positions; every number is computed by the same operations
//! in the same order however they are spread, so the factors and the solutions are the same
//! bits on any number of threads.

use rayon::prelude::*;

use super::dissection::{Dissection, Node};
use super::numeric::{PIVOT_TOLERANCE, Triangle};
use super::{LinearError, Scalar};

/// No index: a row not pivoted yet.
const NONE: usize = usize::MAX;

/// The factors of a dissected block: each node's, in the order of the dissection's nodes.
#[derive(Clone)]
pub(super) struct Tree<T> {
    parts: Vec<Part<T>>,
}

/// The factors of one node of a dissection.
#[derive(Clone)]
enum Part<T> {
    Interior(Box<Interior<T>>),
    Front(Box<Front<T>>),
}

/// What a node leaves its parent: the Schur complement of the rows and columns it did not
/// pivot, and of its border's, dense by columns; its rows and its columns are those it
/// delayed, in their order, then its border's.
struct Update<'a, T> {
    /// The number of rows, and of columns, it delayed.
    delayed: usize,
    values: &'a [T],
    /// The index in `values` of its first row's entry in its first column.
    first: usize,
    /// The distance in `values` from one column to the next.
    leading: usize,
}

impl<T: Scalar> Update<'_, T> {
    /// The entry in `row` and `column`.
    fn get(&self, row: usize, column: usize) -> T {
        self.values[self.first + row + column * self.leading]
    }
}

impl<T: Scalar> Tree<T> {
    /// No factors yet of the block `dissection` dissects.
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

    /// Forgets every pivot, so that the next factorization searches for each.
    pub(super) fn forget(&mut self) {
        for part in &mut self.parts {
            match part {
                Part::Interior(interior) => interior.complete = false,
                Part::Front(front) => front.complete = false,
            }
        }
    }

    /// Whether every node keeps its pivots, so that a factorization can refactor with them.
    pub(super) fn is_complete(&self) -> bool {
        self.parts.iter().all(|part| match part {
            Part::Interior(interior) => interior.complete,
            Part::Front(front) => front.complete,
        })
    }

    /// Factors the block of `values`, the matrix's values by slot, spread as the module says
    /// over `threads` threads, or on the calling thread alone for one: each node keeps its
    /// pivots while they reach the threshold and its children kept theirs, and else
    /// searches for them. Returns whether any node searched.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] naming the position of the first column left with
    /// no nonzero pivot, and [`LinearError::TooLarge`] when the factors do not fit in
    /// memory.
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

    /// Whether every factorization from now on keeps the same pivots as `other`'s, as
    /// [`super::numeric::Numeric::pivots_alike`] says.
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

    /// The number of columns the last factorization delayed from a node to its parent.
    pub(super) fn delays(&self) -> usize {
        self.parts
            .iter()
            .map(|part| match part {
                Part::Interior(interior) => interior.delayed.len(),
                Part::Front(front) => front.fully - front.steps,
            })
            .sum()
    }

    /// The number of entries of the factors.
    pub(super) fn entries(&self) -> usize {
        self.parts
            .iter()
            .map(|part| match part {
                Part::Interior(interior) => {
                    interior.lower.indices.len()
                        + interior.upper.indices.len()
                        + interior.pivots.len()
                        + interior.outer.indices.len()
                }
                Part::Front(front) => {
                    let size = front.rows.len();
                    (0..front.steps).map(|step| 2 * (size - step) - 1).sum()
                }
            })
            .sum()
    }

    /// Solves the block's system with the scratch space `scratch`, its subtrees spread as
    /// the module says for two `threads` or more: from the right-hand side by row of the ordered matrix in `given`,
    /// the solution by column in `solution`, the block's positions of the ordered matrix;
    /// or, `transposed`, from the right-hand side by column, the solution by row.
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
                match (&self.parts[index], transposed) {
                    (Part::Interior(interior), false) => interior.forward(node, given, space),
                    (Part::Interior(interior), true) => {
                        interior.forward_transposed(node, given, space);
                    }
                    (Part::Front(front), false) => front.forward(node, given, space, below),
                    (Part::Front(front), true) => {
                        front.forward_transposed(node, given, space, below);
                    }
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
                match (&self.parts[index], transposed) {
                    (Part::Interior(interior), false) => interior.backward(node, space, solution),
                    (Part::Interior(interior), true) => {
                        interior.backward_transposed(node, space, solution);
                    }
                    (Part::Front(front), false) => {
                        front.backward(node, space, (&self.parts[..], below), solution);
                    }
                    (Part::Front(front), true) => {
                        front.backward_transposed(node, space, (&self.parts[..], below), solution);
                    }
                }
            },
        );
    }
}

impl<T: Scalar> Part<T> {
    /// The positions of the rows it delayed, in their order.
    fn delayed_rows(&self) -> &[usize] {
        match self {
            Self::Interior(interior) => &interior.delayed_rows,
            Self::Front(front) => &front.rows[front.steps..front.fully],
        }
    }

    /// The positions of the columns it delayed, in their order.
    fn delayed_columns(&self) -> &[usize] {
        match self {
            Self::Interior(interior) => &interior.delayed_columns,
            Self::Front(front) => &front.columns[front.steps..front.fully],
        }
    }

    /// Its update.
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

/// The scratch space of one node's solves.
#[derive(Debug, Clone)]
pub(super) struct Scratch<T> {
    /// The node's frame: a value for each of its rows, or for each of its columns.
    frame: Vec<T>,
    /// A value for each step, or for each column of its frame.
    steps: Vec<T>,
    /// The values it passes its parent: one for each row of its update, or each column.
    up: Vec<T>,
    /// The values its parent passes it: one for each column of its update, or each row.
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

/// Where the subtrees of `children`, consecutive in the nodes' order, split for two tasks
/// sharing `threads` threads: the number of the children before the split, the number of
/// their subtrees' nodes, and the share of the threads their task's fronts spread over,
/// in proportion to their positions, the split where the positions balance; each task's
/// fronts spread over at least one thread, the other's over the rest.
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

/// How the nodes of a subtree are visited: their fronts' share of the threads, and whether
/// its subtrees are rayon tasks, which any thread of the pool takes as it frees, or are
/// visited one after another on the calling thread.
#[derive(Clone, Copy)]
struct Visits {
    threads: usize,
    tasks: bool,
}

impl Visits {
    /// The visits of a tree spread over `threads` threads: as tasks for two or more.
    fn on(threads: usize) -> Self {
        Self {
            threads,
            tasks: threads >= 2,
        }
    }

    /// The visits of one of two subtrees, whose fronts take `share` of the threads.
    fn sharing(self, share: usize) -> Self {
        Self {
            threads: share,
            ..self
        }
    }
}

/// Runs `visit` on each node of the subtree of `index`, whose nodes' states are `states`,
/// each node after its children, as `visits` says: with its index and its fronts' share of
/// the threads, its state, its descendants', from its subtree's earliest node, and whether
/// any visit of them returned `true`. Returns whether any visit returned `true`, or the
/// error of the first that failed, in the nodes' order.
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

/// Runs [`upward`] on the subtree of each of `children`, whose states tile `states` in
/// their order, as `visits` says: as tasks, those of the children where their positions
/// balance on one with their share of the threads, the rest on another with the rest, each
/// splitting again down to the interiors, so that a thread that frees takes whichever is
/// left; or else one after another.
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

/// Runs `visit` on each node of the subtree of `index`, whose nodes' states are `states`,
/// each node before its children, as `visits` says: with its state, its descendants' to
/// write to, and the part of `solution` at its subtree's positions.
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

/// Runs [`downward`] on the subtree of each of `children`, whose states tile `states` and
/// whose positions tile `solution`'s first, in their order, as [`upward_all`] does.
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

/// The factors of an interior: of its own rows and columns, by columns, its pivots chosen
/// among its own rows, and its update.
///
/// Its frame's rows and columns are its own, then its border's. A column whose best pivot
/// falls short is delayed, and so is each own row not pivoted, as many.
#[derive(Clone)]
struct Interior<T> {
    /// Whether it keeps the pivots of its last factorization.
    complete: bool,
    /// The row of the frame pivoted at each step.
    pivot_rows: Vec<usize>,
    /// The column of the frame eliminated at each step.
    pivot_columns: Vec<usize>,
    /// The step whose pivot each row of the frame is, or `NONE`.
    steps: Vec<usize>,
    /// `L` below its unit diagonal, by step: each entry's row of the frame.
    lower: Triangle<T>,
    /// `U` above its diagonal, by step: each entry's step, in the order of the eliminations.
    upper: Triangle<T>,
    /// The pivots, by step.
    pivots: Vec<T>,
    /// The own columns delayed, in order.
    delayed: Vec<usize>,
    /// The rows of the update: the own rows not pivoted, in order, then the border's.
    update_rows: Vec<usize>,
    /// `U`'s entries in each column of the update, by step, in the order of the
    /// eliminations.
    outer: Triangle<T>,
    /// The update, dense by columns.
    update: Vec<T>,
    /// The positions of the rows and of the columns delayed.
    delayed_rows: Vec<usize>,
    delayed_columns: Vec<usize>,
    /// The column being eliminated, by row of the frame: zero outside it.
    work: Vec<T>,
    /// The search that last visited each row.
    marks: Vec<usize>,
    /// The number of the search being made.
    stamp: usize,
    /// The rows the search is visiting, each with its next entry of `L` to follow.
    visits: Vec<(usize, usize)>,
    /// The rows of the column being eliminated, in an order whose reverse puts every row
    /// after each row whose elimination changes it.
    reach: Vec<usize>,
}

impl<T: Scalar> Default for Interior<T> {
    fn default() -> Self {
        Self {
            complete: false,
            pivot_rows: Vec::new(),
            pivot_columns: Vec::new(),
            steps: Vec::new(),
            lower: Triangle::new(0),
            upper: Triangle::new(0),
            pivots: Vec::new(),
            delayed: Vec::new(),
            update_rows: Vec::new(),
            outer: Triangle::new(0),
            update: Vec::new(),
            delayed_rows: Vec::new(),
            delayed_columns: Vec::new(),
            work: Vec::new(),
            marks: Vec::new(),
            stamp: 0,
            visits: Vec::new(),
            reach: Vec::new(),
        }
    }
}

/// The entries `node` assembles of the column of its frame `column`: one of its own, or
/// after them, one of its border's, whose entries in its own rows only it assembles.
fn frame_column(node: &Node, column: usize) -> &[(usize, usize)] {
    let own = node.size();
    if column < own {
        node.columns.of(column)
    } else {
        node.border_columns.of(column - own)
    }
}

impl<T: Scalar> Interior<T> {
    /// Whether it keeps the same pivots, in the same patterns, as `other`, or neither keeps
    /// any.
    fn pivots_alike(&self, other: &Self) -> bool {
        match (self.complete, other.complete) {
            (false, false) => true,
            (true, true) => {
                self.pivot_rows == other.pivot_rows
                    && self.pivot_columns == other.pivot_columns
                    && self.delayed == other.delayed
                    && self.lower.starts == other.lower.starts
                    && self.lower.indices == other.lower.indices
                    && self.upper.starts == other.upper.starts
                    && self.upper.indices == other.upper.indices
                    && self.outer.starts == other.outer.starts
                    && self.outer.indices == other.outer.indices
            }
            _ => false,
        }
    }

    /// Factors `node`'s columns of the matrix of `values`, searching for every pivot.
    fn search(&mut self, node: &Node, values: &[T]) -> Result<(), LinearError> {
        let own = node.size();
        let size = own + node.border.len();
        self.complete = false;
        self.steps.clear();
        self.steps.resize(size, NONE);
        self.work.clear();
        self.work.resize(size, T::ZERO);
        self.marks.clear();
        self.marks.resize(size, 0);
        self.stamp = 0;
        self.pivot_rows.clear();
        self.pivot_columns.clear();
        self.pivots.clear();
        self.delayed.clear();
        self.lower.truncate(0);
        self.upper.truncate(0);
        self.outer.truncate(0);
        for column in 0..own {
            self.eliminate(node, values, column)?;
        }
        self.update_rows.clear();
        self.update_rows
            .extend((0..own).filter(|&row| self.steps[row] == NONE));
        debug_assert_eq!(
            self.update_rows.len(),
            self.delayed.len(),
            "a row left for each column"
        );
        self.update_rows.extend(own..size);
        let width = self.update_rows.len();
        self.update.clear();
        self.update.try_reserve(width * width)?;
        self.update.resize(width * width, T::ZERO);
        for index in 0..width {
            let column = self.update_column(index, own);
            let entries = frame_column(node, column);
            self.find_reach(entries);
            for &(row, slot) in entries {
                self.work[row] = values[slot];
            }
            self.outer.reserve(self.reach.len())?;
            for &row in self.reach.iter().rev() {
                let step = self.steps[row];
                if step == NONE {
                    continue;
                }
                let multiplier = self.work[row];
                self.work[row] = T::ZERO;
                self.outer.push(step, multiplier);
                let (indices, factors) = self.lower.column(step);
                for (&index, &factor) in indices.iter().zip(factors) {
                    self.work[index] -= factor * multiplier;
                }
            }
            self.outer.close();
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

    /// Eliminates its own `column` of the matrix of `values`, pivoting on the row it
    /// matched with the column, else on its own row of the largest magnitude, if either
    /// reaches the threshold, and else delaying it.
    fn eliminate(&mut self, node: &Node, values: &[T], column: usize) -> Result<(), LinearError> {
        let own = node.size();
        let entries = node.columns.of(column);
        self.find_reach(entries);
        for &(row, slot) in entries {
            self.work[row] = values[slot];
        }
        self.push_reach();
        let (mut largest, mut best, mut chosen) = (0.0, 0.0, NONE);
        for &row in self.reach.iter().rev() {
            if self.steps[row] == NONE {
                let magnitude = self.work[row].magnitude();
                largest = f64::max(largest, magnitude);
                if row < own && magnitude > best {
                    (best, chosen) = (magnitude, row);
                }
            }
        }
        let least = PIVOT_TOLERANCE * largest;
        // The ordering expects the row it matched with this column as the pivot.
        let diagonal = self.work[column].magnitude();
        if self.steps[column] == NONE && diagonal > 0.0 && diagonal >= least {
            chosen = column;
        } else if chosen == NONE || best < least {
            for &row in &self.reach {
                self.work[row] = T::ZERO;
            }
            self.delayed.push(column);
            return Ok(());
        }
        self.upper.reserve(self.reach.len())?;
        self.lower.reserve(self.reach.len())?;
        let step = self.pivots.len();
        for &row in self.reach.iter().rev() {
            let earlier = self.steps[row];
            if earlier != NONE {
                self.upper.push(earlier, self.work[row]);
                self.work[row] = T::ZERO;
            }
        }
        let pivot = self.work[chosen];
        self.work[chosen] = T::ZERO;
        self.steps[chosen] = step;
        for &row in &self.reach {
            if self.steps[row] == NONE {
                self.lower.push(row, self.work[row] / pivot);
                self.work[row] = T::ZERO;
            }
        }
        self.lower.close();
        self.upper.close();
        self.pivots.push(pivot);
        self.pivot_rows.push(chosen);
        self.pivot_columns.push(column);
        Ok(())
    }

    /// The column of the frame that is the update's `index`th: a delayed one, or after
    /// them one of the border's.
    fn update_column(&self, index: usize, own: usize) -> usize {
        match self.delayed.get(index) {
            Some(&column) => column,
            None => own + index - self.delayed.len(),
        }
    }

    /// Moves the column being eliminated, at the update's rows, into the update's
    /// `index`th column, leaving it zero.
    fn gather(&mut self, index: usize) {
        let width = self.update_rows.len();
        let column = &mut self.update[index * width..(index + 1) * width];
        for (value, &row) in column.iter_mut().zip(&self.update_rows) {
            *value = self.work[row];
            self.work[row] = T::ZERO;
        }
    }

    /// Eliminates, from the column being eliminated, each pivot its reach holds, in the
    /// reach's reverse order.
    fn push_reach(&mut self) {
        for &row in self.reach.iter().rev() {
            let step = self.steps[row];
            if step == NONE {
                continue;
            }
            let multiplier = self.work[row];
            let (indices, factors) = self.lower.column(step);
            for (&index, &factor) in indices.iter().zip(factors) {
                self.work[index] -= factor * multiplier;
            }
        }
    }

    /// Finds the rows of the column of the frame whose entries are `entries` once the steps
    /// so far eliminate it: those of its entries and every row an entry of `L` reaches from
    /// them, in `reach`.
    fn find_reach(&mut self, entries: &[(usize, usize)]) {
        self.stamp += 1;
        let stamp = self.stamp;
        self.reach.clear();
        for &(start, _) in entries {
            if self.marks[start] == stamp {
                continue;
            }
            self.marks[start] = stamp;
            self.visits.push((start, self.first_entry(start)));
            while let Some(&(row, entry)) = self.visits.last() {
                let step = self.steps[row];
                if step != NONE && entry < self.lower.starts[step + 1] {
                    if let Some(visit) = self.visits.last_mut() {
                        visit.1 += 1;
                    }
                    let next = self.lower.indices[entry];
                    if self.marks[next] != stamp {
                        self.marks[next] = stamp;
                        self.visits.push((next, self.first_entry(next)));
                    }
                } else {
                    self.visits.pop();
                    self.reach.push(row);
                }
            }
        }
    }

    /// Where the entries of `L` to follow from `row` start: its step's, if it is a pivot.
    fn first_entry(&self, row: usize) -> usize {
        match self.steps[row] {
            NONE => 0,
            step => self.lower.starts[step],
        }
    }

    /// Factors `node`'s columns of the matrix of `values` with the last factorization's
    /// pivots. Returns whether every pivot reaches the threshold; the factors are those of
    /// a search that chose them.
    fn refactor(&mut self, node: &Node, values: &[T]) -> bool {
        for step in 0..self.pivots.len() {
            for &(row, slot) in node.columns.of(self.pivot_columns[step]) {
                self.work[row] = values[slot];
            }
            for entry in self.upper.starts[step]..self.upper.starts[step + 1] {
                let earlier = self.upper.indices[entry];
                let row = self.pivot_rows[earlier];
                let multiplier = self.work[row];
                self.work[row] = T::ZERO;
                self.upper.values[entry] = multiplier;
                let (indices, factors) = self.lower.column(earlier);
                for (&index, &factor) in indices.iter().zip(factors) {
                    self.work[index] -= factor * multiplier;
                }
            }
            let row = self.pivot_rows[step];
            let pivot = self.work[row];
            self.work[row] = T::ZERO;
            let below = self.lower.starts[step]..self.lower.starts[step + 1];
            let largest = self.lower.indices[below.clone()]
                .iter()
                .map(|&index| self.work[index].magnitude())
                .fold(0.0, f64::max);
            let magnitude = pivot.magnitude();
            let passes = magnitude > 0.0 && magnitude >= PIVOT_TOLERANCE * largest;
            for entry in below {
                let index = self.lower.indices[entry];
                self.lower.values[entry] = self.work[index] / pivot;
                self.work[index] = T::ZERO;
            }
            if !passes {
                return false;
            }
            self.pivots[step] = pivot;
        }
        let own = node.size();
        for index in 0..self.update_rows.len() {
            for &(row, slot) in frame_column(node, self.update_column(index, own)) {
                self.work[row] = values[slot];
            }
            for entry in self.outer.starts[index]..self.outer.starts[index + 1] {
                let step = self.outer.indices[entry];
                let row = self.pivot_rows[step];
                let multiplier = self.work[row];
                self.work[row] = T::ZERO;
                self.outer.values[entry] = multiplier;
                let (indices, factors) = self.lower.column(step);
                for (&index, &factor) in indices.iter().zip(factors) {
                    self.work[index] -= factor * multiplier;
                }
            }
            self.gather(index);
        }
        true
    }

    /// The pass up of a solve: from the right-hand side by row in `given`, each step's
    /// value, and the values of its update's rows for its parent.
    fn forward(&self, node: &Node, given: &[T], space: &mut Scratch<T>) {
        let own = node.size();
        let Scratch {
            frame, steps, up, ..
        } = space;
        frame.clear();
        frame.extend_from_slice(&given[node.own..node.end]);
        frame.resize(own + node.border.len(), T::ZERO);
        steps.clear();
        for (step, &row) in self.pivot_rows.iter().enumerate() {
            let value = frame[row];
            steps.push(value);
            let (indices, factors) = self.lower.column(step);
            for (&index, &factor) in indices.iter().zip(factors) {
                frame[index] -= factor * value;
            }
        }
        up.clear();
        up.extend(self.update_rows.iter().map(|&row| frame[row]));
    }

    /// The pass down of a solve: from the values of its update's columns its parent left,
    /// the solution at its own columns, in `solution`, its subtree's.
    fn backward(&self, node: &Node, space: &mut Scratch<T>, solution: &mut [T]) {
        let Scratch { steps, down, .. } = space;
        for index in (0..down.len()).rev() {
            let value = down[index];
            let (indices, factors) = self.outer.column(index);
            for (&step, &factor) in indices.iter().zip(factors) {
                steps[step] -= factor * value;
            }
        }
        let offset = node.own - node.first;
        for step in (0..self.pivots.len()).rev() {
            let value = steps[step] / self.pivots[step];
            solution[offset + self.pivot_columns[step]] = value;
            let (indices, factors) = self.upper.column(step);
            for (&earlier, &factor) in indices.iter().zip(factors) {
                steps[earlier] -= factor * value;
            }
        }
    }

    /// The pass up of a transposed solve: from the right-hand side by column in `given`,
    /// each step's value, and the values of its update's columns for its parent.
    fn forward_transposed(&self, node: &Node, given: &[T], space: &mut Scratch<T>) {
        let own = node.size();
        let Scratch {
            frame, steps, up, ..
        } = space;
        frame.clear();
        frame.extend_from_slice(&given[node.own..node.end]);
        frame.resize(own + node.border.len(), T::ZERO);
        steps.clear();
        for step in 0..self.pivots.len() {
            let mut value = frame[self.pivot_columns[step]];
            let (indices, factors) = self.upper.column(step);
            for (&earlier, &factor) in indices.iter().zip(factors) {
                value -= factor * steps[earlier];
            }
            steps.push(value / self.pivots[step]);
        }
        up.clear();
        for index in 0..self.update_rows.len() {
            let mut value = frame[self.update_column(index, own)];
            let (indices, factors) = self.outer.column(index);
            for (&step, &factor) in indices.iter().zip(factors) {
                value -= factor * steps[step];
            }
            up.push(value);
        }
    }

    /// The pass down of a transposed solve: from the values of its update's rows its parent
    /// left, the solution at its own rows, in `solution`, its subtree's.
    fn backward_transposed(&self, node: &Node, space: &mut Scratch<T>, solution: &mut [T]) {
        let Scratch {
            frame, steps, down, ..
        } = space;
        for (&row, &value) in self.update_rows.iter().zip(down.iter()) {
            frame[row] = value;
        }
        let offset = node.own - node.first;
        for step in (0..self.pivots.len()).rev() {
            let mut value = steps[step];
            let (indices, factors) = self.lower.column(step);
            for (&index, &factor) in indices.iter().zip(factors) {
                value -= factor * frame[index];
            }
            let row = self.pivot_rows[step];
            frame[row] = value;
            solution[offset + row] = value;
        }
    }
}

/// Has `column` of a frame, dense, take in the `steps` whose columns `earlier` holds, in
/// order, the columns of `column`'s length: each subtracts its multipliers below its
/// diagonal times the column's entry in its row.
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

/// What the elimination of a front did with one of its columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operation {
    /// It pivoted on the row at this slot, swapped into the step's.
    Pivot(usize),
    /// It delayed the column, swapped with the last not yet eliminated, at this slot.
    Delay(usize),
}

/// The factors of a separator: its frame, dense, its own rows and those its children
/// delayed then its border's by its own columns and those its children delayed then its
/// border's, eliminated in place.
#[derive(Clone)]
struct Front<T> {
    /// Whether it keeps the pivots of its last factorization.
    complete: bool,
    /// The positions of the frame's rows and columns, slot by slot, as eliminated.
    rows: Vec<usize>,
    columns: Vec<usize>,
    /// The slot, as eliminated, of each row and each column of the frame as assembled.
    row_slots: Vec<usize>,
    column_slots: Vec<usize>,
    /// The number of rows, and of columns, it may pivot: its own and those delayed to it.
    fully: usize,
    /// The number of steps it eliminated.
    steps: usize,
    /// The frame, dense by columns: `L` below its unit diagonal, `U` on and above it, and
    /// the update after the steps.
    frame: Vec<T>,
    /// What it did with each column, in order.
    operations: Vec<Operation>,
    /// How many steps each column of the frame has taken in, by slot.
    applied: Vec<usize>,
    /// Scratch space: the slot as assembled of each slot of the frame, the slots of a
    /// child's update in the frame.
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
    /// The slot of `node`'s frame, as assembled, of each row of the update of `child`,
    /// its `which`th child, whose delayed rows come at `offset`, after `delayed_in` rows
    /// delayed into the frame in all; and alike of each column.
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

    /// Assembles the frame of `node` from the matrix of `values` and the updates of its
    /// children, whose parts `below` holds from `base` on.
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
        self.frame.clear();
        self.frame.try_reserve(size * size)?;
        self.frame.resize(size * size, T::ZERO);
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

    /// Factors `node`'s frame on up to `threads` threads, searching for every pivot, its
    /// children's parts `below` from their subtree's earliest node. A column delayed at the
    /// root, `root`, has no nonzero pivot left.
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

    /// Factors `node`'s frame on up to `threads` threads with the last factorization's
    /// operations. Returns whether every pivot reaches the threshold.
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

    /// Eliminates the frame's fully summed columns on up to `threads` threads: choosing
    /// each pivot, and recording what it did with each column, or, `replay`, doing again
    /// what the last factorization did. Returns the number of steps, and whether every
    /// pivot replayed reaches the threshold, the steps stopping at the first that does not.
    ///
    /// A column takes in the steps it has not yet taken, in their order, just before it is
    /// eliminated; and every column after the last step takes them in after each panel of
    /// steps, the fully summed columns' share of a thread, the panels' columns spread over
    /// the threads. A row swapped by a later step is swapped in every column at once, which
    /// moves each entry with its multiplier, so every entry undergoes the same operations
    /// in the same order whatever the panels: those of a step that eliminated every column
    /// after it at once.
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

    /// What to do with the frame's column `step`, all its earlier steps taken in, the
    /// columns from `left` on delayed: pivot on the row the ordering matched with it, if it
    /// reaches the threshold, else on the row of the largest magnitude that may be pivoted,
    /// if it does, and else delay it.
    fn choose(&self, step: usize, left: usize) -> Operation {
        let size = self.rows.len();
        let column = &self.frame[step * size..(step + 1) * size];
        let largest = column[step..]
            .iter()
            .map(|value| value.magnitude())
            .fold(0.0, f64::max);
        let least = PIVOT_TOLERANCE * largest;
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

    /// Whether the pivot of `step` reaches the threshold: at least a tenth of the largest
    /// magnitude in its column from its diagonal down, and not zero.
    fn holds(&self, step: usize) -> bool {
        let size = self.rows.len();
        let column = &self.frame[step * size..(step + 1) * size];
        let largest = column[step..]
            .iter()
            .map(|value| value.magnitude())
            .fold(0.0, f64::max);
        let magnitude = column[step].magnitude();
        magnitude > 0.0 && magnitude >= PIVOT_TOLERANCE * largest
    }

    /// Divides the frame's column `step` below its diagonal by its pivot: `L`'s column.
    fn divide(&mut self, step: usize) {
        let size = self.rows.len();
        let column = &mut self.frame[step * size..(step + 1) * size];
        let pivot = column[step];
        for value in &mut column[step + 1..] {
            *value = *value / pivot;
        }
    }

    /// Has the frame's column `column` take in every step before `until` it has not.
    fn catch_up(&mut self, column: usize, until: usize) {
        let size = self.rows.len();
        let (earlier, rest) = self.frame.split_at_mut(column * size);
        take_in(earlier, &mut rest[..size], self.applied[column]..until);
        self.applied[column] = until;
    }

    /// Has every column of the frame from `step` on take in every step before it, the
    /// columns spread over up to `threads` threads.
    fn catch_up_after(&mut self, step: usize, threads: usize) {
        let size = self.rows.len();
        let (earlier, rest) = self.frame.split_at_mut(step * size);
        let applied = &mut self.applied[step..];
        let columns = applied.len();
        if columns == 0 {
            return;
        }
        let share = columns.div_ceil(threads.max(1));
        rest.par_chunks_mut(share * size)
            .zip(applied.par_chunks_mut(share))
            .for_each(|(columns, applied)| {
                for (column, taken) in columns.chunks_exact_mut(size).zip(applied) {
                    take_in(earlier, column, *taken..step);
                    *taken = step;
                }
            });
    }

    /// Records the slots, as eliminated, of the frame's rows and columns, after `steps`.
    fn finish(&mut self, steps: usize) {
        self.steps = steps;
        let size = self.rows.len();
        // The slot each row and column of the frame as assembled went to, by replaying the
        // swaps on the identity.
        for (slots, rows) in [(&mut self.row_slots, true), (&mut self.column_slots, false)] {
            self.origins.clear();
            self.origins.extend(0..size);
            let mut step = 0;
            for &operation in &self.operations {
                match operation {
                    Operation::Pivot(row) => {
                        if rows {
                            self.origins.swap(step, row);
                        }
                        step += 1;
                    }
                    Operation::Delay(other) => {
                        if !rows {
                            self.origins.swap(step, other);
                        }
                    }
                }
            }
            slots.clear();
            slots.resize(size, 0);
            for (slot, &assembled) in self.origins.iter().enumerate() {
                slots[assembled] = slot;
            }
        }
    }

    /// Swaps two rows of the frame, whole.
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

    /// Swaps two columns of the frame, whole, the first before the second.
    fn swap_columns(&mut self, one: usize, other: usize) {
        if one == other {
            return;
        }
        let size = self.rows.len();
        let (left, right) = self.frame.split_at_mut(other * size);
        left[one * size..(one + 1) * size].swap_with_slice(&mut right[..size]);
        self.columns.swap(one, other);
    }

    /// The pass up of a solve: from the right-hand side by row in `given` and its
    /// children's values, each step's value, and the values of its update's rows.
    fn forward(
        &self,
        node: &Node,
        given: &[T],
        space: &mut Scratch<T>,
        (parts, below, base): (&[Part<T>], &[Scratch<T>], usize),
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
        // Each step swapped its rows across the whole frame, so `L`'s rows are in the
        // order of the last: the right-hand side takes every swap first.
        let mut step = 0;
        for &operation in &self.operations {
            if let Operation::Pivot(row) = operation {
                frame.swap(step, row);
                step += 1;
            }
        }
        for step in 0..self.steps {
            let value = frame[step];
            let multipliers = &self.frame[step * size + step + 1..(step + 1) * size];
            for (target, &factor) in frame[step + 1..].iter_mut().zip(multipliers) {
                *target -= factor * value;
            }
        }
        up.clear();
        up.extend_from_slice(&frame[self.steps..]);
    }

    /// The pass down of a solve: from the values of its update's columns its parent left,
    /// the solution at its columns, in `solution`, its subtree's, and the values of its
    /// children's updates' columns.
    fn backward(
        &self,
        node: &Node,
        space: &mut Scratch<T>,
        (parts, below): (&[Part<T>], &mut [Scratch<T>]),
        solution: &mut [T],
    ) {
        let size = self.rows.len();
        let Scratch {
            frame, steps, down, ..
        } = space;
        steps.clear();
        steps.resize(self.steps, T::ZERO);
        steps.extend_from_slice(down);
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
        for (&position, &value) in self.columns.iter().zip(steps.iter()).take(self.steps) {
            solution[position - node.first] = value;
        }
        self.pass_down(node, (parts, below), steps, &self.column_slots);
    }

    /// Leaves each child the values of its update's rows or columns, from `values`, by slot
    /// of the frame as eliminated, whose slots as assembled went to `slots`.
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

    /// The pass up of a transposed solve: from the right-hand side by column in `given` and
    /// its children's values, each step's value, and the values of its update's columns.
    fn forward_transposed(
        &self,
        node: &Node,
        given: &[T],
        space: &mut Scratch<T>,
        (parts, below, base): (&[Part<T>], &[Scratch<T>], usize),
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
        let mut step = 0;
        for &operation in &self.operations {
            match operation {
                Operation::Pivot(_) => step += 1,
                Operation::Delay(other) => frame.swap(step, other),
            }
        }
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
        up.clear();
        up.extend_from_slice(&frame[self.steps..]);
    }

    /// The pass down of a transposed solve: from the values of its update's rows its parent
    /// left, the solution at its rows, in `solution`, its subtree's, and the values of its
    /// children's updates' rows.
    fn backward_transposed(
        &self,
        node: &Node,
        space: &mut Scratch<T>,
        (parts, below): (&[Part<T>], &mut [Scratch<T>]),
        solution: &mut [T],
    ) {
        let size = self.rows.len();
        let Scratch {
            frame, steps, down, ..
        } = space;
        steps.clear();
        steps.resize(self.steps, T::ZERO);
        steps.extend_from_slice(down);
        for step in (0..self.steps).rev() {
            let mut value = frame[step];
            let entries = &self.frame[step * size + step + 1..(step + 1) * size];
            for (&factor, &later) in entries.iter().zip(&steps[step + 1..]) {
                value -= factor * later;
            }
            steps[step] = value;
        }
        for (&position, &value) in self.rows.iter().zip(steps.iter()).take(self.steps) {
            solution[position - node.first] = value;
        }
        self.pass_down(node, (parts, below), steps, &self.row_slots);
    }
}
