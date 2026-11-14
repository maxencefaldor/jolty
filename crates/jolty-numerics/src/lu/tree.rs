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
//! The subtrees of a node's children share no entry, so they factor at once, each on a
//! thread of its own when rayon has one free; and every number is computed by the same
//! operations in the same order however they are spread, so the factors and the solutions
//! are the same bits on any number of threads.

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
    Front(Front<T>),
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
                    Part::Front(Front::default())
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

    /// Factors the block of `values`, the matrix's values by slot: each node keeps its
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
    ) -> Result<bool, LinearError> {
        let nodes = &dissection.nodes;
        let root = dissection.root();
        upward(
            nodes,
            root,
            &mut self.parts,
            &|index, part, below, searched| {
                let node = &nodes[index];
                let base = node.earliest;
                let searched = searched
                    || !match part {
                        Part::Interior(interior) => {
                            interior.complete && interior.refactor(node, values)
                        }
                        Part::Front(front) => {
                            front.complete && front.refactor(node, values, below, base)?
                        }
                    };
                if searched {
                    match part {
                        Part::Interior(interior) => interior.search(node, values),
                        Part::Front(front) => {
                            front.search(node, values, below, base, index == root)
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

    /// Solves the block's system with the scratch space `scratch`: from the right-hand side
    /// by row of the ordered matrix in `given`, the solution by column in `solution`, the
    /// block's positions of the ordered matrix; or, `transposed`, from the right-hand side
    /// by column, the solution by row.
    pub(super) fn solve(
        &self,
        dissection: &Dissection,
        given: &[T],
        solution: &mut [T],
        scratch: &mut Vec<Scratch<T>>,
        transposed: bool,
    ) {
        let nodes = &dissection.nodes;
        scratch.resize_with(nodes.len(), Scratch::default);
        let root = dissection.root();
        let Ok(_) = upward(nodes, root, scratch, &|index, space, below, _| {
            let node = &nodes[index];
            let below = (self.parts.as_slice(), below, node.earliest);
            match (&self.parts[index], transposed) {
                (Part::Interior(interior), false) => interior.forward(node, given, space),
                (Part::Interior(interior), true) => interior.forward_transposed(node, given, space),
                (Part::Front(front), false) => front.forward(node, given, space, below),
                (Part::Front(front), true) => front.forward_transposed(node, given, space, below),
            }
            Ok::<bool, std::convert::Infallible>(false)
        });
        downward(
            nodes,
            root,
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

/// Runs `visit` on each node of the subtree of `index`, whose nodes' states are `states`,
/// each after its children: with its state, its descendants', from its subtree's first
/// node, and whether any visit of them returned `true`. The children's subtrees run at once.
/// Returns whether any visit returned `true`, or the error of the first that failed, in the
/// nodes' order.
fn upward<S: Send, E: Send>(
    nodes: &[Node],
    index: usize,
    states: &mut [S],
    visit: &(impl Fn(usize, &mut S, &[S], bool) -> Result<bool, E> + Sync),
) -> Result<bool, E> {
    let node = &nodes[index];
    let Some((state, below)) = states.split_last_mut() else {
        return Ok(false);
    };
    let mut parts = Vec::with_capacity(node.children.len());
    let mut rest = &mut *below;
    for &child in &node.children {
        let (subtree, after) = rest.split_at_mut(child + 1 - nodes[child].earliest);
        parts.push((child, subtree));
        rest = after;
    }
    let results: Vec<Result<bool, E>> = parts
        .into_par_iter()
        .map(|(child, subtree)| upward(nodes, child, subtree, visit))
        .collect();
    let mut any = false;
    for result in results {
        any |= result?;
    }
    Ok(visit(index, state, below, any)? || any)
}

/// Runs `visit` on each node of the subtree of `index`, whose nodes' states are `states`,
/// each before its children: with its state, its descendants' to write to, and the part of
/// `solution` at its subtree's positions. The children's subtrees run at once.
fn downward<S: Send, T: Send>(
    nodes: &[Node],
    index: usize,
    states: &mut [S],
    solution: &mut [T],
    visit: &(impl Fn(usize, &mut S, &mut [S], &mut [T]) + Sync),
) {
    let node = &nodes[index];
    let Some((state, below)) = states.split_last_mut() else {
        return;
    };
    visit(index, state, below, solution);
    let mut parts = Vec::with_capacity(node.children.len());
    let (mut rest, mut written) = (below, solution);
    for &child in &node.children {
        let child_node = &nodes[child];
        let (subtree, after) = rest.split_at_mut(child + 1 - child_node.earliest);
        let (part, beyond) = written.split_at_mut(child_node.end - child_node.first);
        parts.push((child, subtree, part));
        (rest, written) = (after, beyond);
    }
    parts
        .into_par_iter()
        .for_each(|(child, subtree, part)| downward(nodes, child, subtree, part, visit));
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
        let mut slots = Vec::new();
        for (which, child) in children().enumerate() {
            let update = child.update();
            slots.clear();
            slots.extend(Self::child_slots(node, (child, which), offset, delayed_in));
            for (column, &into) in slots.iter().enumerate() {
                let target = &mut self.frame[into * size..(into + 1) * size];
                for (row, &at) in slots.iter().enumerate() {
                    target[at] += update.get(row, column);
                }
            }
            offset += update.delayed;
        }
        Ok(())
    }

    /// Factors `node`'s frame, searching for every pivot. A column delayed at the root,
    /// `root`, has no nonzero pivot left.
    fn search(
        &mut self,
        node: &Node,
        values: &[T],
        below: &[Part<T>],
        base: usize,
        root: bool,
    ) -> Result<(), LinearError> {
        self.complete = false;
        self.assemble(node, values, below, base)?;
        self.operations.clear();
        let size = self.rows.len();
        let (mut step, mut left) = (0, self.fully);
        while step < left {
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
            let operation = if diagonal != NONE
                && column[diagonal].magnitude() > 0.0
                && column[diagonal].magnitude() >= least
            {
                Operation::Pivot(diagonal)
            } else if chosen != NONE && best >= least {
                Operation::Pivot(chosen)
            } else {
                Operation::Delay(left - 1)
            };
            self.operations.push(operation);
            match operation {
                Operation::Pivot(row) => {
                    self.swap_rows(step, row);
                    self.eliminate(step);
                    step += 1;
                }
                Operation::Delay(other) => {
                    left -= 1;
                    self.swap_columns(step, other);
                }
            }
        }
        self.finish(step);
        if root && step < self.fully {
            return Err(LinearError::Singular {
                columns: vec![self.columns[step]],
            });
        }
        self.complete = true;
        Ok(())
    }

    /// Factors `node`'s frame with the last factorization's operations. Returns whether
    /// every pivot reaches the threshold.
    fn refactor(
        &mut self,
        node: &Node,
        values: &[T],
        below: &[Part<T>],
        base: usize,
    ) -> Result<bool, LinearError> {
        self.assemble(node, values, below, base)?;
        let size = self.rows.len();
        let mut step = 0;
        for index in 0..self.operations.len() {
            match self.operations[index] {
                Operation::Pivot(row) => {
                    self.swap_rows(step, row);
                    let column = &self.frame[step * size..(step + 1) * size];
                    let largest = column[step..]
                        .iter()
                        .map(|value| value.magnitude())
                        .fold(0.0, f64::max);
                    let magnitude = column[step].magnitude();
                    if !(magnitude > 0.0 && magnitude >= PIVOT_TOLERANCE * largest) {
                        self.complete = false;
                        return Ok(false);
                    }
                    self.eliminate(step);
                    step += 1;
                }
                Operation::Delay(other) => self.swap_columns(step, other),
            }
        }
        self.finish(step);
        Ok(true)
    }

    /// Records the slots, as eliminated, of the frame's rows and columns, after `steps`.
    fn finish(&mut self, steps: usize) {
        self.steps = steps;
        let size = self.rows.len();
        // The slots each row and column of the frame as assembled went to, by replaying the
        // swaps on the identity.
        let mut rows: Vec<usize> = (0..size).collect();
        let mut columns = rows.clone();
        let mut step = 0;
        for &operation in &self.operations {
            match operation {
                Operation::Pivot(row) => {
                    rows.swap(step, row);
                    step += 1;
                }
                Operation::Delay(other) => columns.swap(step, other),
            }
        }
        self.row_slots.clear();
        self.row_slots.resize(size, 0);
        self.column_slots.clear();
        self.column_slots.resize(size, 0);
        for (slot, &assembled) in rows.iter().enumerate() {
            self.row_slots[assembled] = slot;
        }
        for (slot, &assembled) in columns.iter().enumerate() {
            self.column_slots[assembled] = slot;
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

    /// Eliminates the frame's column `step` on its diagonal, from every row and column
    /// after it.
    fn eliminate(&mut self, step: usize) {
        let size = self.rows.len();
        let (left, right) = self.frame.split_at_mut((step + 1) * size);
        let pivot_column = &mut left[step * size..];
        let pivot = pivot_column[step];
        for value in &mut pivot_column[step + 1..] {
            *value = *value / pivot;
        }
        let multipliers = &pivot_column[step + 1..];
        for column in right.chunks_exact_mut(size) {
            let multiplier = column[step];
            if multiplier != T::ZERO {
                for (value, &factor) in column[step + 1..].iter_mut().zip(multipliers) {
                    *value -= factor * multiplier;
                }
            }
        }
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
