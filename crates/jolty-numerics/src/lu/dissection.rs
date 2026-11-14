//! The nested dissection of a large diagonal block into interiors and the separators that
//! border them, after George and Liu ("An automatic nested dissection algorithm for
//! irregular finite element problems", SIAM Journal on Numerical Analysis 15(5), 1978).
//!
//! A part of the block's graph, its pattern plus its transpose, splits at the middle level
//! of a level structure rooted at a pseudo-peripheral vertex (George and Liu, "An
//! implementation of a pseudoperipheral node finder", ACM Transactions on Mathematical
//! Software 5(3), 1979): the vertices of that level next to the level beyond it separate
//! the vertices before them from those after. Each component the separator leaves is a
//! part again, until a part holds no more than the interiors may, or has too few levels to
//! split. The separators and the interiors make a tree, each separator the parent of the
//! parts it separates, laid out in postorder: every subtree's positions are consecutive,
//! a node's own after its children's, so that no entry joins two subtrees of one node and
//! each subtree factors on its own.

use super::ordering::Entries;

/// No index: a vertex in no level yet, a row in no border.
const NONE: usize = usize::MAX;

/// A symmetric graph on the vertices of a block, by adjacency lists.
pub(super) struct Graph {
    starts: Vec<usize>,
    neighbours: Vec<usize>,
}

impl Graph {
    /// The graph whose vertex `v` has an edge to each of `lists[v]`, made symmetric, with no
    /// edge from a vertex to itself.
    pub(super) fn symmetric(lists: &[Vec<usize>]) -> Self {
        let size = lists.len();
        let mut all: Vec<Vec<usize>> = lists.to_vec();
        for (vertex, list) in lists.iter().enumerate() {
            for &other in list {
                all[other].push(vertex);
            }
        }
        let mut starts = Vec::with_capacity(size + 1);
        let mut neighbours = Vec::new();
        starts.push(0);
        for (vertex, list) in all.iter_mut().enumerate() {
            list.sort_unstable();
            list.dedup();
            neighbours.extend(list.iter().copied().filter(|&other| other != vertex));
            starts.push(neighbours.len());
        }
        Self { starts, neighbours }
    }

    /// The vertices joined to `vertex`.
    fn of(&self, vertex: usize) -> &[usize] {
        &self.neighbours[self.starts[vertex]..self.starts[vertex + 1]]
    }

    /// The number of vertices.
    fn size(&self) -> usize {
        self.starts.len() - 1
    }
}

/// A node of the tree a dissection makes, as found: its own vertices, and its children.
#[derive(Debug, Clone)]
pub(super) struct Found {
    /// Its vertices: an interior's, or a separator's, in the order of its level.
    pub(super) vertices: Vec<usize>,
    /// The indices of its children among the nodes found, in order; none for an interior.
    pub(super) children: Vec<usize>,
}

/// Dissects `graph`, a connected graph, until each interior holds at most `most` vertices
/// or cannot split: the nodes of the tree, in postorder, the last its root.
pub(super) fn dissect(graph: &Graph, most: usize) -> Vec<Found> {
    let mut dissector = Dissector {
        graph,
        part: vec![0; graph.size()],
        parts: 1,
        level: vec![NONE; graph.size()],
        order: Vec::new(),
        found: Vec::new(),
    };
    dissector.dissect((0..graph.size()).collect(), most);
    dissector.found
}

/// The state of a dissection: the part each vertex is in, and the nodes found.
struct Dissector<'a> {
    graph: &'a Graph,
    /// The part each vertex is in: only vertices of one part are searched together.
    part: Vec<usize>,
    /// The number of parts made.
    parts: usize,
    /// Each vertex's level in the last level structure that reached it, or `NONE`.
    level: Vec<usize>,
    /// The vertices of the last level structure, level by level.
    order: Vec<usize>,
    found: Vec<Found>,
}

impl Dissector<'_> {
    /// Dissects the connected part whose vertices are `vertices`, all in one part, and
    /// returns the index of the node it becomes.
    fn dissect(&mut self, vertices: Vec<usize>, most: usize) -> usize {
        let Some(first) = vertices.first().copied() else {
            return self.leaf(vertices);
        };
        if vertices.len() <= most {
            return self.leaf(vertices);
        }
        // The level structure last built is the pseudo-peripheral vertex's.
        let starts = self.pseudo_peripheral(first, vertices.len());
        // Fewer than three levels leave no level with one on either side.
        let levels = starts.len() - 1;
        if levels < 3 {
            return self.leaf(vertices);
        }
        // George and Liu's middle level, numbering levels from one: (levels + 2) / 2.
        let middle = levels / 2;
        let separator: Vec<usize> = self.order[starts[middle]..starts[middle + 1]]
            .iter()
            .copied()
            .filter(|&vertex| {
                self.graph
                    .of(vertex)
                    .iter()
                    .any(|&other| self.level[other] == middle + 1)
            })
            .collect();
        let part = self.part[first];
        let separated = self.parts;
        self.parts += 1;
        for &vertex in &separator {
            self.part[vertex] = separated;
        }
        // Each component of the rest, in the order of its first vertex in the levels.
        let levelled = self.order.clone();
        let mut components = Vec::new();
        for start in levelled {
            if self.part[start] != part {
                continue;
            }
            let component = self.parts;
            self.parts += 1;
            self.part[start] = component;
            let mut members = vec![start];
            let mut next = 0;
            while let Some(&vertex) = members.get(next) {
                next += 1;
                for &other in self.graph.of(vertex) {
                    if self.part[other] == part {
                        self.part[other] = component;
                        members.push(other);
                    }
                }
            }
            members.sort_unstable();
            components.push(members);
        }
        let children = components
            .into_iter()
            .map(|component| self.dissect(component, most))
            .collect();
        self.found.push(Found {
            vertices: separator,
            children,
        });
        self.found.len() - 1
    }

    /// Records the interior whose vertices are `vertices`.
    fn leaf(&mut self, vertices: Vec<usize>) -> usize {
        self.found.push(Found {
            vertices,
            children: Vec::new(),
        });
        self.found.len() - 1
    }

    /// Builds the level structure of a pseudo-peripheral vertex of the part of `start`,
    /// which holds `size` vertices, and returns where its levels start: from `start`, the
    /// vertex of least degree in the last level of the latest structure, while its own
    /// structure is deeper.
    fn pseudo_peripheral(&mut self, start: usize, size: usize) -> Vec<usize> {
        let mut root = start;
        let mut starts = self.levels(root);
        loop {
            let levels = starts.len() - 1;
            if levels == 1 || levels == size {
                return starts;
            }
            let part = self.part[root];
            let degree = |vertex: usize| {
                self.graph
                    .of(vertex)
                    .iter()
                    .filter(|&&other| self.part[other] == part)
                    .count()
            };
            let last = &self.order[starts[levels - 1]..starts[levels]];
            let mut candidate = last[0];
            let mut fewest = degree(candidate);
            for &vertex in &last[1..] {
                let count = degree(vertex);
                if count < fewest {
                    (candidate, fewest) = (vertex, count);
                }
            }
            let deeper = self.levels(candidate);
            if deeper.len() <= starts.len() {
                // Build `root`'s again, for the caller to read.
                return self.levels(root);
            }
            (root, starts) = (candidate, deeper);
        }
    }

    /// Builds the level structure of `root`'s part rooted at `root` into `order` and
    /// `level`, and returns where each level starts in `order`, then its length.
    fn levels(&mut self, root: usize) -> Vec<usize> {
        for &vertex in &self.order {
            self.level[vertex] = NONE;
        }
        self.order.clear();
        let part = self.part[root];
        self.order.push(root);
        self.level[root] = 0;
        let mut starts = vec![0];
        let mut next = 0;
        while next < self.order.len() {
            let end = self.order.len();
            let depth = starts.len();
            starts.push(end);
            while next < end {
                let vertex = self.order[next];
                next += 1;
                for &other in self.graph.of(vertex) {
                    if self.part[other] == part && self.level[other] == NONE {
                        self.level[other] = depth;
                        self.order.push(other);
                    }
                }
            }
        }
        starts
    }
}

/// A node of a block's dissection: an interior, or a separator and the subtrees it
/// separates, at consecutive positions of the ordered matrix.
#[derive(Debug)]
pub(super) struct Node {
    /// The index of the earliest node of its subtree, which runs from it to this one.
    pub(super) earliest: usize,
    /// The first position of its subtree.
    pub(super) first: usize,
    /// The first of its own positions, which run to `end`, after its children's.
    pub(super) own: usize,
    /// The position after its own.
    pub(super) end: usize,
    /// Its children's indices among the nodes, in order; none for an interior.
    pub(super) children: Vec<usize>,
    /// The positions of its ancestors joined to its subtree, in increasing order: the rows
    /// and columns beyond its own that its factorization updates.
    pub(super) border: Vec<usize>,
    /// The entries of each of its own columns that it assembles, each its row's index in
    /// the node's frame and the slot of its value: an own row's index among its own, or a
    /// border row's index in the border after them.
    pub(super) columns: Entries,
    /// The entries in its own rows of each column of its border, each its row's index
    /// among its own and the slot of its value.
    pub(super) border_columns: Entries,
    /// For each child, the index in this node's frame of each position of the child's
    /// border: among its own positions, or after them, in its border.
    pub(super) child_borders: Vec<Vec<usize>>,
}

impl Node {
    /// The number of its own positions.
    pub(super) const fn size(&self) -> usize {
        self.end - self.own
    }

    /// Whether it is an interior, with no children.
    pub(super) fn is_interior(&self) -> bool {
        self.children.is_empty()
    }

    /// The index in its frame of `position`, one of its own or of its border.
    fn index(&self, position: usize) -> Option<usize> {
        if (self.own..self.end).contains(&position) {
            return Some(position - self.own);
        }
        self.border
            .binary_search(&position)
            .ok()
            .map(|index| self.size() + index)
    }
}

/// The nested dissection of one diagonal block of an ordered matrix.
#[derive(Debug)]
pub(super) struct Dissection {
    /// The nodes in postorder, the root last.
    pub(super) nodes: Vec<Node>,
}

impl Dissection {
    /// The dissection of the block from `start` to `end` whose nodes, in postorder, own the
    /// numbers of positions in `sizes` with the children in `children`, its positions laid
    /// out in that order, and the entries of whose columns, within the block, are
    /// `within(position)`: each its row's position and its value's slot.
    pub(super) fn new<'a>(
        (start, end): (usize, usize),
        found: &[Found],
        within: impl Fn(usize) -> &'a [(usize, usize)],
    ) -> Self {
        let size = end - start;
        // Each position's neighbours within the block, both ways.
        let mut lists = vec![Vec::new(); size];
        for column in start..end {
            for &(row, _) in within(column) {
                lists[column - start].push(row - start);
            }
        }
        let graph = Graph::symmetric(&lists);
        let mut nodes: Vec<Node> = Vec::with_capacity(found.len());
        let mut next = start;
        for (index, each) in found.iter().enumerate() {
            let earliest = each
                .children
                .first()
                .map_or(index, |&child| nodes[child].earliest);
            let first = each
                .children
                .first()
                .map_or(next, |&child| nodes[child].first);
            let own = next;
            next += each.vertices.len();
            let mut border: Vec<usize> = each
                .children
                .iter()
                .flat_map(|&child| nodes[child].border.iter().copied())
                .chain((own..next).flat_map(|position| {
                    graph
                        .of(position - start)
                        .iter()
                        .map(|&other| other + start)
                }))
                .filter(|&position| position >= next)
                .collect();
            border.sort_unstable();
            border.dedup();
            nodes.push(Node {
                earliest,
                first,
                own,
                end: next,
                children: each.children.clone(),
                border,
                columns: Entries::empty(0),
                border_columns: Entries::empty(0),
                child_borders: Vec::new(),
            });
        }
        debug_assert_eq!(next, end, "the nodes own every position of the block");
        for index in 0..nodes.len() {
            let node = &nodes[index];
            let mut columns = Entries::empty(0);
            for column in node.own..node.end {
                columns.entries.extend(
                    within(column)
                        .iter()
                        .filter_map(|&(row, slot)| node.index(row).map(|index| (index, slot))),
                );
                columns.starts.push(columns.entries.len());
            }
            let mut border_columns = Entries::empty(0);
            for &column in &node.border {
                border_columns.entries.extend(
                    within(column)
                        .iter()
                        .filter(|&&(row, _)| (node.own..node.end).contains(&row))
                        .map(|&(row, slot)| (row - node.own, slot)),
                );
                border_columns.starts.push(border_columns.entries.len());
            }
            let child_borders = node
                .children
                .iter()
                .map(|&child| in_frame(node, &nodes[child].border))
                .collect();
            let node = &mut nodes[index];
            node.columns = columns;
            node.border_columns = border_columns;
            node.child_borders = child_borders;
        }
        Self { nodes }
    }

    /// The root of the tree.
    pub(super) fn root(&self) -> usize {
        self.nodes.len() - 1
    }
}

/// The index in `node`'s frame of each position of the border of a child of it.
#[expect(
    clippy::expect_used,
    reason = "a child's border is its ancestors' positions joined to it, and no edge joins two \
              subtrees of a node: those of the child's parent are its own, and the rest are \
              joined to the parent's subtree through the child, so in its border"
)]
fn in_frame(node: &Node, border: &[usize]) -> Vec<usize> {
    border
        .iter()
        .map(|&position| node.index(position).expect("in the parent's frame"))
        .collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// The graph of a `width` × `height` grid, each vertex joined to its neighbours.
    fn grid(width: usize, height: usize) -> Graph {
        let lists: Vec<Vec<usize>> = (0..width * height)
            .map(|vertex| {
                let (x, y) = (vertex % width, vertex / width);
                let mut list = Vec::new();
                if x + 1 < width {
                    list.push(vertex + 1);
                }
                if y + 1 < height {
                    list.push(vertex + width);
                }
                list
            })
            .collect();
        Graph::symmetric(&lists)
    }

    /// Checks that `found` is a tree in postorder over every vertex of `graph` once, whose
    /// separators each leave no edge between two of their subtrees.
    fn check(graph: &Graph, found: &[Found]) -> Result<(), TestCaseError> {
        let mut node_of = vec![NONE; graph.size()];
        for (index, node) in found.iter().enumerate() {
            for &vertex in &node.vertices {
                prop_assert_eq!(node_of[vertex], NONE, "a vertex in two nodes");
                node_of[vertex] = index;
            }
            prop_assert!(
                node.children.iter().all(|&child| child < index),
                "postorder"
            );
            if !node.children.is_empty() {
                prop_assert!(node.children.len() >= 2, "a separator separates");
            }
        }
        prop_assert!(
            node_of.iter().all(|&node| node != NONE),
            "every vertex placed"
        );
        let mut parent = vec![NONE; found.len()];
        for (index, node) in found.iter().enumerate() {
            for &child in &node.children {
                parent[child] = index;
            }
        }
        // An edge joins a node to itself, an ancestor or a descendant.
        let is_ancestor = |mut node: usize, of: usize| {
            while node != NONE {
                if node == of {
                    return true;
                }
                node = parent[node];
            }
            false
        };
        for vertex in 0..graph.size() {
            for &other in graph.of(vertex) {
                let (a, b) = (node_of[vertex], node_of[other]);
                prop_assert!(
                    is_ancestor(a, b) || is_ancestor(b, a),
                    "an edge across subtrees"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn splits_a_grid_at_its_middle_diagonal() {
        let graph = grid(20, 20);
        let found = dissect(&graph, 50);
        check(&graph, &found).unwrap();
        // From a corner, the levels are the grid's anti-diagonals: the middle one, of 20
        // vertices, separates it into two triangles.
        let root = found.last().unwrap();
        assert_eq!(root.vertices.len(), 20);
        assert_eq!(root.children.len(), 2);
        assert!(
            found
                .iter()
                .filter(|node| node.children.is_empty())
                .all(|leaf| leaf.vertices.len() <= 50)
        );
    }

    #[test]
    fn splits_a_path_at_its_middle_vertex() {
        let lists: Vec<Vec<usize>> = (0..9)
            .map(|vertex| vec![(vertex + 1) % 10])
            .take(9)
            .collect();
        let mut lists = lists;
        lists.push(Vec::new());
        let graph = Graph::symmetric(&lists);
        let found = dissect(&graph, 2);
        check(&graph, &found).unwrap();
        assert_eq!(found.last().unwrap().vertices, [5]);
    }

    #[test]
    fn keeps_a_small_or_shallow_part_whole() {
        let graph = grid(3, 3);
        assert_eq!(dissect(&graph, 9).len(), 1);
        // A complete graph has two levels from any vertex: it cannot split.
        let complete: Vec<Vec<usize>> = (0..6).map(|vertex| (vertex + 1..6).collect()).collect();
        assert_eq!(dissect(&Graph::symmetric(&complete), 1).len(), 1);
        // A star has three from a leaf: its centre separates every leaf from the others.
        let star: Vec<Vec<usize>> = (0..8)
            .map(|vertex| {
                if vertex == 0 {
                    (1..8).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let found = dissect(&Graph::symmetric(&star), 1);
        assert_eq!(found.len(), 8);
        assert_eq!(found.last().unwrap().vertices, [0]);
    }

    proptest! {
        /// Random connected graphs dissect into trees whose separators separate.
        #[test]
        fn dissects_random_connected_graphs(
            size in 1usize..60,
            extra in proptest::collection::vec((0usize..60, 0usize..60), 0..120),
            parents in proptest::collection::vec(any::<proptest::sample::Index>(), 60),
            most in 1usize..20,
        ) {
            let mut lists = vec![Vec::new(); size];
            for vertex in 1..size {
                lists[vertex].push(parents[vertex].index(vertex));
            }
            for (a, b) in extra {
                if a < size && b < size {
                    lists[a].push(b);
                }
            }
            let graph = Graph::symmetric(&lists);
            let found = dissect(&graph, most);
            check(&graph, &found)?;
        }
    }
}
