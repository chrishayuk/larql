//! Draft-tree data structure and attention-mask construction.
//!
//! A `DraftTree` represents the speculative candidates produced by
//! a [`Drafter`](super::Drafter) at one decode step: a root (the
//! token whose hidden state was used to draft) plus a fan-out of
//! candidate continuations.
//!
//! Phase 3's `cuda::attn_tree::tree_decode_attention` consumes the
//! flattened token list and the per-q-token attention mask produced
//! here. The tree mask is the load-bearing correctness piece for
//! tree verification: each query token at tree position `q` may
//! attend to the cache (causal) **plus** the tree positions on its
//! own root-to-leaf path (its ancestors), and nothing else.

use super::{DraftToken, TokenId};

/// One node in a draft tree. `parent == None` for the root.
/// `depth` is the distance from the root (root = 0).
#[derive(Clone, Debug)]
pub struct TreeNode {
    pub token: DraftToken,
    pub parent: Option<usize>,
    pub depth: usize,
}

/// A flattened draft tree. Nodes are stored in BFS order so:
///
/// - Index 0 is always the root.
/// - Children of a node have higher indices than the node.
/// - The mask construction can iterate the array in order and
///   propagate per-node ancestor sets without random access.
#[derive(Clone, Debug)]
pub struct DraftTree {
    nodes: Vec<TreeNode>,
}

impl DraftTree {
    /// Construct a tree with just a root token.
    pub fn from_root(root: DraftToken) -> Self {
        Self {
            nodes: vec![TreeNode {
                token: root,
                parent: None,
                depth: 0,
            }],
        }
    }

    /// Append a child of `parent_idx`. Returns the new node index.
    pub fn add_child(&mut self, parent_idx: usize, token: DraftToken) -> usize {
        let parent_depth = self.nodes[parent_idx].depth;
        self.nodes.push(TreeNode {
            token,
            parent: Some(parent_idx),
            depth: parent_depth + 1,
        });
        self.nodes.len() - 1
    }

    /// Build a balanced K-ary tree of `depth` levels with `branches`
    /// fanout per level, using a constant `proposer` to fill each
    /// slot. Convenience for tests and the depth=2/branches=2 default.
    pub fn balanced(depth: usize, branches: usize, mut proposer: impl FnMut(usize) -> DraftToken) -> Self {
        let root = proposer(0);
        let mut tree = Self::from_root(root);
        let mut frontier = vec![0usize];
        let mut counter = 1usize;
        for _ in 0..depth {
            let mut next = Vec::with_capacity(frontier.len() * branches);
            for parent in frontier.drain(..) {
                for _ in 0..branches {
                    let tok = proposer(counter);
                    counter += 1;
                    next.push(tree.add_child(parent, tok));
                }
            }
            frontier = next;
        }
        tree
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    pub fn root(&self) -> &TreeNode {
        &self.nodes[0]
    }

    /// Maximum depth across all nodes (root depth = 0). For a
    /// balanced(d, b) tree this is `d`.
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Walk from `node_idx` up to the root, yielding ancestor
    /// indices. The walked node itself is included as the first
    /// element.
    pub fn ancestors(&self, node_idx: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = Some(node_idx);
        while let Some(i) = cur {
            out.push(i);
            cur = self.nodes[i].parent;
        }
        out
    }

    /// All root-to-leaf paths through the tree. Each path is a
    /// sequence of node indices starting from the root. A leaf is
    /// any node with no children.
    pub fn root_to_leaf_paths(&self) -> Vec<Vec<usize>> {
        // Collect children-of map first.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            if let Some(p) = n.parent {
                children[p].push(i);
            }
        }
        let mut out = Vec::new();
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, vec![0])];
        while let Some((node, path)) = stack.pop() {
            if children[node].is_empty() {
                out.push(path);
                continue;
            }
            for &c in &children[node] {
                let mut p = path.clone();
                p.push(c);
                stack.push((c, p));
            }
        }
        out
    }

    pub fn tokens(&self) -> Vec<TokenId> {
        self.nodes.iter().map(|n| n.token.id).collect()
    }
}

/// Per-q-token attention mask for a draft tree, packed as a bitmask
/// over `kv_len + tree_len` positions.
///
/// Layout: row `q` (0 ≤ q < tree_len) is `((kv_len + tree_len) +
/// 31) / 32` u32 words. Bit `j` of row `q` is set when q-token `q`
/// is allowed to attend to absolute position `j`. Positions
/// `[0, kv_len)` are KV-cache positions (causal — every q sees them
/// all). Positions `[kv_len, kv_len + tree_len)` are the tree's own
/// query tokens; q-token `q` sees only its ancestors (including
/// itself) within the tree.
///
/// This is the layout the GPU `tree_decode_attention` kernel
/// (phase 3 task 3.1) consumes directly.
#[derive(Clone, Debug)]
pub struct TreeAttentionMask {
    pub kv_len: usize,
    pub tree_len: usize,
    pub words_per_row: usize,
    pub bits: Vec<u32>,
}

impl TreeAttentionMask {
    pub fn build(tree: &DraftTree, kv_len: usize) -> Self {
        let tree_len = tree.len();
        let total_cols = kv_len + tree_len;
        let words_per_row = total_cols.div_ceil(32);
        let mut bits = vec![0u32; tree_len * words_per_row];

        for q in 0..tree_len {
            let row_off = q * words_per_row;
            // Causal portion: every q sees all of KV cache.
            for j in 0..kv_len {
                let w = j / 32;
                let b = j % 32;
                bits[row_off + w] |= 1u32 << b;
            }
            // Tree portion: q sees its own ancestors (including
            // itself). Self-attention is included so the kernel
            // can reuse the same loop for every position.
            for ancestor in tree.ancestors(q) {
                let j = kv_len + ancestor;
                let w = j / 32;
                let b = j % 32;
                bits[row_off + w] |= 1u32 << b;
            }
        }

        Self {
            kv_len,
            tree_len,
            words_per_row,
            bits,
        }
    }

    /// Returns true iff q-token `q` is allowed to attend to absolute
    /// position `pos` (0..kv_len for cache, kv_len..kv_len+tree_len
    /// for tree).
    pub fn allows(&self, q: usize, pos: usize) -> bool {
        let w = pos / 32;
        let b = pos % 32;
        let row_off = q * self.words_per_row;
        (self.bits[row_off + w] >> b) & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(id: TokenId) -> DraftToken {
        DraftToken { id, p_draft: 1.0 }
    }

    #[test]
    fn root_only_tree() {
        let tree = DraftTree::from_root(t(7));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root().token.id, 7);
        assert_eq!(tree.max_depth(), 0);
        assert_eq!(tree.root_to_leaf_paths(), vec![vec![0]]);
    }

    #[test]
    fn balanced_depth2_branches2_has_seven_nodes() {
        let tree = DraftTree::balanced(2, 2, |i| t(i as TokenId));
        // 1 (root) + 2 + 4 = 7
        assert_eq!(tree.len(), 7);
        assert_eq!(tree.max_depth(), 2);
        // 4 root-to-leaf paths, all length 3.
        let paths = tree.root_to_leaf_paths();
        assert_eq!(paths.len(), 4);
        for p in &paths {
            assert_eq!(p.len(), 3);
            assert_eq!(p[0], 0);
        }
    }

    #[test]
    fn ancestors_walk_to_root() {
        let mut tree = DraftTree::from_root(t(0));
        let a = tree.add_child(0, t(1));
        let b = tree.add_child(a, t(2));
        let c = tree.add_child(b, t(3));
        let walk = tree.ancestors(c);
        assert_eq!(walk, vec![c, b, a, 0]);
    }

    #[test]
    fn mask_root_only_with_zero_kv() {
        let tree = DraftTree::from_root(t(0));
        let mask = TreeAttentionMask::build(&tree, 0);
        assert_eq!(mask.tree_len, 1);
        assert_eq!(mask.kv_len, 0);
        assert!(mask.allows(0, 0)); // root sees itself
    }

    #[test]
    fn mask_includes_full_kv_for_every_q() {
        let tree = DraftTree::balanced(2, 2, |i| t(i as TokenId));
        let kv_len = 100;
        let mask = TreeAttentionMask::build(&tree, kv_len);
        for q in 0..tree.len() {
            for j in 0..kv_len {
                assert!(mask.allows(q, j), "q={q} must see all of KV at pos {j}");
            }
        }
    }

    #[test]
    fn mask_only_allows_ancestors_in_tree_portion() {
        // Tree:    0 --> 1 --> 3
        //               \--> 2 --> 4
        let mut tree = DraftTree::from_root(t(0));
        let n1 = tree.add_child(0, t(1));
        let n2 = tree.add_child(0, t(2));
        let n3 = tree.add_child(n1, t(3));
        let n4 = tree.add_child(n2, t(4));

        let kv_len = 5;
        let mask = TreeAttentionMask::build(&tree, kv_len);

        // Node 4 should see [0, 2, 4] in tree, not 1 or 3.
        let want_for_4: Vec<usize> = vec![0, n2, n4]
            .into_iter()
            .map(|i| kv_len + i)
            .collect();
        for j in kv_len..(kv_len + tree.len()) {
            let allowed = mask.allows(n4, j);
            let expected = want_for_4.contains(&j);
            assert_eq!(
                allowed, expected,
                "node 4 attending tree pos {j}: got {allowed}, want {expected}"
            );
        }

        // Node 3 should see [0, 1, 3], not 2 or 4.
        let want_for_3: Vec<usize> = vec![0, n1, n3]
            .into_iter()
            .map(|i| kv_len + i)
            .collect();
        for j in kv_len..(kv_len + tree.len()) {
            let allowed = mask.allows(n3, j);
            let expected = want_for_3.contains(&j);
            assert_eq!(allowed, expected);
        }
    }

    #[test]
    fn mask_words_per_row_handles_unaligned() {
        let tree = DraftTree::balanced(1, 3, |i| t(i as TokenId)); // 4 nodes
        let kv_len = 33; // forces 2nd word
        let mask = TreeAttentionMask::build(&tree, kv_len);
        assert_eq!(mask.words_per_row, ((33 + 4) + 31) / 32);
        // Every q sees pos 32 (in 2nd word) of KV.
        for q in 0..tree.len() {
            assert!(mask.allows(q, 32));
        }
    }

    #[test]
    fn root_to_leaf_paths_match_depth() {
        let tree = DraftTree::balanced(3, 2, |i| t(i as TokenId));
        // 1 + 2 + 4 + 8 = 15 nodes, 8 paths of length 4.
        assert_eq!(tree.len(), 15);
        let paths = tree.root_to_leaf_paths();
        assert_eq!(paths.len(), 8);
        for p in &paths {
            assert_eq!(p.len(), 4);
        }
    }
}
