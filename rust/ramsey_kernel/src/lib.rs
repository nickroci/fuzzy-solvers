//! Exact maximum-independent-set over small bitset graphs.
//!
//! This is the hot loop of the Ramsey search and nothing else. The Python
//! referee owns correctness and is the definition of the answer; this crate
//! exists only because that definition, evaluated in Python on a 139-vertex
//! graph, runs at about one graph per second, which is too slow to search
//! anything.
//!
//! The algorithm is deliberately identical to the Python one, so the two agree
//! on every input: branch and bound, recording the incumbent, saturating at a
//! caller-supplied ceiling, and pruning with a clique-cover bound. In a
//! triangle-free graph every clique is a vertex or an edge, so a matching *is*
//! the best cover of that shape and `alpha <= |S| - nu` holds for any matching.
//!
//! Deliberately dependency-free: it is loaded through `ctypes`, which releases
//! the interpreter lock across the call, so a thread pool calling in here gets
//! real parallelism rather than contending on the GIL.

/// Vertex capacity, in 64-bit words. Four words covers 256 vertices, which is
/// far above anything this campaign builds.
const MAX_WORDS: usize = 4;

type Bitset = [u64; MAX_WORDS];

#[inline]
fn count(set: &Bitset, words: usize) -> u32 {
    let mut total = 0;
    for word in set.iter().take(words) {
        total += word.count_ones();
    }
    total
}

#[inline]
fn is_empty(set: &Bitset, words: usize) -> bool {
    set.iter().take(words).all(|word| *word == 0)
}

#[inline]
fn lowest(set: &Bitset, words: usize) -> Option<usize> {
    for (index, word) in set.iter().take(words).enumerate() {
        if *word != 0 {
            return Some(index * 64 + word.trailing_zeros() as usize);
        }
    }
    None
}

#[inline]
fn contains(set: &Bitset, vertex: usize) -> bool {
    set[vertex / 64] >> (vertex % 64) & 1 == 1
}

#[inline]
fn clear(set: &mut Bitset, vertex: usize) {
    set[vertex / 64] &= !(1u64 << (vertex % 64));
}

#[inline]
fn row<'a>(adjacency: &'a [u64], vertex: usize, words: usize) -> &'a [u64] {
    &adjacency[vertex * words..(vertex + 1) * words]
}

#[inline]
fn intersection_count(set: &Bitset, other: &[u64], words: usize) -> u32 {
    let mut total = 0;
    for index in 0..words {
        total += (set[index] & other[index]).count_ones();
    }
    total
}

/// Bound the independence number of an induced subgraph by a clique cover.
///
/// An independent set takes at most one vertex from each clique, so a matching
/// of size `nu` absorbs two vertices while contributing one and
/// `alpha <= |S| - nu`. A greedy maximal matching is enough for soundness.
fn clique_cover_bound(candidates: &Bitset, adjacency: &[u64], words: usize) -> u32 {
    let mut remaining = *candidates;
    let mut matched = 0;
    while let Some(vertex) = lowest(&remaining, words) {
        clear(&mut remaining, vertex);
        let neighbours = row(adjacency, vertex, words);
        let mut partner = None;
        for index in 0..words {
            let shared = remaining[index] & neighbours[index];
            if shared != 0 {
                partner = Some(index * 64 + shared.trailing_zeros() as usize);
                break;
            }
        }
        if let Some(other) = partner {
            clear(&mut remaining, other);
            matched += 1;
        }
    }
    count(candidates, words) - matched
}

/// Return the candidate with the most neighbours still in play.
///
/// Branching on the busiest vertex shrinks the candidate set fastest in the
/// inclusion branch, which is the cheaper of the two to exhaust.
fn busiest(candidates: &Bitset, adjacency: &[u64], words: usize) -> usize {
    let mut best_vertex = usize::MAX;
    let mut best_degree = -1i64;
    let mut remaining = *candidates;
    while let Some(vertex) = lowest(&remaining, words) {
        clear(&mut remaining, vertex);
        let degree = intersection_count(candidates, row(adjacency, vertex, words), words) as i64;
        if degree > best_degree {
            best_degree = degree;
            best_vertex = vertex;
        }
    }
    best_vertex
}

fn search(
    candidates: &Bitset,
    size: u32,
    best: &mut u32,
    adjacency: &[u64],
    words: usize,
    ceiling: u32,
) {
    if size > *best {
        *best = size;
    }
    if *best >= ceiling {
        return;
    }
    if is_empty(candidates, words) || size + clique_cover_bound(candidates, adjacency, words) <= *best
    {
        return;
    }
    let pivot = busiest(candidates, adjacency, words);
    let neighbours = row(adjacency, pivot, words);

    let mut included = *candidates;
    for index in 0..words {
        included[index] &= !neighbours[index];
    }
    clear(&mut included, pivot);
    search(&included, size + 1, best, adjacency, words, ceiling);
    if *best >= ceiling {
        return;
    }

    let mut excluded = *candidates;
    clear(&mut excluded, pivot);
    search(&excluded, size, best, adjacency, words, ceiling);
}

/// Return `min(alpha, ceiling)` for the graph given as packed adjacency rows.
///
/// `adjacency` is `order * words` u64 values, row-major, one row of `words`
/// words per vertex. Returns `u32::MAX` when the shape is unsupported, which
/// the caller treats as "fall back to the reference implementation" rather
/// than as an answer.
///
/// # Safety
///
/// `adjacency` must point to at least `order * words` readable u64 values.
#[no_mangle]
pub unsafe extern "C" fn capped_independence(
    adjacency: *const u64,
    order: u32,
    words: u32,
    ceiling: u32,
) -> u32 {
    if adjacency.is_null() || words as usize > MAX_WORDS || order as usize > MAX_WORDS * 64 {
        return u32::MAX;
    }
    if ceiling == 0 || order == 0 {
        return 0;
    }
    let words = words as usize;
    let rows = std::slice::from_raw_parts(adjacency, order as usize * words);
    let mut candidates: Bitset = [0; MAX_WORDS];
    for vertex in 0..order as usize {
        candidates[vertex / 64] |= 1u64 << (vertex % 64);
    }
    let mut best = 0;
    search(&candidates, 0, &mut best, rows, words, ceiling);
    best
}

/// Report the vertex capacity, so callers can check compatibility.
#[no_mangle]
pub extern "C" fn max_order() -> u32 {
    (MAX_WORDS * 64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(order: usize, edges: &[(usize, usize)]) -> (Vec<u64>, usize) {
        let words = order.div_ceil(64);
        let mut rows = vec![0u64; order * words];
        for &(a, b) in edges {
            rows[a * words + b / 64] |= 1u64 << (b % 64);
            rows[b * words + a / 64] |= 1u64 << (a % 64);
        }
        (rows, words)
    }

    #[test]
    fn cycle_of_five_has_independence_two() {
        let (rows, words) = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
        let alpha =
            unsafe { capped_independence(rows.as_ptr(), 5, words as u32, 99) };
        assert_eq!(alpha, 2);
    }

    #[test]
    fn the_ceiling_saturates() {
        let (rows, words) = graph(5, &[(0, 1)]);
        let alpha = unsafe { capped_independence(rows.as_ptr(), 5, words as u32, 2) };
        assert_eq!(alpha, 2);
    }

    #[test]
    fn an_empty_graph_is_wholly_independent() {
        let (rows, words) = graph(6, &[]);
        let alpha = unsafe { capped_independence(rows.as_ptr(), 6, words as u32, 99) };
        assert_eq!(alpha, 6);
    }

    #[test]
    fn an_unsupported_shape_is_refused() {
        let (rows, words) = graph(5, &[]);
        let alpha = unsafe { capped_independence(rows.as_ptr(), 5, (words + 9) as u32, 9) };
        assert_eq!(alpha, u32::MAX);
    }
}
