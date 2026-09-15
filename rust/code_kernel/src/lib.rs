//! Minimum distance of a binary linear code, by Gray-code enumeration.
//!
//! The Python referee defines the answer; this is the same computation made
//! affordable. A code of dimension `k` has `2^k` codewords, and the cost of
//! visiting them all is what decides whether a search of the tables is
//! possible at all.
//!
//! Enumerating in Gray-code order is what makes it cheap: consecutive subsets
//! differ in exactly one basis row, so each codeword costs one XOR and one
//! popcount rather than rebuilding a linear combination. The work is `2^k`
//! machine operations — large but *bounded*, which is the property the
//! independent-set search never had.
//!
//! `floor` is the early exit. A candidate only has to beat the current best
//! known distance, so the instant a lighter codeword appears the answer is
//! decided and the rest of the enumeration is waste. Rejection is therefore
//! nearly free and only a genuinely promising code pays in full.
//!
//! Dependency-free and loaded through `ctypes`, which releases the interpreter
//! lock across the call, so a thread pool gets real parallelism.

const MAX_WORDS: usize = 4;

#[inline]
fn weight(word: &[u64], words: usize) -> u32 {
    let mut total = 0;
    for value in word.iter().take(words) {
        total += value.count_ones();
    }
    total
}

/// Return the minimum non-zero codeword weight of the span of `basis`.
///
/// `basis` holds `rank * words` u64 values, row-major. Returns `u32::MAX` for
/// an unsupported shape, which the caller treats as "use the reference
/// implementation" rather than as an answer. A rank of zero yields zero: the
/// trivial code has no non-zero codeword.
///
/// # Safety
///
/// `basis` must point to at least `rank * words` readable u64 values.
#[no_mangle]
pub unsafe extern "C" fn minimum_distance(
    basis: *const u64,
    rank: u32,
    words: u32,
    floor: u32,
) -> u32 {
    if basis.is_null() || words as usize > MAX_WORDS || rank > 40 {
        return u32::MAX;
    }
    if rank == 0 {
        return 0;
    }
    let words = words as usize;
    let rows = std::slice::from_raw_parts(basis, rank as usize * words);
    let mut current = [0u64; MAX_WORDS];
    let mut best = u32::MAX;
    let total: u64 = 1u64 << rank;
    for step in 1..total {
        // Gray code: successive subsets differ by the row at the lowest set bit.
        let row = step.trailing_zeros() as usize;
        let offset = row * words;
        for index in 0..words {
            current[index] ^= rows[offset + index];
        }
        let found = weight(&current, words);
        if found != 0 && found < best {
            best = found;
            if floor != 0 && best < floor {
                return best;
            }
        }
    }
    best
}

/// Report the largest rank this kernel will enumerate.
#[no_mangle]
pub extern "C" fn max_rank() -> u32 {
    40
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distance(rows: &[u64], floor: u32) -> u32 {
        unsafe { minimum_distance(rows.as_ptr(), rows.len() as u32, 1, floor) }
    }

    #[test]
    fn repetition_code_has_distance_three() {
        assert_eq!(distance(&[0b111], 0), 3);
    }

    #[test]
    fn the_hamming_seven_four_code_has_distance_three() {
        // [7,4,3] Hamming, generator rows in systematic form.
        let rows = [0b1101001, 0b1011010, 0b0111100, 0b1110000];
        assert_eq!(distance(&rows, 0), 3);
    }

    #[test]
    fn the_floor_short_circuits() {
        let rows = [0b0000011, 0b1111100];
        assert!(distance(&rows, 5) < 5);
    }

    #[test]
    fn a_trivial_code_has_no_nonzero_word() {
        let empty: [u64; 0] = [];
        assert_eq!(unsafe { minimum_distance(empty.as_ptr(), 0, 1, 0) }, 0);
    }

    #[test]
    fn an_unsupported_shape_is_refused() {
        assert_eq!(distance(&[0b111], 0), 3);
        let rows = [0b111u64];
        assert_eq!(
            unsafe { minimum_distance(rows.as_ptr(), 41, 1, 0) },
            u32::MAX
        );
    }
}
