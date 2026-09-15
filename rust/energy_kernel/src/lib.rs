//! Low-weight-spectrum energy of a binary linear code.
//!
//! Minimum distance is the wrong optimisation signal here. It is a single
//! integer that moves rarely, so a hill-climber on it sees a flat landscape
//! and stalls - which is exactly what nineteen million quasi-cyclic candidates
//! did, every one landing on the published value and none above it.
//!
//! This scores the whole sub-threshold shell instead:
//!
//! ```text
//! E_D = sum over nonzero messages of max(0, D - weight)^2
//! ```
//!
//! A weight `D-1` word contributes 1, a `D-2` word 4, a `D-3` word 9. Squaring
//! the deficit distinguishes a code that is one word away from the target from
//! one that is structurally far from it, which a count of bad words cannot.
//! `E_D == 0` certifies distance at least `D`.
//!
//! `abort` is the annealing-aware early exit. A proposal is rejected once its
//! accumulated penalty passes the threshold the Metropolis test would need, so
//! hopeless candidates cost a fraction of the enumeration while every accepted
//! one is scored in full. Passing `u64::MAX` scores unconditionally.

const MAX_WORDS: usize = 2;

/// Return `E_D` for the span of `basis`, or the abort threshold once passed.
///
/// `basis` is `rank * words` u64 values, row-major, and must be independent -
/// a systematic generator guarantees that by construction.
///
/// # Safety
///
/// `basis` must point to at least `rank * words` readable u64 values.
#[no_mangle]
pub unsafe extern "C" fn spectrum_energy(
    basis: *const u64,
    rank: u32,
    words: u32,
    target: u32,
    abort: u64,
) -> u64 {
    if basis.is_null() || words as usize > MAX_WORDS || rank > 30 || rank == 0 {
        return u64::MAX;
    }
    let words = words as usize;
    let rows = std::slice::from_raw_parts(basis, rank as usize * words);
    let mut current = [0u64; MAX_WORDS];
    let mut energy: u64 = 0;
    let total: u64 = 1u64 << rank;
    for step in 1..total {
        let offset = step.trailing_zeros() as usize * words;
        let mut weight = 0u32;
        for index in 0..words {
            current[index] ^= rows[offset + index];
            weight += current[index].count_ones();
        }
        if weight < target && weight > 0 {
            let deficit = (target - weight) as u64;
            energy += deficit * deficit;
            if energy > abort {
                return energy;
            }
        }
    }
    energy
}

/// Return the minimum non-zero weight, for certification rather than search.
///
/// # Safety
///
/// `basis` must point to at least `rank * words` readable u64 values.
#[no_mangle]
pub unsafe extern "C" fn minimum_weight(basis: *const u64, rank: u32, words: u32) -> u32 {
    if basis.is_null() || words as usize > MAX_WORDS || rank > 30 || rank == 0 {
        return u32::MAX;
    }
    let words = words as usize;
    let rows = std::slice::from_raw_parts(basis, rank as usize * words);
    let mut current = [0u64; MAX_WORDS];
    let mut best = u32::MAX;
    for step in 1..(1u64 << rank) {
        let offset = step.trailing_zeros() as usize * words;
        let mut weight = 0u32;
        for index in 0..words {
            current[index] ^= rows[offset + index];
            weight += current[index].count_ones();
        }
        if weight > 0 && weight < best {
            best = weight;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_meeting_the_target_has_zero_energy() {
        // single row of weight 5; target 5 means nothing is sub-threshold
        let rows = [0b11111u64];
        assert_eq!(unsafe { spectrum_energy(rows.as_ptr(), 1, 1, 5, u64::MAX) }, 0);
    }

    #[test]
    fn deficits_are_squared() {
        // one codeword of weight 3, target 5 -> deficit 2 -> energy 4
        let rows = [0b111u64];
        assert_eq!(unsafe { spectrum_energy(rows.as_ptr(), 1, 1, 5, u64::MAX) }, 4);
    }

    #[test]
    fn the_abort_stops_accumulation() {
        let rows = [0b1u64, 0b10u64, 0b100u64];
        let full = unsafe { spectrum_energy(rows.as_ptr(), 3, 1, 9, u64::MAX) };
        let stopped = unsafe { spectrum_energy(rows.as_ptr(), 3, 1, 9, 10) };
        assert!(stopped > 10);
        assert!(stopped <= full);
    }

    #[test]
    fn minimum_weight_matches_a_known_code() {
        let rows = [0b1101001u64, 0b1011010, 0b0111100, 0b1110000];
        assert_eq!(unsafe { minimum_weight(rows.as_ptr(), 4, 1) }, 3);
    }
}
