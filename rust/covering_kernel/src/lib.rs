//! Simulated annealing over covering designs, with a prescribed automorphism group.
//!
//! A design is held as `n_base` base blocks developed under `n_perms`
//! permutations of the point set.  Setting `n_perms = 1` to the identity makes
//! that an ordinary free design of `n_base` blocks; a larger group makes one
//! move carry a whole orbit.  Both are the same code path.
//!
//! The reason this exists is throughput.  The reference scores a design by
//! rebuilding coverage from every block, which is `B * C(k,t)` work per
//! evaluation.  Here coverage is a count per requirement, and swapping one
//! point of one base block touches only the requirements that contain the
//! point leaving and the point arriving: `2 * C(k-1,t-1)` per orbit member.
//! Annealing needs tens of millions of moves to find a covering, so that
//! difference decides whether the search works at all.

use std::slice;

const NONE: u32 = u32::MAX;
const REFUSED: u32 = u32::MAX;
const MAX_POINTS: usize = 255;
const MAX_BLOCK: usize = 64;
const MAX_STRENGTH: usize = 12;
const MAX_REQUIREMENTS: usize = 100_000_000;
const REPAIR_SAMPLE: usize = 8;

/// Xorshift64*, which is enough randomness for a Metropolis test and costs
/// nothing to carry.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mixed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        Rng(mixed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// Advance an ascending index combination in place, reporting whether one was left.
fn next_combination(idx: &mut [usize], n: usize) -> bool {
    let m = idx.len();
    let mut a = m;
    while a > 0 {
        a -= 1;
        if idx[a] != a + n - m {
            idx[a] += 1;
            for b in a + 1..m {
                idx[b] = idx[b - 1] + 1;
            }
            return true;
        }
    }
    false
}

struct Search {
    v: usize,
    k: usize,
    t: usize,
    n_base: usize,
    n_perms: usize,
    base: Vec<u32>,
    perms: Vec<u32>,
    inverse: Vec<u32>,
    binom: Vec<u64>,
    reqs: usize,
    cover: Vec<u32>,
    unc_list: Vec<u32>,
    unc_pos: Vec<u32>,
    lost: Vec<u32>,
    gained: Vec<u32>,
}

impl Search {
    fn new(
        v: usize,
        k: usize,
        t: usize,
        n_base: usize,
        base: &[u32],
        n_perms: usize,
        perms: &[u32],
    ) -> Option<Self> {
        let stride = t + 1;
        let mut binom = vec![0u64; (v + 1) * stride];
        for n in 0..=v {
            binom[n * stride] = 1;
            for r in 1..=t.min(n) {
                let upper = binom[(n - 1) * stride + r];
                let left = binom[(n - 1) * stride + r - 1];
                binom[n * stride + r] = upper + left;
            }
        }
        let reqs = binom[v * stride + t] as usize;
        if reqs == 0 || reqs > MAX_REQUIREMENTS {
            return None;
        }
        let mut inverse = vec![0u32; n_perms * v];
        for g in 0..n_perms {
            for point in 0..v {
                let image = perms[g * v + point] as usize;
                if image >= v {
                    return None;
                }
                inverse[g * v + image] = point as u32;
            }
        }
        for &point in base {
            if point as usize >= v {
                return None;
            }
        }
        let mut search = Search {
            v,
            k,
            t,
            n_base,
            n_perms,
            base: base.to_vec(),
            perms: perms.to_vec(),
            inverse,
            binom,
            reqs,
            cover: Vec::new(),
            unc_list: Vec::new(),
            unc_pos: Vec::new(),
            lost: Vec::with_capacity(n_perms * 1024),
            gained: Vec::with_capacity(n_perms * 1024),
        };
        search.rebuild();
        Some(search)
    }

    /// Colex rank of an ascending `t`-subset built by inserting `x` into `combo`.
    #[inline]
    fn rank_insert(&self, combo: &[u32], x: u32) -> usize {
        let stride = self.t + 1;
        let mut acc: u64 = 0;
        let mut j = 0usize;
        while j < combo.len() && combo[j] < x {
            acc += self.binom[combo[j] as usize * stride + j + 1];
            j += 1;
        }
        acc += self.binom[x as usize * stride + j + 1];
        let mut p = j;
        while p < combo.len() {
            acc += self.binom[combo[p] as usize * stride + p + 2];
            p += 1;
        }
        acc as usize
    }

    #[inline]
    fn rank_of(&self, members: &[u32]) -> usize {
        let stride = self.t + 1;
        let mut acc: u64 = 0;
        for (j, &point) in members.iter().enumerate() {
            acc += self.binom[point as usize * stride + j + 1];
        }
        acc as usize
    }

    /// Recover the ascending members of a requirement from its rank.
    fn unrank(&self, rank: usize, out: &mut [u32]) {
        let stride = self.t + 1;
        let mut remainder = rank;
        for j in (0..self.t).rev() {
            let mut a = self.v;
            loop {
                a -= 1;
                let value = self.binom[a * stride + j + 1] as usize;
                if value <= remainder {
                    out[j] = a as u32;
                    remainder -= value;
                    break;
                }
            }
        }
    }

    /// Recompute coverage from the blocks alone, then index what is uncovered.
    fn rebuild(&mut self) {
        self.cover.clear();
        self.cover.resize(self.reqs, 0);
        let (k, t, v, np, nb) = (self.k, self.t, self.v, self.n_perms, self.n_base);
        let mut img = [0u32; MAX_BLOCK];
        let mut combo = [0u32; MAX_STRENGTH];
        let mut idx = [0usize; MAX_STRENGTH];
        for b in 0..nb {
            for g in 0..np {
                for j in 0..k {
                    img[j] = self.perms[g * v + self.base[b * k + j] as usize];
                }
                img[..k].sort_unstable();
                for (a, slot) in idx.iter_mut().enumerate().take(t) {
                    *slot = a;
                }
                loop {
                    for a in 0..t {
                        combo[a] = img[idx[a]];
                    }
                    let rank = self.rank_of(&combo[..t]);
                    self.cover[rank] += 1;
                    if !next_combination(&mut idx[..t], k) {
                        break;
                    }
                }
            }
        }
        self.unc_list.clear();
        self.unc_pos.clear();
        self.unc_pos.resize(self.reqs, NONE);
        for rank in 0..self.reqs {
            if self.cover[rank] == 0 {
                self.unc_pos[rank] = self.unc_list.len() as u32;
                self.unc_list.push(rank as u32);
            }
        }
    }

    /// Collect the requirements the whole orbit loses and gains from one swap.
    fn build_move(&mut self, index: usize, leaving: u32, arriving: u32) {
        let (k, t, v, np) = (self.k, self.t, self.v, self.n_perms);
        let mut rem = [0u32; MAX_BLOCK];
        let mut count = 0usize;
        for j in 0..k {
            let point = self.base[index * k + j];
            if point != leaving {
                rem[count] = point;
                count += 1;
            }
        }
        self.lost.clear();
        self.gained.clear();
        let mut img = [0u32; MAX_BLOCK];
        let mut combo = [0u32; MAX_STRENGTH];
        let mut idx = [0usize; MAX_STRENGTH];
        let width = t - 1;
        for g in 0..np {
            let (out_image, in_image) = {
                let perm = &self.perms[g * v..(g + 1) * v];
                for j in 0..count {
                    img[j] = perm[rem[j] as usize];
                }
                (perm[leaving as usize], perm[arriving as usize])
            };
            img[..count].sort_unstable();
            for (a, slot) in idx.iter_mut().enumerate().take(width) {
                *slot = a;
            }
            loop {
                for a in 0..width {
                    combo[a] = img[idx[a]];
                }
                let dropped = self.rank_insert(&combo[..width], out_image) as u32;
                let added = self.rank_insert(&combo[..width], in_image) as u32;
                self.lost.push(dropped);
                self.gained.push(added);
                if width == 0 || !next_combination(&mut idx[..width], count) {
                    break;
                }
            }
        }
    }

    /// Apply the collected move to coverage, returning the change in uncovered count.
    ///
    /// Applying and measuring in one pass is deliberate: orbit images can
    /// collide, so the same requirement may appear several times across the
    /// two lists, and any delta computed by inspection rather than by doing
    /// the work would be wrong exactly in the symmetric cases this kernel
    /// exists to search.
    fn apply(&mut self) -> i64 {
        let mut delta = 0i64;
        for position in 0..self.lost.len() {
            let rank = self.lost[position] as usize;
            self.cover[rank] -= 1;
            if self.cover[rank] == 0 {
                delta += 1;
            }
        }
        for position in 0..self.gained.len() {
            let rank = self.gained[position] as usize;
            if self.cover[rank] == 0 {
                delta -= 1;
            }
            self.cover[rank] += 1;
        }
        delta
    }

    fn rollback(&mut self) {
        for position in 0..self.gained.len() {
            let rank = self.gained[position] as usize;
            self.cover[rank] -= 1;
        }
        for position in 0..self.lost.len() {
            let rank = self.lost[position] as usize;
            self.cover[rank] += 1;
        }
    }

    #[inline]
    fn sync(&mut self, rank: u32) {
        let empty = self.cover[rank as usize] == 0;
        let position = self.unc_pos[rank as usize];
        if empty && position == NONE {
            self.unc_pos[rank as usize] = self.unc_list.len() as u32;
            self.unc_list.push(rank);
        } else if !empty && position != NONE {
            let last = self.unc_list[self.unc_list.len() - 1];
            self.unc_list[position as usize] = last;
            self.unc_pos[last as usize] = position;
            self.unc_list.pop();
            self.unc_pos[rank as usize] = NONE;
        }
    }

    fn sync_lists(&mut self) {
        for position in 0..self.lost.len() {
            let rank = self.lost[position];
            self.sync(rank);
        }
        for position in 0..self.gained.len() {
            let rank = self.gained[position];
            self.sync(rank);
        }
    }

    /// Choose a swap, aiming at an uncovered requirement with probability `guided`.
    ///
    /// The guided move is what makes the endgame finish.  A uniformly random
    /// swap almost never touches the handful of requirements still missing
    /// once a design is nearly complete, so the search wanders; pulling a
    /// missing requirement's own points into a block attacks the actual
    /// deficit.  Under a group the requirement is pulled back through a random
    /// orbit element first, since it is the base block that has to change.
    ///
    /// Which block to repair matters as much as which requirement.  A random
    /// block usually shares nothing with the missing subset, so covering it
    /// costs several other requirements; a block that already holds all but
    /// one of its points costs almost nothing.  So a few blocks are sampled
    /// and the one with the most of the requirement already in it wins.
    fn propose(&mut self, rng: &mut Rng, guided: f64) -> (usize, u32, u32) {
        let (k, t, v) = (self.k, self.t, self.v);
        let mut index = rng.below(self.n_base);
        if guided > 0.0 && !self.unc_list.is_empty() && rng.unit() < guided {
            let mut wanted = [0u32; MAX_STRENGTH];
            let pick = self.unc_list[rng.below(self.unc_list.len())] as usize;
            self.unrank(pick, &mut wanted);
            let g = rng.below(self.n_perms);
            for slot in wanted.iter_mut().take(t) {
                *slot = self.inverse[g * v + *slot as usize];
            }
            let samples = REPAIR_SAMPLE.min(self.n_base);
            let mut overlap = 0usize;
            for sample in 0..samples {
                let candidate = if sample == 0 { index } else { rng.below(self.n_base) };
                let block = &self.base[candidate * k..(candidate + 1) * k];
                let mut shared = 0usize;
                for point in wanted.iter().take(t) {
                    if block.contains(point) {
                        shared += 1;
                    }
                }
                if shared > overlap || sample == 0 {
                    overlap = shared;
                    index = candidate;
                }
            }
            let block = &self.base[index * k..(index + 1) * k];
            let mut outside = [0u32; MAX_STRENGTH];
            let mut n_outside = 0usize;
            for &point in wanted.iter().take(t) {
                if !block.contains(&point) {
                    outside[n_outside] = point;
                    n_outside += 1;
                }
            }
            if n_outside > 0 {
                let arriving = outside[rng.below(n_outside)];
                let mut spare = [0u32; MAX_BLOCK];
                let mut n_spare = 0usize;
                for j in 0..k {
                    let point = block[j];
                    if !wanted[..t].contains(&point) {
                        spare[n_spare] = point;
                        n_spare += 1;
                    }
                }
                if n_spare > 0 {
                    return (index, spare[rng.below(n_spare)], arriving);
                }
            }
        }
        let leaving = self.base[index * k + rng.below(k)];
        loop {
            let arriving = rng.below(v) as u32;
            if !self.base[index * k..(index + 1) * k].contains(&arriving) {
                return (index, leaving, arriving);
            }
        }
    }

    /// Anneal, returning the fewest uncovered requirements reached.
    fn anneal(
        &mut self,
        steps: u64,
        seed: u64,
        start_temp: f64,
        end_temp: f64,
        guided: f64,
        best_base: &mut [u32],
    ) -> (u32, u64) {
        let mut rng = Rng::new(seed);
        let mut uncovered = self.unc_list.len() as i64;
        let mut best = uncovered;
        best_base.copy_from_slice(&self.base);
        if best == 0 || steps == 0 {
            return (best as u32, 0);
        }
        let ratio = (end_temp / start_temp).ln() / steps as f64;
        let mut temperature = start_temp;
        let mut executed: u64 = 0;
        for step in 0..steps {
            executed += 1;
            if step % 1024 == 0 {
                temperature = start_temp * (ratio * step as f64).exp();
            }
            let (index, leaving, arriving) = self.propose(&mut rng, guided);
            self.build_move(index, leaving, arriving);
            let delta = self.apply();
            let accept = delta <= 0 || rng.unit() < (-(delta as f64) / temperature).exp();
            if !accept {
                self.rollback();
                continue;
            }
            self.sync_lists();
            for j in 0..self.k {
                if self.base[index * self.k + j] == leaving {
                    self.base[index * self.k + j] = arriving;
                    break;
                }
            }
            uncovered += delta;
            if uncovered < best {
                best = uncovered;
                best_base.copy_from_slice(&self.base);
                if best == 0 {
                    break;
                }
            }
        }
        (best as u32, executed)
    }
}

fn shape_ok(v: usize, k: usize, t: usize, n_base: usize, n_perms: usize) -> bool {
    v <= MAX_POINTS
        && k <= MAX_BLOCK
        && t <= MAX_STRENGTH
        && 1 <= t
        && t <= k
        && k <= v
        && n_base >= 1
        && n_perms >= 1
}

/// Anneal a design, writing the best base blocks found to `best_out` and the
/// number of moves actually performed to `moves_done`.
///
/// The search stops the moment nothing is uncovered, so the moves requested
/// and the moves performed are different numbers. Reporting only the request
/// makes every caller's budget ledger fiction exactly when the search succeeds.
///
/// # Safety
///
/// `base` and `best_out` must each address `n_base * k` words, `perms` must
/// address `n_perms * v` words, and `moves_done` must be a valid `u64`.
#[no_mangle]
pub unsafe extern "C" fn covering_anneal(
    v: u32,
    k: u32,
    t: u32,
    n_base: u32,
    base: *const u32,
    n_perms: u32,
    perms: *const u32,
    steps: u64,
    seed: u64,
    start_temp: f64,
    end_temp: f64,
    guided: f64,
    best_out: *mut u32,
    moves_done: *mut u64,
) -> u32 {
    let (v, k, t) = (v as usize, k as usize, t as usize);
    let (n_base, n_perms) = (n_base as usize, n_perms as usize);
    if !shape_ok(v, k, t, n_base, n_perms) || start_temp <= 0.0 || end_temp <= 0.0 {
        return REFUSED;
    }
    let base_slice = slice::from_raw_parts(base, n_base * k);
    let perm_slice = slice::from_raw_parts(perms, n_perms * v);
    let out_slice = slice::from_raw_parts_mut(best_out, n_base * k);
    match Search::new(v, k, t, n_base, base_slice, n_perms, perm_slice) {
        None => REFUSED,
        Some(mut search) => {
            let (best, executed) =
                search.anneal(steps, seed, start_temp, end_temp, guided, out_slice);
            if !moves_done.is_null() {
                *moves_done = executed;
            }
            best
        }
    }
}

/// Count the requirements a developed design leaves uncovered.
///
/// # Safety
///
/// `base` must address `n_base * k` words and `perms` must address
/// `n_perms * v` words.
#[no_mangle]
pub unsafe extern "C" fn covering_uncovered(
    v: u32,
    k: u32,
    t: u32,
    n_base: u32,
    base: *const u32,
    n_perms: u32,
    perms: *const u32,
) -> u32 {
    let (v, k, t) = (v as usize, k as usize, t as usize);
    let (n_base, n_perms) = (n_base as usize, n_perms as usize);
    if !shape_ok(v, k, t, n_base, n_perms) {
        return REFUSED;
    }
    let base_slice = slice::from_raw_parts(base, n_base * k);
    let perm_slice = slice::from_raw_parts(perms, n_perms * v);
    match Search::new(v, k, t, n_base, base_slice, n_perms, perm_slice) {
        None => REFUSED,
        Some(search) => search.unc_list.len() as u32,
    }
}

/// Report the kernel ABI version, so a stale build is detected rather than trusted.
#[no_mangle]
pub extern "C" fn covering_kernel_version() -> u32 {
    2
}
