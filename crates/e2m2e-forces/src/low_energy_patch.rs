//! 低能转移流形截面态配对的纯 Rust 数值核。
//!
//! 输入为两组展平的六维截面态和位置/速度权重。串行与 Rayon 并行路径均
//! 生成 `(i_a, i_b)` 字典序的原始候选，再作稳定的代价排序；因此代价相等时
//! 保持 Python 原有嵌套循环的顺序。

use std::cmp::Ordering;

/// 两个流形管截面态的拼接候选。
#[derive(Clone, Debug)]
pub struct LowEnergyPatchCandidate {
    pub i_a: usize,
    pub i_b: usize,
    pub state_a: [f64; 6],
    pub state_b: [f64; 6],
    pub delta_r: f64,
    pub delta_v: f64,
    pub cost: f64,
}

fn validate_states(states: &[f64], name: &str) {
    assert!(
        states.len().is_multiple_of(6),
        "{name} 展平长度必须是 6 的倍数，得到 {}",
        states.len()
    );
}

fn candidate_at(
    states_a: &[f64],
    states_b: &[f64],
    i_a: usize,
    i_b: usize,
    weight_r: f64,
    weight_v: f64,
) -> LowEnergyPatchCandidate {
    let mut state_a = [0.0_f64; 6];
    let mut state_b = [0.0_f64; 6];
    state_a.copy_from_slice(&states_a[i_a * 6..i_a * 6 + 6]);
    state_b.copy_from_slice(&states_b[i_b * 6..i_b * 6 + 6]);

    let delta_r = state_a[..3]
        .iter()
        .zip(&state_b[..3])
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let delta_v = state_a[3..]
        .iter()
        .zip(&state_b[3..])
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    LowEnergyPatchCandidate {
        i_a,
        i_b,
        state_a,
        state_b,
        delta_r,
        delta_v,
        cost: weight_r * delta_r + weight_v * delta_v,
    }
}

fn stable_sort_by_cost(candidates: &mut [LowEnergyPatchCandidate]) {
    // `sort_by` 是稳定排序；不可比较的 NaN 按相等处理，和 Python float
    // 比较不改变其相对顺序的行为对齐。
    candidates.sort_by(|left, right| {
        left.cost
            .partial_cmp(&right.cost)
            .unwrap_or(Ordering::Equal)
    });
}

/// 串行生成并稳定排序全部截面态配对。
pub fn low_energy_patch_serial(
    states_a: &[f64],
    states_b: &[f64],
    weight_r: f64,
    weight_v: f64,
    progress_tx: Option<&crossbeam_channel::Sender<usize>>,
) -> Vec<LowEnergyPatchCandidate> {
    validate_states(states_a, "states_a");
    validate_states(states_b, "states_b");

    let n_a = states_a.len() / 6;
    let n_b = states_b.len() / 6;
    let total = n_a.checked_mul(n_b).expect("n_a * n_b 溢出");
    let mut candidates = Vec::with_capacity(total);
    for i_a in 0..n_a {
        for i_b in 0..n_b {
            candidates.push(candidate_at(
                states_a, states_b, i_a, i_b, weight_r, weight_v,
            ));
            if let Some(tx) = progress_tx {
                let _ = tx.send(1);
            }
        }
    }
    stable_sort_by_cost(&mut candidates);
    candidates
}

/// Rayon 并行生成并稳定排序全部截面态配对。
pub fn low_energy_patch_parallel(
    states_a: &[f64],
    states_b: &[f64],
    weight_r: f64,
    weight_v: f64,
    progress_tx: Option<&crossbeam_channel::Sender<usize>>,
) -> Vec<LowEnergyPatchCandidate> {
    use rayon::prelude::*;

    validate_states(states_a, "states_a");
    validate_states(states_b, "states_b");

    let n_a = states_a.len() / 6;
    let n_b = states_b.len() / 6;
    let total = n_a.checked_mul(n_b).expect("n_a * n_b 溢出");
    let mut candidates: Vec<_> = (0..total)
        .into_par_iter()
        .map(|idx| {
            let i_a = idx / n_b;
            let i_b = idx % n_b;
            let candidate = candidate_at(states_a, states_b, i_a, i_b, weight_r, weight_v);
            if let Some(tx) = progress_tx {
                let _ = tx.send(1);
            }
            candidate
        })
        .collect();
    stable_sort_by_cost(&mut candidates);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_preserves_equal_cost_order() {
        let states_a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let states_b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let serial = low_energy_patch_serial(&states_a, &states_b, 1.0, 1.0, None);
        let parallel = low_energy_patch_parallel(&states_a, &states_b, 1.0, 1.0, None);

        let expected = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        assert_eq!(
            serial.iter().map(|c| (c.i_a, c.i_b)).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            parallel.iter().map(|c| (c.i_a, c.i_b)).collect::<Vec<_>>(),
            expected
        );
    }
}
