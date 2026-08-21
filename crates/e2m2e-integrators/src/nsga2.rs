//! NSGA-II 演化算子（纯 Rust 数值内核）。
//!
//! 目标函数评估、随机数生成和代际编排留在 Python：前两者分别承载 Python
//! 回调/ProcessPoolExecutor 契约与既有 NumPy 种子语义。本模块接收已生成的
//! 随机抽样，执行约束非支配排序、选择和变异算子。

use std::cmp::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_population(fit: &[Vec<f64>], viol: &[f64]) -> PyResult<usize> {
    if fit.is_empty() {
        return Err(PyValueError::new_err("fit must not be empty"));
    }
    if fit.len() != viol.len() {
        return Err(PyValueError::new_err(
            "fit and viol must have the same length",
        ));
    }
    let n_obj = fit[0].len();
    if n_obj == 0 || fit.iter().any(|row| row.len() != n_obj) {
        return Err(PyValueError::new_err(
            "fit must be a non-empty rectangular matrix",
        ));
    }
    Ok(n_obj)
}

/// 对齐 NumPy 升序 argsort 的标量顺序：-0.0 与 0.0 相等，NaN 排在末尾。
fn numpy_argsort_order(a: f64, b: f64) -> Ordering {
    match a.partial_cmp(&b) {
        Some(order) => order,
        None if a.is_nan() && !b.is_nan() => Ordering::Greater,
        None if !a.is_nan() && b.is_nan() => Ordering::Less,
        None => Ordering::Equal,
    }
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

fn non_dominated_sort(fit: &[Vec<f64>]) -> Vec<i64> {
    let n = fit.len();
    let mut domination_count = vec![0_usize; n];
    let mut dominated = vec![Vec::new(); n];
    let mut rank = vec![-1_i64; n];
    let mut current_front = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if dominates(&fit[p], &fit[q]) {
                dominated[p].push(q);
            } else if dominates(&fit[q], &fit[p]) {
                domination_count[p] += 1;
            }
        }
        if domination_count[p] == 0 {
            rank[p] = 0;
            current_front.push(p);
        }
    }

    let mut current_rank = 0_i64;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for p in current_front {
            for q in &dominated[p] {
                domination_count[*q] -= 1;
                if domination_count[*q] == 0 {
                    rank[*q] = current_rank + 1;
                    next_front.push(*q);
                }
            }
        }
        current_rank += 1;
        current_front = next_front;
    }
    rank
}

fn constrained_rank(fit: &[Vec<f64>], viol: &[f64]) -> Vec<i64> {
    let n = fit.len();
    let feasible: Vec<usize> = (0..n).filter(|&i| viol[i] <= 0.0).collect();
    let mut rank = vec![-1_i64; n];

    if !feasible.is_empty() {
        let feasible_fit: Vec<Vec<f64>> = feasible.iter().map(|&i| fit[i].clone()).collect();
        let feasible_rank = non_dominated_sort(&feasible_fit);
        for (position, &index) in feasible.iter().enumerate() {
            rank[index] = feasible_rank[position];
        }

        let infeasible: Vec<usize> = (0..n)
            .filter(|&i| matches!(viol[i].partial_cmp(&0.0), Some(Ordering::Greater) | None))
            .collect();
        if !infeasible.is_empty() {
            let max_feasible_rank = *feasible_rank.iter().max().unwrap_or(&0);
            let mut order: Vec<usize> = (0..infeasible.len()).collect();
            order.sort_by(|&a, &b| viol[infeasible[a]].total_cmp(&viol[infeasible[b]]));
            let mut current_rank = max_feasible_rank + 1;
            let mut previous_viol = -1.0;
            for (position, local_index) in order.into_iter().enumerate() {
                let index = infeasible[local_index];
                if viol[index] != previous_viol {
                    if position > 0 {
                        current_rank += 1;
                    }
                    previous_viol = viol[index];
                }
                rank[index] = current_rank;
            }
        }
    } else {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| viol[a].total_cmp(&viol[b]));
        let mut current_rank = 0_i64;
        let mut previous_viol = -1.0;
        for (position, index) in order.into_iter().enumerate() {
            if viol[index] != previous_viol {
                if position > 0 {
                    current_rank += 1;
                }
                previous_viol = viol[index];
            }
            rank[index] = current_rank;
        }
    }
    rank
}

fn crowding_distance(fit: &[Vec<f64>], rank: &[i64], n_obj: usize) -> Vec<f64> {
    let mut crowd = vec![0.0; fit.len()];
    let mut ranks = rank.to_vec();
    ranks.sort_unstable();
    ranks.dedup();

    for current_rank in ranks {
        let indices: Vec<usize> = (0..fit.len())
            .filter(|&index| rank[index] == current_rank)
            .collect();
        if indices.len() <= 2 {
            for index in indices {
                crowd[index] = f64::INFINITY;
            }
            continue;
        }

        for (objective, _) in fit[0].iter().enumerate().take(n_obj) {
            let mut order = indices.clone();
            order.sort_unstable_by(|&a, &b| {
                numpy_argsort_order(fit[a][objective], fit[b][objective])
            });
            crowd[order[0]] = f64::INFINITY;
            crowd[*order.last().expect("non-empty order")] = f64::INFINITY;
            let min = fit[order[0]][objective];
            let max = fit[*order.last().expect("non-empty order")][objective];
            let span = max - min;
            if span < 1e-30 {
                continue;
            }
            for position in 1..order.len() - 1 {
                let index = order[position];
                crowd[index] += (fit[order[position + 1]][objective]
                    - fit[order[position - 1]][objective])
                    / span;
            }
        }
    }
    crowd
}

/// Deb 可行支配规则的非支配排序与拥挤度距离。
#[pyfunction]
pub fn nsga2_sort_py(fit: Vec<Vec<f64>>, viol: Vec<f64>) -> PyResult<(Vec<i64>, Vec<f64>)> {
    let n_obj = validate_population(&fit, &viol)?;
    let rank = constrained_rank(&fit, &viol);
    let crowd = crowding_distance(&fit, &rank, n_obj);
    Ok((rank, crowd))
}

/// 按 ``(rank, -crowding)`` 进行环境选择。
#[pyfunction]
pub fn nsga2_environmental_selection_py(
    rank: Vec<i64>,
    crowd: Vec<f64>,
    n_keep: usize,
) -> PyResult<Vec<usize>> {
    if rank.len() != crowd.len() || n_keep > rank.len() {
        return Err(PyValueError::new_err("invalid rank, crowd or n_keep"));
    }
    let mut order: Vec<usize> = (0..rank.len()).collect();
    order.sort_by(|&a, &b| match rank[a].cmp(&rank[b]) {
        Ordering::Equal => crowd[b].total_cmp(&crowd[a]),
        other => other,
    });
    order.truncate(n_keep);
    Ok(order)
}

/// 使用 Python 已按原顺序产生的随机索引执行二元锦标赛。
#[pyfunction]
pub fn nsga2_tournament_selection_py(
    rank: Vec<i64>,
    crowd: Vec<f64>,
    draws: Vec<usize>,
) -> PyResult<Vec<usize>> {
    if rank.is_empty() || rank.len() != crowd.len() || !draws.len().is_multiple_of(2) {
        return Err(PyValueError::new_err(
            "invalid rank, crowd or tournament draws",
        ));
    }
    let n = rank.len();
    let mut selected = Vec::with_capacity(draws.len() / 2);
    for &[a, b] in draws.as_chunks::<2>().0 {
        if a >= n || b >= n {
            return Err(PyValueError::new_err(
                "tournament draw is outside population",
            ));
        }
        if rank[a] < rank[b] || (rank[a] == rank[b] && crowd[a] >= crowd[b]) {
            selected.push(a);
        } else {
            selected.push(b);
        }
    }
    Ok(selected)
}

fn validate_draws(draws: &[f64], expected: usize, name: &str) -> PyResult<()> {
    if draws.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{name} must contain {expected} values, got {}",
            draws.len()
        )));
    }
    Ok(())
}

/// 执行 SBX 交叉与多项式变异。
///
/// 随机抽样由 Python 按既有条件分支顺序准备；未使用的位置允许为 NaN。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn nsga2_variation_py(
    parents: Vec<Vec<f64>>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    crossover_prob: f64,
    eta_c: f64,
    mutation_prob: f64,
    eta_m: f64,
    crossover_draws: Vec<f64>,
    gene_draws: Vec<f64>,
    beta_draws: Vec<f64>,
    swap_draws: Vec<f64>,
    mutation_draws: Vec<f64>,
    mutation_value_draws: Vec<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    if parents.is_empty() || lo.is_empty() || lo.len() != hi.len() {
        return Err(PyValueError::new_err("invalid parents or bounds"));
    }
    let n = parents.len();
    let n_dim = lo.len();
    if parents.iter().any(|row| row.len() != n_dim) {
        return Err(PyValueError::new_err(
            "parents must match bounds dimensions",
        ));
    }
    let n_pairs = n / 2;
    validate_draws(&crossover_draws, n_pairs, "crossover_draws")?;
    for (draws, name) in [
        (&gene_draws, "gene_draws"),
        (&beta_draws, "beta_draws"),
        (&swap_draws, "swap_draws"),
    ] {
        validate_draws(draws, n_pairs * n_dim, name)?;
    }
    validate_draws(&mutation_draws, n * n_dim, "mutation_draws")?;
    validate_draws(&mutation_value_draws, n * n_dim, "mutation_value_draws")?;

    let mut offspring = parents.clone();
    for (pair, &crossover_draw) in crossover_draws.iter().enumerate() {
        if crossover_draw > crossover_prob {
            continue;
        }
        let first = 2 * pair;
        let second = first + 1;
        for dimension in 0..n_dim {
            let draw_index = pair * n_dim + dimension;
            let p1 = parents[first][dimension];
            let p2 = parents[second][dimension];
            if gene_draws[draw_index] > 0.5 || (p1 - p2).abs() <= 1e-14 {
                continue;
            }
            let (y1, y2) = if p1 < p2 { (p1, p2) } else { (p2, p1) };
            let random = beta_draws[draw_index];
            let beta = 1.0 + 2.0 * (y1 - lo[dimension]) / (y2 - y1);
            let alpha = 2.0 - beta.powf(-(eta_c + 1.0));
            let beta_q = if random <= 1.0 / alpha {
                (random * alpha).powf(1.0 / (eta_c + 1.0))
            } else {
                (1.0 / (2.0 - random * alpha)).powf(1.0 / (eta_c + 1.0))
            };
            let c1 = (0.5 * ((y1 + y2) - beta_q * (y2 - y1))).clamp(lo[dimension], hi[dimension]);

            let beta = 1.0 + 2.0 * (hi[dimension] - y2) / (y2 - y1);
            let alpha = 2.0 - beta.powf(-(eta_c + 1.0));
            let beta_q = if random <= 1.0 / alpha {
                (random * alpha).powf(1.0 / (eta_c + 1.0))
            } else {
                (1.0 / (2.0 - random * alpha)).powf(1.0 / (eta_c + 1.0))
            };
            let c2 = (0.5 * ((y1 + y2) + beta_q * (y2 - y1))).clamp(lo[dimension], hi[dimension]);
            if swap_draws[draw_index] <= 0.5 {
                offspring[first][dimension] = c2;
                offspring[second][dimension] = c1;
            } else {
                offspring[first][dimension] = c1;
                offspring[second][dimension] = c2;
            }
        }
    }

    for (individual, row) in offspring.iter_mut().enumerate() {
        for dimension in 0..n_dim {
            let draw_index = individual * n_dim + dimension;
            if mutation_draws[draw_index] > mutation_prob {
                continue;
            }
            let y = row[dimension];
            let delta1 = (y - lo[dimension]) / (hi[dimension] - lo[dimension]);
            let delta2 = (hi[dimension] - y) / (hi[dimension] - lo[dimension]);
            let random = mutation_value_draws[draw_index];
            let mutation_power = 1.0 / (eta_m + 1.0);
            let delta_q = if random <= 0.5 {
                let value = 2.0 * random + (1.0 - 2.0 * random) * (1.0 - delta1).powf(eta_m + 1.0);
                value.powf(mutation_power) - 1.0
            } else {
                let value =
                    2.0 * (1.0 - random) + 2.0 * (random - 0.5) * (1.0 - delta2).powf(eta_m + 1.0);
                1.0 - value.powf(mutation_power)
            };
            row[dimension] =
                (y + delta_q * (hi[dimension] - lo[dimension])).clamp(lo[dimension], hi[dimension]);
        }
    }
    Ok(offspring)
}

#[cfg(test)]
mod tests {
    use super::{constrained_rank, crowding_distance, non_dominated_sort};

    #[test]
    fn constrained_sort_puts_feasible_front_first() {
        let fit = vec![vec![1.0, 4.0], vec![2.0, 3.0], vec![0.0, 0.0]];
        let viol = vec![0.0, 0.0, 0.5];
        let rank = constrained_rank(&fit, &viol);
        assert_eq!(rank, vec![0, 0, 1]);
    }

    #[test]
    fn non_dominated_sort_assigns_layers() {
        let fit = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![1.0, 3.0]];
        assert_eq!(non_dominated_sort(&fit), vec![0, 1, 1]);
        let rank = vec![0, 1, 1];
        let crowd = crowding_distance(&fit, &rank, 2);
        assert!(crowd[0].is_infinite());
    }
}
