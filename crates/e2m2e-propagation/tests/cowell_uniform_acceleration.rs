//! Cowell 匀加速精确再现（ADR 0013：按定义验证）。
//!
//! 匀加速运动 x'' = a₀ 有解析解 x(t) = x₀ + v₀ t + ½ a₀ t²。
//! Cowell 积分的核心特征是对二次多项式的精确再现——若加速度为常数，
//! 任何阶数 ≥ 2 的 Störmer-Cowell 方法应在机器精度内产生精确结果。
//!
//! 依据：Cowell 校正器的权重满足 Σᵢ qᵢ = 1 且所有 ⱼ ≥ 1 阶差分
//! ∇ʲa 在 a=const 时为零，故 h²·q₀·a = h²·a，∑_{j≥1} q_j·∇ʲa = 0。
//! 结合 x_{n+1} = 2x_n − x_{n-1} + h²a 精确满足匀加速运动的封闭解。
//!
//! # ADR 0013
//! 匀加速运动的封闭解是二次多项式的"定义"——不依赖任何外部软件或 golden 文件。

use e2m2e_propagation::cowell::{cowell_step, COWELL_HISTORY_LEN};

/// 匀加速运动 1D 测试：a = const，已知 x(t) = x₀ + v₀ t + ½ a t²。
///
/// 用采样精确 a = const 填充 Cowell 历史缓冲（Cowell 的 a 与 t 无关，
/// 但与 x 无关也无关）。多次步进后与解析解对比误差应为机器噪声级别。
#[test]
fn test_constant_acceleration_exact_reproduction() {
    let a0 = 5.0; // constant acceleration (m/s²)
    let accel =
        |_t: f64, _x: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> { Ok(vec![a0]) };

    let x0 = 3.0;
    let v0 = 2.0;
    let h: f64 = 0.01;

    // 历史上 a 为 const，所有 8 个加速度采样均为 a0。
    // x(-h) = x0 − v0·h + ½ a0·h²
    // x(0)  = x0
    let x_neg_h = x0 - v0 * h + 0.5 * a0 * h * h;
    let mut history: Vec<Vec<f64>> = vec![vec![x_neg_h], vec![x0]];
    for _ in 0..COWELL_HISTORY_LEN - 2 {
        history.push(vec![a0]);
    }

    let n_steps = 100;
    let mut t = 0.0;
    for _ in 0..n_steps {
        let (_x, _error, new_history) = cowell_step(t, h, &history, accel).unwrap();
        history = new_history;
        t += h;
    }

    // 正确答案：x(t) = x₀ + v₀ t + ½ a₀ t² 为匀加速运动二次式
    let x_exact = x0 + v0 * t + 0.5 * a0 * t * t;
    let x_cowell = history[1][0]; // Cowell 位置存于 history[1]
    let err = (x_cowell - x_exact).abs();
    assert!(
        err < 1e-12,
        "Cowell 匀加速传播 {n_steps} 步后误差 {err:.2e} > 1e-12"
    );
}

/// 2D/3D 多分量匀加速也有类似精确性：每个分量独立为二次多项式。
#[test]
fn test_3d_constant_acceleration_exact_reproduction() {
    let a0 = [1.0, -2.0, 3.0];
    let accel =
        |_t: f64, _x: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> { Ok(a0.to_vec()) };

    let x0 = [1.0, 2.0, 3.0];
    let v0 = [0.5, -0.3, 1.2];
    let h: f64 = 0.005;

    let x_neg_h: Vec<f64> = (0..3)
        .map(|i| x0[i] - v0[i] * h + 0.5 * a0[i] * h * h)
        .collect();
    let mut history: Vec<Vec<f64>> = vec![x_neg_h, x0.to_vec()];
    for _ in 0..COWELL_HISTORY_LEN - 2 {
        history.push(a0.to_vec());
    }

    let n_steps = 200;
    let mut t = 0.0;
    for _ in 0..n_steps {
        let (_x, _error, new_history) = cowell_step(t, h, &history, accel).unwrap();
        history = new_history;
        t += h;
    }

    for i in 0..3 {
        let exact = x0[i] + v0[i] * t + 0.5 * a0[i] * t * t;
        let cowell_x = history[1][i];
        let err = (cowell_x - exact).abs();
        assert!(
            err < 1e-11,
            "3D 分量 {i}: Cowell {} vs 精确 {exact}, 误差 {err:.2e} > 1e-11",
            cowell_x
        );
    }
}

/// Cowell 历史缓冲长度常量检验：确保 8 个加速度采样 + 2 个位置采样 = 10。
#[test]
fn test_cowell_history_len_constant() {
    assert_eq!(
        COWELL_HISTORY_LEN, 10,
        "Cowell 历史缓冲长度应为 10（2 pos + 8 accel）"
    );
}
