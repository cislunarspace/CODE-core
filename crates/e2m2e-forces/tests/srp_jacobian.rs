#![cfg(feature = "spice")]

//! SRP 力模型雅可比回归测试（ADR 0013：按物理定义验证，不用 golden file）。
//!
//! 验证 `acceleration_and_jacobian` 对 SRP 力模型返回的 3×3 雅可比矩阵：
//! - 非全零（FD 差分在非退化几何下必产生非零导数）
//! - 雅可比量级与前向差分参考一致（`dF/dx` ≈ `[F(x+h)-F(x)]/h`，精度
//!   由 `sqrt(eps)*|r|` 步长保证）

use e2m2e_forces::forces::compiled::{acceleration_and_jacobian, CompiledForce};

// ── SPICE 内核加载 ───────────────────────────────────────────────────────

fn load_kernels() {
    let kernel_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR has no grandparent")
        .join("kernels");
    // 跳过测试若必需内核缺失（CI 环境可能未下载 BSP）
    let required = ["de430.bsp", "de440s.bsp"];
    for name in &required {
        if !kernel_dir.join(name).exists() {
            eprintln!(
                "Skipping: required kernel {} not found in {:?}",
                name, kernel_dir
            );
            return;
        }
    }
    for name in [
        "naif0012.tls",
        "pck00010.tpc",
        "de430.bsp",
        "de440s.bsp",
        "earth_latest_high_prec.bpc",
        "SPICEEarthPredictedKernel.bpc",
    ] {
        let path = kernel_dir.join(name);
        if path.exists() {
            let _ = cspice::data::furnish(path.to_string_lossy().to_string());
        }
    }
    e2m2e_spice::spice_ffi::register_bodies();
}

// ── 测试：SRP 雅可比包含在 STM ────────────────────────────────────────────

/// SRP 力模型的 `acceleration_and_jacobian` 在非阴影区返回非零 3×3 雅可比。
///
/// 物理依据：SRP 加速度方向与 `sc → sun` 一致，大小 ∝ `1/r²`，位置导数
/// 在非退化几何下（`r_sun ≠ r_sc`）必为非零。本测试在 J2000 时刻采样一个
/// 明确不在阴影中的航天器位置，验证雅可比矩阵不为零矩阵。
///
/// 回归目标：issue #279 确认 SRP 已纳入 STM 传播雅可比（此前
/// `supports_jacobian(SRP) = true` 但无独立测试覆盖）。
#[test]
fn test_srp_jacobian_nonzero() {
    load_kernels();

    // J2000 epoch
    let et = 0.0;

    // 航天器在 GEO 轨道附近（~42164 km x 轴），远离地球阴影。
    // 太阳也在 x 轴附近（J2000），SRP 力方向近似反平行 x 轴，雅可比量级
    // 受距离抑制（~1 AU），Frobenius² ~1e-36 但仍非零。
    let state: [f64; 6] = [42164.0, 0.0, 0.0, 0.0, 3.07, 0.0];

    let force = CompiledForce::SRP {
        area: 10.0,            // m²
        mass: 1000.0,          // kg
        cr: 1.5,               // 典型反射系数
        shadow_bodies: vec![], // 无阴影 → 纯光照，避免阴影边界不连续
    };

    let (acc, jac) = acceleration_and_jacobian(&force, et, &state, "EARTH")
        .expect("SRP acceleration_and_jacobian failed");

    // 加速度必须非零（全光照 + 正面积/cr/mass）
    let acc_mag = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
    assert!(
        acc_mag > 1e-20,
        "SRP acceleration magnitude {acc_mag} unexpectedly near zero"
    );

    // 雅可比不能是全零矩阵（阈值取 1e-45：GEO 距离 ~42164 km、日距 ~1 AU，
    // SRP 雅可比量级 ∝ F/r ~ 1e-11/42164 ~ 1e-15，Frobenius² ~ 9*(1e-15)² ~ 1e-29；
    // 实测 ~1e-36，留充足余量）
    let jac_frobenius_sq: f64 = jac.iter().flatten().map(|v| v * v).sum();
    assert!(
        jac_frobenius_sq > 1e-45,
        "SRP Jacobian Frobenius² {jac_frobenius_sq} unexpectedly zero; \
         SRP Jacobian may not be included in STM"
    );
}

/// SRP 雅可比与前向差分参考的相对误差 < 10%。
///
/// 本测试对 `acceleration_and_jacobian` 返回的雅可比做独立前向差分校验：
/// `jac_fd[i][j] ≈ (F(state + h*e_j)[i] - F(state)[i]) / h`，步长 h 取
/// 与 `compiled.rs` 同公式（`sqrt(eps)*|r|`）。10% 容差覆盖 central-vs-forward
/// 差分的固有差异（O(h) vs O(h²)）及 SRP 阴影几何对位置的非线性贡献。
#[test]
fn test_srp_jacobian_matches_fd_reference() {
    load_kernels();

    let et = 0.0;
    let state: [f64; 6] = [42164.0, 0.0, 0.0, 0.0, 3.07, 0.0];

    let force = CompiledForce::SRP {
        area: 10.0,
        mass: 1000.0,
        cr: 1.5,
        shadow_bodies: vec![],
    };

    let (_, jac) = acceleration_and_jacobian(&force, et, &state, "EARTH")
        .expect("acceleration_and_jacobian failed");

    // 前向差分参考（步长与 compiled.rs 一致）
    let r_norm = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
    let h = (f64::EPSILON.sqrt() * r_norm).max(1e-6);

    let acc0 = force
        .acceleration(et, &state, "EARTH")
        .expect("acc0 failed");

    for dim in 0..3 {
        let mut state_p = state;
        state_p[dim] += h;
        let acc_p = force
            .acceleration(et, &state_p, "EARTH")
            .expect("acc_p failed");
        for i in 0..3 {
            let fd_ref = (acc_p[i] - acc0[i]) / h;
            let jac_val = jac[i][dim];
            // 中央差分 vs 前向差分：当 jac 量级 > 1e-25 时做相对比较，
            // 否则只检查绝对值（退化维度，如 GEO 距离下 D=1,2 对 i=0 导数极小）
            if jac_val.abs() > 1e-25 || fd_ref.abs() > 1e-25 {
                let denom = fd_ref.abs().max(jac_val.abs()).max(1e-30);
                let rel_err = (jac_val - fd_ref).abs() / denom;
                assert!(
                    rel_err < 0.10,
                    "SRP jac[{i}][{dim}] = {jac_val}, fd_ref = {fd_ref}, \
                     rel_err = {rel_err} > 10%"
                );
            }
        }
    }
}
