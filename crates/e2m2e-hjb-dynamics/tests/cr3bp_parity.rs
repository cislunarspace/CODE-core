//! CR3BP Hamiltonian 与 e2m2e-forces 解析动力学的对拍（issue #497 验收：
//! 零控向量场与 propagate_cr3bp_py 所用 CR3BP 动力学逐点一致）。
//!
//! propagate_cr3bp_py 的动力学内核是 e2m2e-forces 的 `cr3bp_eom`
//! （6 维，含 z 方向）。平面模型应与其 z = vz = 0 截面逐位一致。

use e2m2e_forces::cr3bp::{cr3bp_eom, propagate_cr3bp};
use e2m2e_hjb_dynamics::Cr3bpSynodic;

const MU_EARTH_MOON: f64 = 0.01215;

fn add_scaled(state: [f64; 4], k: [f64; 4], scale: f64) -> [f64; 4] {
    [
        state[0] + scale * k[0],
        state[1] + scale * k[1],
        state[2] + scale * k[2],
        state[3] + scale * k[3],
    ]
}

/// 随机（确定性伪随机）状态下，平面向量场等于 cr3bp_eom 的平面切片。
#[test]
fn vector_field_matches_cr3bp_eom_planar_slice() {
    let h = Cr3bpSynodic::new(MU_EARTH_MOON, 0.5, 0.1);
    for k in 0..200 {
        let s = |i: usize| ((k * 13 + i * 29) % 101) as f64 / 50.0 - 1.0;
        let state4 = [s(0) + 0.5, s(1), s(2), s(3)];
        let state6 = [state4[0], state4[1], 0.0, state4[2], state4[3], 0.0];
        let f4 = h.vector_field(state4);
        let f6 = cr3bp_eom(MU_EARTH_MOON, &state6);
        let planar6 = [f6[0], f6[1], f6[3], f6[4]];
        for d in 0..4 {
            assert!(
                (f4[d] - planar6[d]).abs() < 1e-14,
                "状态 {state4:?} 第 {d} 维：{} vs {}",
                f4[d],
                planar6[d]
            );
        }
    }
}

/// μ 取值敏感性：不同 μ 下两实现仍逐项一致（防止参数位置写反）。
#[test]
fn vector_field_parity_across_mu() {
    for mu in [0.01215, 0.1, 0.3] {
        let h = Cr3bpSynodic::new(mu, 0.5, 0.1);
        let state4 = [0.8, 0.2, -0.1, 0.4];
        let state6 = [state4[0], state4[1], 0.0, state4[2], state4[3], 0.0];
        let f4 = h.vector_field(state4);
        let f6 = cr3bp_eom(mu, &state6);
        assert!((f4[2] - f6[3]).abs() < 1e-14, "μ = {mu}");
        assert!((f4[3] - f6[4]).abs() < 1e-14, "μ = {mu}");
    }
}

/// 无控轨迹对拍（issue #497 验收：轨道周期对照）。同一初值传播一个会合
/// 周期 2π：本 crate 四维向量场用固定步长 RK4 积分，参考为
/// propagate_cr3bp（PD78 自适应，传播器实际使用的动力学路径），
/// 末态应一致。
#[test]
fn uncontrolled_trajectory_matches_propagator() {
    let h = Cr3bpSynodic::new(MU_EARTH_MOON, 0.5, 0.1);
    let state0 = [0.8, 0.0, 0.0, 0.4];
    let t_final = 2.0 * std::f64::consts::PI;

    let state6 = [state0[0], state0[1], 0.0, state0[2], state0[3], 0.0];
    let reference = propagate_cr3bp(
        MU_EARTH_MOON,
        (0.0, t_final),
        &[t_final],
        &state6,
        1e-12,
        1e-12,
        None,
        None,
    )
    .expect("参考传播应成功");
    let ref_state = reference.states.last().expect("至少一个输出点");

    // 固定步长 RK4，步长 2π/100000 ≈ 6.3e-5，全局截断误差远低于容差。
    let steps = 100_000usize;
    let dt = t_final / steps as f64;
    let mut s = state0;
    for _ in 0..steps {
        let k1 = h.vector_field(s);
        let k2 = h.vector_field(add_scaled(s, k1, dt / 2.0));
        let k3 = h.vector_field(add_scaled(s, k2, dt / 2.0));
        let k4 = h.vector_field(add_scaled(s, k3, dt));
        for d in 0..4 {
            s[d] += dt * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]) / 6.0;
        }
    }

    let planar_ref = [ref_state[0], ref_state[1], ref_state[3], ref_state[4]];
    for d in 0..4 {
        assert!(
            (s[d] - planar_ref[d]).abs() < 1e-8,
            "第 {d} 维：RK4 {} vs PD78 {}",
            s[d],
            planar_ref[d]
        );
    }
}

fn rk4_step(h: &Cr3bpSynodic, s: [f64; 4], dt: f64) -> [f64; 4] {
    let k1 = h.vector_field(s);
    let k2 = h.vector_field(add_scaled(s, k1, dt / 2.0));
    let k3 = h.vector_field(add_scaled(s, k2, dt / 2.0));
    let k4 = h.vector_field(add_scaled(s, k3, dt));
    let mut out = s;
    for d in 0..4 {
        out[d] += dt * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]) / 6.0;
    }
    out
}

/// 从初值传播到下一次穿越 x 轴（y 变号），步内二分细化穿越点，
/// 返回 (τ, 穿越状态)。t_max 内无穿越返回 None。
fn to_x_axis_crossing(
    h: &Cr3bpSynodic,
    s0: [f64; 4],
    dt: f64,
    t_max: f64,
) -> Option<(f64, [f64; 4])> {
    let mut prev = s0;
    let mut t = 0.0;
    while t < t_max {
        let next = rk4_step(h, prev, dt);
        t += dt;
        if prev[1] * next[1] < 0.0 {
            // 在 [prev, next] 这一步内二分 y = 0 的位置。
            let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                let probe = rk4_step(h, prev, mid * dt);
                if prev[1] * probe[1] <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let frac = 0.5 * (lo + hi);
            return Some((t - dt + frac * dt, rk4_step(h, prev, frac * dt)));
        }
        prev = next;
    }
    None
}

/// 真周期轨道的周期对照（issue #497 验收）。镜像对称打靶：x 轴初值
/// (x0, 0, 0, vy0)，扫描加二分调整 vy0 使下一次过 x 轴时 vx = 0，
/// 得周期 T = 2τ（镜像定理）。随后以同初值用 propagate_cr3bp（PD78）
/// 传播 T，应回到初态——同时验证本 crate 动力学的轨道周期与传播器一致。
/// 全部数据在测试内自生，不依赖外部文献初值。
#[test]
fn periodic_orbit_period_matches_propagator() {
    let h = Cr3bpSynodic::new(MU_EARTH_MOON, 0.5, 0.1);
    let dt = 1e-4;
    let t_max = 4.0 * std::f64::consts::PI;
    let x0 = 1.1;

    let vx_at_crossing = |vy0: f64| -> Option<f64> {
        to_x_axis_crossing(&h, [x0, 0.0, 0.0, vy0], dt, t_max).map(|(_, s)| s[2])
    };

    // 扫描 vy0 找 vx 变号区间，再二分。
    let mut bracket: Option<(f64, f64)> = None;
    let mut prev: Option<(f64, f64)> = None;
    for i in 1..=15 {
        let vy0 = 0.05 * i as f64;
        if let Some(vx) = vx_at_crossing(vy0) {
            if let Some((pv, pvx)) = prev {
                if pvx * vx < 0.0 {
                    bracket = Some((pv, vy0));
                    break;
                }
            }
            prev = Some((vy0, vx));
        }
    }
    let (mut lo, mut hi) = bracket.expect("扫描区间内应有 vx 变号");
    let mut flo = vx_at_crossing(lo).expect("lo 应有穿越");
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let fm = vx_at_crossing(mid).expect("mid 应有穿越");
        if flo * fm <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
    }
    let vy0 = 0.5 * (lo + hi);
    let (tau, crossing) = to_x_axis_crossing(&h, [x0, 0.0, 0.0, vy0], dt, t_max).expect("应有穿越");
    assert!(crossing[2].abs() < 1e-8, "打靶残差 vx = {}", crossing[2]);
    let period = 2.0 * tau;

    // 自洽：本 crate 传播一个周期应回到初态。
    let mut s = [x0, 0.0, 0.0, vy0];
    let mut t = 0.0;
    while t < period {
        let step = dt.min(period - t);
        s = rk4_step(&h, s, step);
        t += step;
    }
    for d in 0..4 {
        assert!(
            (s[d] - [x0, 0.0, 0.0, vy0][d]).abs() < 1e-6,
            "自传播一周后第 {d} 维未回到初态"
        );
    }

    // 对拍：传播器以同初值传播 T，回到初态。
    let state6 = [x0, 0.0, 0.0, 0.0, vy0, 0.0];
    let result = propagate_cr3bp(
        MU_EARTH_MOON,
        (0.0, period),
        &[period],
        &state6,
        1e-12,
        1e-12,
        None,
        None,
    )
    .expect("参考传播应成功");
    let end = result.states.last().expect("至少一个输出点");
    for (d, &expect) in state6.iter().enumerate() {
        assert!(
            (end[d] - expect).abs() < 1e-5,
            "传播器传播 T = {period} 后第 {d} 维：{} vs {expect}",
            end[d]
        );
    }
}
