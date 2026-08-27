//! USSA76 分段指数大气密度模型（Rust 实现）。
//!
//! 从 Python `atmosphere.py` 迁移，纯数学（高度 → 密度），不依赖 SPICE，
//! 为后续 drag 力 Rust 化铺路；drag.rs 将内部调用 [`density`]，不暴露 Python。
//!
//! - 14 段 USSA76 断点（[`USSA76_BREAKPOINTS`]），层内自洽标高
//!   `H = Δh / ln(ρ₀/ρ₁)`，保证层间密度连续且单调递减。
//! - F10.7 太阳射电通量、Ap 地磁指数通过线性乘法因子对基准密度做一阶修正。
//!
//! 与 Python `ExponentialAtmosphere.density` 同公式同断点，密度逐位一致。

use std::sync::LazyLock;

/// US Standard Atmosphere 1976 断点密度（km, kg/m³）。
///
/// 数据来源：USSA76 标准大气表，覆盖 0–1000 km。移植自 `atmosphere.py:9-25`。
const USSA76_BREAKPOINTS: [(f64, f64); 15] = [
    (0.0, 1.225e0),
    (25.0, 4.048e-2),
    (50.0, 1.057e-3),
    (75.0, 3.313e-5),
    (100.0, 5.604e-7),
    (150.0, 2.384e-9),
    (200.0, 2.541e-10),
    (300.0, 1.916e-11),
    (400.0, 2.803e-12),
    (500.0, 5.215e-13),
    (600.0, 1.189e-13),
    (700.0, 3.381e-14),
    (800.0, 1.137e-14),
    (900.0, 4.390e-15),
    (1000.0, 1.879e-15),
];

/// 模型上限高度（km），不低于此高度才有非零密度。对应 `_CEILING_ALTITUDE_KM`。
const CEILING_ALTITUDE_KM: f64 = 1000.0;

/// 默认 F10.7 太阳射电通量（sfu，中等太阳活动）。对应 `_DEFAULT_F107`。
const DEFAULT_F107: f64 = 150.0;

/// 默认 Ap 地磁指数（中等地磁活动）。对应 `_DEFAULT_AP`。
const DEFAULT_AP: f64 = 15.0;

/// F10.7 敏感系数。对应 `_F107_SENSITIVITY`。
const F107_SENSITIVITY: f64 = 0.5;

/// Ap 敏感系数。对应 `_AP_SENSITIVITY`。
const AP_SENSITIVITY: f64 = 0.1;

/// 一段高度层：(基准高度 km, 基准密度 kg/m³, 自洽标高 km)。
struct Layer {
    h_base: f64,
    rho_base: f64,
    scale_height: f64,
}

/// 从断点密度推导自洽标高，构造连续分段层表。
///
/// 每层标高 `H = Δh / ln(ρ₀/ρ₁)` 确保层间密度连续且单调递减。
/// 移植自 `atmosphere.py:28-41`。
fn build_layers() -> Vec<Layer> {
    let mut layers = Vec::with_capacity(USSA76_BREAKPOINTS.len() - 1);
    for window in USSA76_BREAKPOINTS.windows(2) {
        let (h0, rho0) = window[0];
        let (h1, rho1) = window[1];
        let scale_height = (h1 - h0) / (rho0 / rho1).ln();
        layers.push(Layer {
            h_base: h0,
            rho_base: rho0,
            scale_height,
        });
    }
    layers
}

/// 缓存的层表：断点为常量，标高只需算一次（drag 每步都会调 [`density`]）。
static LAYERS: LazyLock<Vec<Layer>> = LazyLock::new(build_layers);

/// 查找包含给定高度的层（最大 `h_base <= altitude`）。
///
/// 二分实现，结果与 Python `_lookup_layer` 的线性扫描一致：返回基高度不超过
/// `altitude` 的最后一层。调用方保证 `altitude ∈ [0, CEILING_ALTITUDE_KM)`，
/// 故 `LAYERS[0].h_base == 0 <= altitude`，`partition_point` 至少返回 1。
fn lookup_layer(altitude: f64) -> &'static Layer {
    let idx = LAYERS.partition_point(|l| l.h_base <= altitude);
    &LAYERS[idx - 1]
}

/// 计算指定高度处的大气密度（kg/m³）。
///
/// 层内 `ρ(h) = ρ₀ · exp(-(h - h₀) / H)`，再乘 F10.7/Ap 一阶修正因子。
/// 高度超出模型范围时：`h >= 1000 km` 返回 0（阻力可忽略），`h < 0` 钳到
/// 0 km（用地表密度，避免负高度导致 exp 爆炸）。
///
/// `f107`/`ap` 为构造时固定的太阳活动参数（对应 Python `ExponentialAtmosphere`
/// 的 `self._f107`/`self._ap`），传入默认值 `150.0`/`15.0` 时修正因子为 1。
///
/// 移植自 `ExponentialAtmosphere.density`（`atmosphere.py:80-98`）。
pub fn density(altitude_km: f64, f107: f64, ap: f64) -> f64 {
    let h = altitude_km.max(0.0);
    if h >= CEILING_ALTITUDE_KM {
        return 0.0;
    }
    let layer = lookup_layer(h);
    let rho_ref = layer.rho_base * ((layer.h_base - h) / layer.scale_height).exp();
    rho_ref * solar_activity_factor(f107, ap)
}

/// F10.7 和 Ap 的一阶线性密度修正因子。
///
/// `f_factor = 1 + 0.5·(f107 − 150)/150`、`a_factor = 1 + 0.1·(ap − 15)/15`，
/// 两者相乘。移植自 `_solar_activity_factor`（`atmosphere.py:113-117`）。
fn solar_activity_factor(f107: f64, ap: f64) -> f64 {
    let f_factor = 1.0 + F107_SENSITIVITY * (f107 - DEFAULT_F107) / DEFAULT_F107;
    let a_factor = 1.0 + AP_SENSITIVITY * (ap - DEFAULT_AP) / DEFAULT_AP;
    f_factor * a_factor
}

#[cfg(test)]
mod tests {
    // 黄金值表由 Python `%.17e` 全精度打印生成：18 位有效数字中末位超出
    // f64 精度、解析值不变，但保留它可使重新生成的表与原表逐字节可比对。
    // 属有意为之，豁免 excessive_precision。
    #![allow(clippy::excessive_precision)]
    use super::*;

    /// 密度跨 15 个数量级（1.2 到 1e-15），纯 atol 对 <1e-12 的密度失效
    /// （错层时差值仍可能 < atol 而漏检）。故对非零期望用相对容差 rtol=1e-12，
    /// 对零期望（ceiling 钳制）要求严格相等——这是"逐位一致"在宽量程下的
    /// 正确实现，比纯 atol 更严。同公式同 libm（exp/ln），实际差为 0。
    fn assert_close(altitude: f64, got: f64, want: f64) {
        if want == 0.0 {
            assert_eq!(got, 0.0, "h={altitude}: ceiling 应严格返 0, got {got}");
        } else {
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel <= 1e-12,
                "h={altitude}: got={got:e} want={want:e} 相对误差={rel:e}",
            );
        }
    }

    /// 默认参数 (f107=150, ap=15) 下各高度密度，逐点对照 Python
    /// `ExponentialAtmosphere().density(h)`（取值时用 `%.17e` 全精度打印）。
    /// 覆盖：负高度钳制、海平面、每段断点、层内点、接近 ceiling、ceiling 与超限。
    #[test]
    fn density_default_matches_python() {
        let cases: &[(f64, f64)] = &[
            (-10.0, 1.22500000000000009e0),
            (0.0, 1.22500000000000009e0),
            (1.0, 1.06880940150163095e0),
            (10.0, 3.13168343238207636e-1),
            (25.0, 4.04800000000000021e-2),
            (50.0, 1.05700000000000000e-3),
            (75.0, 3.31300000000000030e-5),
            (100.0, 5.60399999999999962e-7),
            (125.0, 3.65512462167844574e-8),
            (150.0, 2.38400000000000004e-9),
            (200.0, 2.54099999999999981e-10),
            (300.0, 1.91600000000000016e-11),
            (400.0, 2.80300000000000007e-12),
            (500.0, 5.21500000000000050e-13),
            (600.0, 1.18900000000000003e-13),
            (700.0, 3.38100000000000031e-14),
            (800.0, 1.13699999999999998e-14),
            (900.0, 4.38999999999999994e-15),
            (999.999, 1.87901594506448371e-15),
            (1000.0, 0.0),
            (1500.0, 0.0),
        ];
        for &(h, want) in cases {
            let got = density(h, DEFAULT_F107, DEFAULT_AP);
            assert_close(h, got, want);
        }
    }

    /// F10.7/Ap 修正：h=400 km 下多组 (f107, ap) 对照 Python。
    /// 默认 (150,15) 因子为 1，等于基准密度 2.803e-12。
    #[test]
    fn solar_activity_matches_python() {
        // (f107, ap, density at h=400 km)
        let cases: &[(f64, f64, f64)] = &[
            (150.0, 15.0, 2.80300000000000007e-12),
            (200.0, 15.0, 3.27016666666666715e-12),
            (100.0, 15.0, 2.33583333333333339e-12),
            (150.0, 50.0, 3.45703333333333350e-12),
            (150.0, 5.0, 2.61613333333333332e-12),
            (200.0, 50.0, 4.03320555555555602e-12),
        ];
        for &(f107, ap, want) in cases {
            let got = density(400.0, f107, ap);
            assert_close(400.0, got, want);
        }
    }

    /// 层表标高对照 Python `_LAYERS`（全精度），验证 `build_layers` 逐位一致。
    #[test]
    fn scale_heights_match_python() {
        let want: &[(f64, f64)] = &[
            (0.0, 7.33161889232260844e0),
            (25.0, 6.85800817050232503e0),
            (50.0, 7.21969372435890833e0),
            (75.0, 6.12813636914091564e0),
            (100.0, 9.15772541189510569e0),
            (150.0, 2.23333216847847424e1),
            (200.0, 3.86861694397860560e1),
            (300.0, 5.20254956046351182e1),
            (400.0, 5.94623552955645351e1),
            (500.0, 6.76394815561372269e1),
            (600.0, 7.95212068230541291e1),
            (700.0, 9.17617823533993970e1),
            (800.0, 1.05080750909812437e2),
            (900.0, 1.17842607309740302e2),
        ];
        let layers = build_layers();
        assert_eq!(layers.len(), want.len());
        for (layer, &(h0, sh)) in layers.iter().zip(want) {
            assert!(
                (layer.h_base - h0).abs() <= 0.0,
                "h_base 不匹配: {} vs {}",
                layer.h_base,
                h0
            );
            let rel = (layer.scale_height - sh).abs() / sh.abs();
            assert!(
                rel <= 1e-12,
                "H(h0={}) 解析={:e} want={:e} 相对误差={:e}",
                h0,
                layer.scale_height,
                sh,
                rel
            );
        }
    }

    /// 断点处密度严格等于该层基准密度（`exp(0) = 1`，且 `lookup_layer` 选
    /// 基高度等于断点的层），验证层基密度与选层逻辑。
    #[test]
    fn breakpoint_density_equals_base() {
        // 跳过首断点（h=0 已在 default 用例覆盖）与末断点（h=1000 返 0）。
        for &(h0, rho0) in &USSA76_BREAKPOINTS[1..USSA76_BREAKPOINTS.len() - 1] {
            let got = density(h0, DEFAULT_F107, DEFAULT_AP);
            assert_eq!(
                got.to_bits(),
                rho0.to_bits(),
                "断点 h0={h0}: 密度 {got:e} 应逐位等于基准密度 {rho0:e}"
            );
        }
    }

    /// 层界连续性：断点两侧（±1e-9 km）密度逼近断点密度，相对差 < 1e-7。
    /// 由自洽标高 `H = Δh/ln(ρ₀/ρ₁)` 保证：下层在 h=h₀ 恰等于上层 ρ₀。
    #[test]
    fn layer_continuity_at_breakpoints() {
        let eps = 1e-9_f64;
        for &(h0, _rho0) in &USSA76_BREAKPOINTS[1..USSA76_BREAKPOINTS.len() - 1] {
            let d_mid = density(h0, DEFAULT_F107, DEFAULT_AP);
            let d_below = density(h0 - eps, DEFAULT_F107, DEFAULT_AP);
            let d_above = density(h0 + eps, DEFAULT_F107, DEFAULT_AP);
            assert!(
                (d_below - d_mid).abs() / d_mid.abs() < 1e-7,
                "h0={h0}: 下方 {d_below:e} 与断点 {d_mid:e} 不连续"
            );
            assert!(
                (d_above - d_mid).abs() / d_mid.abs() < 1e-7,
                "h0={h0}: 上方 {d_above:e} 与断点 {d_mid:e} 不连续"
            );
        }
    }
}
