//! NRLMSISE-00 大气密度模型（Picone et al. 2002 规格的 Rust 实现）。
//!
//! 服务地球停泊弧段的高保真阻力建模：输入不再是纯高度，而是历元（年积日 +
//! UT 秒）、WGS84 大地经纬度与空间天气（F10.7 / 81 日平均 / Ap 史）。
//!
//! # 实现来源与误差口径
//!
//! 按发表的 NRLMSISE-00 规格自行实现，系数表取自 NRL 公有领域模型数据
//! （见 [`coefficients`] 文件头的逐点转录说明）。与官方参考实现相比：
//!
//! - 数值路径（`spline` / `densu` / `densm` / `globe7` / `glob7s` / `gts7`）
//!   逐句对应，含运算顺序；差异仅在浮点舍入量级（相对 < 1e-12）。
//! - 官方实现用文件级静态变量（FORTRAN COMMON）保存跨调用的中间状态（`plg`、
//!   `dfa`、`apdf`、`apt`、局地时三角量）。本实现把这份状态收进每次调用新建的
//!   [`Ctx`]，等价于参考实现的"首次调用"行为；参考实现在 72.5 km 以下依赖上一次
//!   调用的残留值，本实现不复制这处未定义行为（对阻力应用的高度域无影响）。
//! - 只支持标准开关集：switch 0 = 0（内部用 cgs 密度，[`density`] 换算为 kg/m³），
//!   switch 1–23 全开；switch 9 取 `+1`（标量 Ap）或 `−1`（Ap 史）由
//!   `storm_time` 决定。参考实现里所有 `swc[i] = 1`、`|sw[i]| = 1` 的乘法与
//!   分支已按恒等式去掉。
//!
//! # 高度域
//!
//! `altitude_km < 0` 钳到 0；`altitude_km >= 1000` 直接返回密度 0 与温度 0
//! （与 [`crate::atmosphere`] 的上限约定一致，阻力在 1000 km 以上可忽略）。

mod coefficients;

use coefficients::{PAVGM, PD, PDL, PDM, PMA, PS, PT, PTL, PTM};

/// 模型上限高度（km）：不低于此高度才有非零密度。
const CEILING_ALTITUDE_KM: f64 = 1000.0;

/// 度 → 弧度（参考实现 `dgtr`）。
const DGTR: f64 = 1.74533E-2;

/// 2π/365，年积日相位用（参考实现 `dr`）。
const DR: f64 = 1.72142E-2;

/// 2π/24，局地时相位用（参考实现 `hr`）。
const HR: f64 = 0.2618;

/// 2π/86400，UT 秒相位用（参考实现 `sr`）。
const SR: f64 = 7.2722E-5;

/// 通用气体常数（参考实现内部的 `rgas`）。
const RGAS: f64 = 831.4;

/// 分子量（amu）→ 质量的换算常数，配合 cm⁻³ 数密度得 g/cm³。
const AMU_G: f64 = 1.66E-24;

/// g/cm³ → kg/m³。
const G_CM3_TO_KG_M3: f64 = 1000.0;

/// gts7 的 spline 节点高度（km），索引 0 由 `pdl[1][15]` 覆盖。
const ZN1_BASE: [f64; 5] = [120.0, 110.0, 100.0, 90.0, 72.5];

/// gtd7 中层/上平流层节点高度（km）。
const ZN2: [f64; 4] = [72.5, 55.0, 45.0, 32.5];

/// gtd7 平流层/对流层节点高度（km）。
const ZN3: [f64; 5] = [32.5, 20.0, 15.0, 10.0, 0.0];

/// gtd7 湍流层顶混合过渡起始高度（km）。
const ZMIX: f64 = 62.5;

/// 各物种 Bates 剖面温度梯度参数。
const ALPHA: [f64; 9] = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0];

/// 各物种启用湍流层修正的高度上限（km）。
const ALTL: [f64; 8] = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0];

/// NRLMSISE-00 输入。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nrlmsise00Input {
    /// 年积日，1..=366。
    pub day_of_year: u16,
    /// UTC 日内秒，0..86400。
    pub ut_seconds: f64,
    /// WGS84 椭球面以上大地高度（km）。
    pub altitude_km: f64,
    /// WGS84 大地纬度（deg）。
    pub geodetic_lat_deg: f64,
    /// 大地经度（deg，东经为正）。
    pub geodetic_lon_deg: f64,
    /// 前一日 F10.7 太阳射电通量（sfu）。
    pub f107_daily: f64,
    /// 81 日滑动平均 F10.7（sfu）。
    pub f107_avg: f64,
    /// 3 小时分辨率地磁 Ap 指数史，元素语义见模块文档与 Python 侧 docstring。
    pub ap: [f64; 7],
}

/// NRLMSISE-00 输出。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nrlmsise00Output {
    /// drag 有效总质量密度（kg/m³，含异常氧，即 `gtd7d` 的 `d[5]`）。
    pub density_kg_m3: f64,
    /// 该高度处温度（K）。
    pub temperature_k: f64,
}

/// 计算大气密度与温度。
///
/// `storm_time` 只决定地磁 Ap 的输入形态：`false` 用标量日 Ap（`ap[0]`），
/// `true` 用 3 小时分辨率的 Ap 史（`ap[1..]`）。
///
/// 总质量密度始终取 `gtd7d` 的"drag 有效总质量密度"口径（含 500 km 以上不可
/// 忽略的异常氧贡献）。这与 nyx 的公开对拍夹具一致；`gtd7` 的不含异常氧口径
/// 不对外暴露（阻力建模只关心前者）。
pub fn density(input: &Nrlmsise00Input, storm_time: bool) -> Nrlmsise00Output {
    if input.altitude_km >= CEILING_ALTITUDE_KM {
        return Nrlmsise00Output {
            density_kg_m3: 0.0,
            temperature_k: 0.0,
        };
    }
    let mut ctx = Ctx::new();
    ctx.storm_time = storm_time;
    ctx.gtd7(input, input.altitude_km.max(0.0));
    Nrlmsise00Output {
        density_kg_m3: ctx.mass_density_drag() * G_CM3_TO_KG_M3,
        temperature_k: ctx.t[1],
    }
}

/// Ap 史是否应启用暴时先验（3 小时分辨率）。
///
/// 判定：7 元中只要有元素与 `ap[0]` 不同，即视为暴时历史；全等（含标量广播）
/// 即静态空间天气，只用当日 Ap。这是模型的**输入语义**而非调用方选项，因此
/// drag 力解析与直接查询绑定共用同一实现——两处各自演进会让同一个 `ap` 数组
/// 在两条路径上给出不同密度（见 ADR 0049 决策 4）。
pub fn storm_time_from_ap(ap: &[f64; 7]) -> bool {
    ap[1..].iter().any(|a| a != &ap[0])
}

/// 局地太阳时（秒，折入 `[0, 86400)`）。
///
/// 模型要求的是**平局地太阳时**：`ut_seconds + lon_deg·240`（经度每度 4 分钟），
/// 而非含时差的真太阳时；后者会位移密度日侧隆起，破坏模型球谐拟合的标定。
pub fn local_solar_time_seconds(ut_seconds: f64, lon_deg: f64) -> f64 {
    (ut_seconds + lon_deg * 240.0).rem_euclid(86400.0)
}

/// ET → `(年, 年积日, UTC 日内秒)`。
///
/// 先查星历预采样缓存（[`e2m2e_spice::ephem_cache::lookup_utc_calendar`]）：传播
/// 打靶的并行区（ADR 0016 的 `StrictGuard`）内不走 cspice，改为查内存表插值。
/// 缓存未启用时回退 `et2utc`("ISOC", 3)——毫秒分辨率足够：1 ms 对应的密度变化
/// 远低于模型自身 15–20% 的不确定度。内核池为空时由 `et2utc` 返回明确错误，
/// 此处直接透传。
#[cfg(feature = "spice")]
pub fn et_to_utc_doy(et: f64) -> Result<(i32, u16, f64), e2m2e_spice::spice_ffi::SpiceFfiError> {
    if let Some(calendar) = e2m2e_spice::ephem_cache::lookup_utc_calendar(et)? {
        return Ok(calendar);
    }
    let utc_seconds = e2m2e_spice::spice_ffi::et_to_utc_seconds(et)?;
    Ok(e2m2e_spice::spice_ffi::utc_seconds_to_calendar(utc_seconds))
}

// ── 模型内部状态 ──────────────────────────────────────────────────────────
//
// 对应参考实现的文件级静态变量；每次 [`density`] 调用新建，保证可重入
// （drag 在传播中可能被 Rayon 并行调用，不能有全局可变状态）。

struct Ctx {
    /// 本次评估是否使用 Ap 史（switch 9 = −1）。
    storm_time: bool,
    /// 纬度相关重力与有效地球半径（`glatf`）。
    gsurf: f64,
    re: f64,
    /// 物种数密度（cm⁻³，索引 5 为 g/cm³ 总质量密度）与温度（K）。
    d: [f64; 9],
    t: [f64; 2],
    /// 湍流层混合密度（`gts7` 内跨物种共享）。
    dm04: f64,
    dm16: f64,
    dm28: f64,
    dm32: f64,
    dm40: f64,
    dm01: f64,
    dm14: f64,
    /// 下层大气温度节点与端节点梯度。
    meso_tn1: [f64; 5],
    meso_tn2: [f64; 4],
    meso_tn3: [f64; 5],
    meso_tgn1: [f64; 2],
    meso_tgn2: [f64; 2],
    meso_tgn3: [f64; 2],
    /// `globe7` 副产物：F10.7 偏差、勒让德多项式、局地时三角量、磁活动因子。
    dfa: f64,
    plg: [[f64; 9]; 4],
    ctloc: f64,
    stloc: f64,
    c2tloc: f64,
    s2tloc: f64,
    c3tloc: f64,
    s3tloc: f64,
    apdf: f64,
    apt: [f64; 4],
}

impl Ctx {
    fn new() -> Self {
        Self {
            storm_time: false,
            gsurf: 0.0,
            re: 0.0,
            d: [0.0; 9],
            t: [0.0; 2],
            dm04: 0.0,
            dm16: 0.0,
            dm28: 0.0,
            dm32: 0.0,
            dm40: 0.0,
            dm01: 0.0,
            dm14: 0.0,
            meso_tn1: [0.0; 5],
            meso_tn2: [0.0; 4],
            meso_tn3: [0.0; 5],
            meso_tgn1: [0.0; 2],
            meso_tgn2: [0.0; 2],
            meso_tgn3: [0.0; 2],
            dfa: 0.0,
            plg: [[0.0; 9]; 4],
            ctloc: 0.0,
            stloc: 0.0,
            c2tloc: 0.0,
            s2tloc: 0.0,
            c3tloc: 0.0,
            s3tloc: 0.0,
            apdf: 0.0,
            apt: [0.0; 4],
        }
    }

    /// 重力加速度与有效地球半径随纬度的变化（参考实现 `glatf`）。
    fn glatf(&mut self, lat_deg: f64) {
        let c2 = (2.0 * DGTR * lat_deg).cos();
        self.gsurf = 980.616 * (1.0 - 0.0026373 * c2);
        self.re = 2.0 * self.gsurf / (3.085462E-6 + 2.27E-9 * c2) * 1.0E-5;
    }

    /// 标高（参考实现 `scalh`）。
    fn scalh(&self, alt: f64, xm: f64, temp: f64) -> f64 {
        let g = self.gsurf / (1.0 + alt / self.re).powi(2);
        RGAS * temp / (g * xm)
    }

    // ── gtd7 ────────────────────────────────────────────────────────────

    /// 从地表到外逸层的完整剖面（参考实现 `gtd7`）。
    fn gtd7(&mut self, input: &Nrlmsise00Input, alt: f64) {
        // switch 2 = 1 → 用输入纬度（否则参考实现固定 45°）。
        self.glatf(input.geodetic_lat_deg);

        let xmm = PDM[2][4];

        // 热层/中层（zn2[0] 以上）。
        self.gts7(input, alt.max(ZN2[0]));
        if alt >= ZN2[0] {
            return;
        }

        // 低中层/上平流层（zn3[0]–zn2[0]）与低平流层/对流层（zn3[0] 以下）：
        // 节点温度与端节点梯度由球谐展开给出。
        self.meso_tgn2[0] = self.meso_tgn1[1];
        self.meso_tn2[0] = self.meso_tn1[4];
        let (v1, v2, v3, v9) = (
            self.glob7s(&PMA[0], input),
            self.glob7s(&PMA[1], input),
            self.glob7s(&PMA[2], input),
            self.glob7s(&PMA[9], input),
        );
        self.meso_tn2[1] = PMA[0][0] * PAVGM[0] / (1.0 - v1);
        self.meso_tn2[2] = PMA[1][0] * PAVGM[1] / (1.0 - v2);
        self.meso_tn2[3] = PMA[2][0] * PAVGM[2] / (1.0 - v3);
        let tn2_3 = self.meso_tn2[3];
        self.meso_tgn2[1] =
            PAVGM[8] * PMA[9][0] * (1.0 + v9) * tn2_3 * tn2_3 / (PMA[2][0] * PAVGM[2]).powi(2);
        self.meso_tn3[0] = self.meso_tn2[3];

        if alt <= ZN3[0] {
            self.meso_tgn3[0] = self.meso_tgn2[1];
            let (w3, w4, w5, w6, w7) = (
                self.glob7s(&PMA[3], input),
                self.glob7s(&PMA[4], input),
                self.glob7s(&PMA[5], input),
                self.glob7s(&PMA[6], input),
                self.glob7s(&PMA[7], input),
            );
            self.meso_tn3[1] = PMA[3][0] * PAVGM[3] / (1.0 - w3);
            self.meso_tn3[2] = PMA[4][0] * PAVGM[4] / (1.0 - w4);
            self.meso_tn3[3] = PMA[5][0] * PAVGM[5] / (1.0 - w5);
            self.meso_tn3[4] = PMA[6][0] * PAVGM[6] / (1.0 - w6);
            let tn3_4 = self.meso_tn3[4];
            self.meso_tgn3[1] =
                PMA[7][0] * PAVGM[7] * (1.0 + w7) * tn3_4 * tn3_4 / (PMA[6][0] * PAVGM[6]).powi(2);
        }

        // 向 zn2[0] 以下全混合区的线性过渡。
        let dmc = if alt > ZMIX {
            1.0 - (ZN2[0] - alt) / (ZN2[0] - ZMIX)
        } else {
            0.0
        };
        let dm28m = self.dm28;
        let dz28 = self.d[2];

        // N2：混合密度 + 过渡修正。
        let dmr = self.d[2] / dm28m - 1.0;
        let (dens, _) = self.densm(alt, dm28m, xmm);
        self.d[2] = dens * (1.0 + dmr * dmc);

        // He。
        let dmr = self.d[0] / (dz28 * PDM[0][1]) - 1.0;
        self.d[0] = self.d[2] * PDM[0][1] * (1.0 + dmr * dmc);

        self.d[1] = 0.0; // O（72.5 km 以下为 0）
        self.d[8] = 0.0; // 异常氧

        // O2。
        let dmr = self.d[3] / (dz28 * PDM[3][1]) - 1.0;
        self.d[3] = self.d[2] * PDM[3][1] * (1.0 + dmr * dmc);

        // Ar。
        let dmr = self.d[4] / (dz28 * PDM[4][1]) - 1.0;
        self.d[4] = self.d[2] * PDM[4][1] * (1.0 + dmr * dmc);

        self.d[6] = 0.0; // H
        self.d[7] = 0.0; // N

        self.update_mass_density();

        // 该高度温度（xm = 0 → 只算温度）。
        let (_, tz) = self.densm(alt, 1.0, 0.0);
        self.t[1] = tz;
    }

    // ── gts7 ────────────────────────────────────────────────────────────

    /// 热层部分（参考实现 `gts7`，要求 `alt > 72.5 km`）。
    fn gts7(&mut self, input: &Nrlmsise00Input, alt: f64) {
        let zn1 = [
            PDL[1][15],
            ZN1_BASE[1],
            ZN1_BASE[2],
            ZN1_BASE[3],
            ZN1_BASE[4],
        ];
        self.d = [0.0; 9];

        // 外逸层温度。
        let tinf = if alt > zn1[0] {
            PTM[0] * PT[0] * (1.0 + self.globe7(&PT, input))
        } else {
            PTM[0] * PT[0]
        };
        self.t[0] = tinf;

        // 温度梯度。
        let g0v = if alt > zn1[4] {
            PTM[3] * PS[0] * (1.0 + self.globe7(&PS, input))
        } else {
            PTM[3] * PS[0]
        };
        let tlb = PTM[1] * (1.0 + self.globe7(&PD[3], input)) * PD[3][0];
        let s = g0v / (tinf - tlb);

        // 低热层温度节点（300 km 以上不显著）。
        if alt < 300.0 {
            let a = self.glob7s(&PTL[0], input);
            let b = self.glob7s(&PTL[1], input);
            let c = self.glob7s(&PTL[2], input);
            let d = self.glob7s(&PTL[3], input);
            self.meso_tn1[1] = PTM[6] * PTL[0][0] / (1.0 - a);
            self.meso_tn1[2] = PTM[2] * PTL[1][0] / (1.0 - b);
            self.meso_tn1[3] = PTM[7] * PTL[2][0] / (1.0 - c);
            self.meso_tn1[4] = PTM[4] * PTL[3][0] / (1.0 - d);
            let tn1_4 = self.meso_tn1[4];
            self.meso_tgn1[1] =
                PTM[8] * PMA[8][0] * (1.0 + self.glob7s(&PMA[8], input)) * tn1_4 * tn1_4
                    / (PTM[4] * PTL[3][0]).powi(2);
        } else {
            self.meso_tn1[1] = PTM[6] * PTL[0][0];
            self.meso_tn1[2] = PTM[2] * PTL[1][0];
            self.meso_tn1[3] = PTM[7] * PTL[2][0];
            self.meso_tn1[4] = PTM[4] * PTL[3][0];
            let tn1_4 = self.meso_tn1[4];
            self.meso_tgn1[1] = PTM[8] * PMA[8][0] * tn1_4 * tn1_4 / (PTM[4] * PTL[3][0]).powi(2);
        }

        // N2 在湍流层底的变化因子。
        let g28 = self.globe7(&PD[2], input);

        // 湍流层顶高度变化。
        let zhf = PDL[1][24]
            * (1.0
                + PDL[0][24]
                    * (DGTR * input.geodetic_lat_deg).sin()
                    * (DR * (f64::from(input.day_of_year) - PT[13])).cos());
        self.t[0] = tinf;
        let xmm = PDM[2][4];
        let z = alt;

        // N2。
        let db28 = PDM[2][0] * g28.exp() * PD[2][0];
        let (d2, _) = self.densu(z, db28, tinf, tlb, 28.0, ALPHA[2], PTM[5], s, &zn1);
        self.d[2] = d2;
        let zh28 = PDM[2][2] * zhf;
        let zhm28 = PDM[2][3] * PDL[1][5];
        let xmd = 28.0 - xmm;
        let (b28, _) = self.densu(zh28, db28, tinf, tlb, xmd, ALPHA[2] - 1.0, PTM[5], s, &zn1);
        if z <= ALTL[2] {
            let (dm28, _) = self.densu(z, b28, tinf, tlb, xmm, ALPHA[2], PTM[5], s, &zn1);
            self.dm28 = dm28;
            self.d[2] = dnet(self.d[2], dm28, zhm28, xmm, 28.0);
        }

        // He。
        let g4 = self.globe7(&PD[0], input);
        let db04 = PDM[0][0] * g4.exp() * PD[0][0];
        let (d0, _) = self.densu(z, db04, tinf, tlb, 4.0, ALPHA[0], PTM[5], s, &zn1);
        self.d[0] = d0;
        if z < ALTL[0] {
            let zh04 = PDM[0][2];
            let (b04, _) = self.densu(
                zh04,
                db04,
                tinf,
                tlb,
                4.0 - xmm,
                ALPHA[0] - 1.0,
                PTM[5],
                s,
                &zn1,
            );
            let (dm04, _) = self.densu(z, b04, tinf, tlb, xmm, 0.0, PTM[5], s, &zn1);
            self.dm04 = dm04;
            self.d[0] = dnet(self.d[0], dm04, zhm28, xmm, 4.0);
            // 地面混合比的修正。
            let rl = (b28 * PDM[0][1] / b04).ln();
            let zc04 = PDM[0][4] * PDL[1][0];
            let hc04 = PDM[0][5] * PDL[1][1];
            self.d[0] *= ccor(z, rl, hc04, zc04);
        }

        // O。
        let g16 = self.globe7(&PD[1], input);
        let db16 = PDM[1][0] * g16.exp() * PD[1][0];
        let (d1, _) = self.densu(z, db16, tinf, tlb, 16.0, ALPHA[1], PTM[5], s, &zn1);
        self.d[1] = d1;
        if z <= ALTL[1] {
            let zh16 = PDM[1][2];
            let (b16, _) = self.densu(
                zh16,
                db16,
                tinf,
                tlb,
                16.0 - xmm,
                ALPHA[1] - 1.0,
                PTM[5],
                s,
                &zn1,
            );
            let (dm16, _) = self.densu(z, b16, tinf, tlb, xmm, 0.0, PTM[5], s, &zn1);
            self.dm16 = dm16;
            self.d[1] = dnet(self.d[1], dm16, zhm28, xmm, 16.0);
            let rl = PDM[1][1] * PDL[1][16] * (1.0 + PDL[0][23] * (input.f107_avg - 150.0));
            let hc16 = PDM[1][5] * PDL[1][3];
            let zc16 = PDM[1][4] * PDL[1][2];
            let hc216 = PDM[1][5] * PDL[1][4];
            self.d[1] *= ccor2(z, rl, hc16, zc16, hc216);
            let hcc16 = PDM[1][7] * PDL[1][13];
            let zcc16 = PDM[1][6] * PDL[1][12];
            let rc16 = PDM[1][3] * PDL[1][14];
            self.d[1] *= ccor(z, rc16, hcc16, zcc16);
        }

        // O2。
        let g32 = self.globe7(&PD[4], input);
        let db32 = PDM[3][0] * g32.exp() * PD[4][0];
        let (d3, _) = self.densu(z, db32, tinf, tlb, 32.0, ALPHA[3], PTM[5], s, &zn1);
        self.d[3] = d3;
        {
            if z <= ALTL[3] {
                let zh32 = PDM[3][2];
                let (b32, _) = self.densu(
                    zh32,
                    db32,
                    tinf,
                    tlb,
                    32.0 - xmm,
                    ALPHA[3] - 1.0,
                    PTM[5],
                    s,
                    &zn1,
                );
                let (dm32, _) = self.densu(z, b32, tinf, tlb, xmm, 0.0, PTM[5], s, &zn1);
                self.dm32 = dm32;
                self.d[3] = dnet(self.d[3], dm32, zhm28, xmm, 32.0);
                let rl = (b28 * PDM[3][1] / b32).ln();
                let hc32 = PDM[3][5] * PDL[1][7];
                let zc32 = PDM[3][4] * PDL[1][6];
                self.d[3] *= ccor(z, rl, hc32, zc32);
            }
            // 湍流层底以上偏离扩散平衡的修正。
            let hcc32 = PDM[3][7] * PDL[1][22];
            let hcc232 = PDM[3][7] * PDL[0][22];
            let zcc32 = PDM[3][6] * PDL[1][21];
            let rc32 = PDM[3][3] * PDL[1][23] * (1.0 + PDL[0][23] * (input.f107_avg - 150.0));
            self.d[3] *= ccor2(z, rc32, hcc32, zcc32, hcc232);
        }

        // Ar。
        let g40 = self.globe7(&PD[5], input);
        let db40 = PDM[4][0] * g40.exp() * PD[5][0];
        let (d4, _) = self.densu(z, db40, tinf, tlb, 40.0, ALPHA[4], PTM[5], s, &zn1);
        self.d[4] = d4;
        if z <= ALTL[4] {
            let zh40 = PDM[4][2];
            let (b40, _) = self.densu(
                zh40,
                db40,
                tinf,
                tlb,
                40.0 - xmm,
                ALPHA[4] - 1.0,
                PTM[5],
                s,
                &zn1,
            );
            let (dm40, _) = self.densu(z, b40, tinf, tlb, xmm, 0.0, PTM[5], s, &zn1);
            self.dm40 = dm40;
            self.d[4] = dnet(self.d[4], dm40, zhm28, xmm, 40.0);
            let rl = (b28 * PDM[4][1] / b40).ln();
            let hc40 = PDM[4][5] * PDL[1][9];
            let zc40 = PDM[4][4] * PDL[1][8];
            self.d[4] *= ccor(z, rl, hc40, zc40);
        }

        // H。
        let g1 = self.globe7(&PD[6], input);
        let db01 = PDM[5][0] * g1.exp() * PD[6][0];
        let (d6, _) = self.densu(z, db01, tinf, tlb, 1.0, ALPHA[6], PTM[5], s, &zn1);
        self.d[6] = d6;
        if z <= ALTL[6] {
            let zh01 = PDM[5][2];
            let (b01, _) = self.densu(
                zh01,
                db01,
                tinf,
                tlb,
                1.0 - xmm,
                ALPHA[6] - 1.0,
                PTM[5],
                s,
                &zn1,
            );
            let (dm01, _) = self.densu(z, b01, tinf, tlb, xmm, 0.0, PTM[5], s, &zn1);
            self.dm01 = dm01;
            self.d[6] = dnet(self.d[6], dm01, zhm28, xmm, 1.0);
            let rl = (b28 * PDM[5][1] * PDL[1][17].abs() / b01).ln();
            let hc01 = PDM[5][5] * PDL[1][11];
            let zc01 = PDM[5][4] * PDL[1][10];
            self.d[6] *= ccor(z, rl, hc01, zc01);
            let hcc01 = PDM[5][7] * PDL[1][19];
            let zcc01 = PDM[5][6] * PDL[1][18];
            let rc01 = PDM[5][3] * PDL[1][20];
            self.d[6] *= ccor(z, rc01, hcc01, zcc01);
        }

        // 原子氮。
        let g14 = self.globe7(&PD[7], input);
        let db14 = PDM[6][0] * g14.exp() * PD[7][0];
        let (d7, _) = self.densu(z, db14, tinf, tlb, 14.0, ALPHA[7], PTM[5], s, &zn1);
        self.d[7] = d7;
        if z <= ALTL[7] {
            let zh14 = PDM[6][2];
            let (b14, _) = self.densu(
                zh14,
                db14,
                tinf,
                tlb,
                14.0 - xmm,
                ALPHA[7] - 1.0,
                PTM[5],
                s,
                &zn1,
            );
            let (dm14, _) = self.densu(z, b14, tinf, tlb, xmm, 0.0, PTM[5], s, &zn1);
            self.dm14 = dm14;
            self.d[7] = dnet(self.d[7], dm14, zhm28, xmm, 14.0);
            let rl = (b28 * PDM[6][1] * PDL[0][2].abs() / b14).ln();
            let hc14 = PDM[6][5] * PDL[0][1];
            let zc14 = PDM[6][4] * PDL[0][0];
            self.d[7] *= ccor(z, rl, hc14, zc14);
            let hcc14 = PDM[6][7] * PDL[0][4];
            let zcc14 = PDM[6][6] * PDL[0][3];
            let rc14 = PDM[6][3] * PDL[0][5];
            self.d[7] *= ccor(z, rc14, hcc14, zcc14);
        }

        // 异常氧。
        let g16h = self.globe7(&PD[8], input);
        let db16h = PDM[7][0] * g16h.exp() * PD[8][0];
        let tho = PDM[7][9] * PDL[0][6];
        let (dd, _) = self.densu(z, db16h, tho, tho, 16.0, ALPHA[8], PTM[5], s, &zn1);
        let zsht = PDM[7][5];
        let zmho = PDM[7][4];
        let zsho = self.scalh(zmho, 16.0, tho);
        self.d[8] = dd * (-zsht / zsho * ((-(z - zmho) / zsht).exp() - 1.0)).exp();

        // 总质量密度（不含异常氧）。
        self.update_mass_density();

        // 温度：xm = 0 → 只算温度。
        let (_, tz) = self.densu(z.abs(), 1.0, tinf, tlb, 0.0, 0.0, PTM[5], s, &zn1);
        self.t[1] = tz;
    }

    /// 由物种数密度汇总 cgs 总质量密度（`gtd7` 口径，不含异常氧）。
    fn update_mass_density(&mut self) {
        let d = &mut self.d;
        d[5] = AMU_G
            * (4.0 * d[0]
                + 16.0 * d[1]
                + 28.0 * d[2]
                + 32.0 * d[3]
                + 40.0 * d[4]
                + d[6]
                + 14.0 * d[7]);
    }

    /// cgs "drag 有效总质量密度"（`gtd7d` 口径，含异常氧）。
    fn mass_density_drag(&self) -> f64 {
        let d = &self.d;
        AMU_G
            * (4.0 * d[0]
                + 16.0 * d[1]
                + 28.0 * d[2]
                + 32.0 * d[3]
                + 40.0 * d[4]
                + d[6]
                + 14.0 * d[7]
                + 16.0 * d[8])
    }

    // ── globe7 / glob7s ─────────────────────────────────────────────────

    /// 上层热层球谐展开（参考实现 `globe7`）。
    fn globe7(&mut self, p: &[f64], input: &Nrlmsise00Input) -> f64 {
        let mut t = [0.0_f64; 14];
        let tloc = input.ut_seconds / 3600.0 + input.geodetic_lon_deg / 15.0;

        // 勒让德多项式（注意参考实现中 c = sin(lat)、s = cos(lat)）。
        let c = (input.geodetic_lat_deg * DGTR).sin();
        let s = (input.geodetic_lat_deg * DGTR).cos();
        let (c2, c4, s2) = (c * c, c * c * c * c, s * s);

        let plg = &mut self.plg;
        plg[0][1] = c;
        plg[0][2] = 0.5 * (3.0 * c2 - 1.0);
        plg[0][3] = 0.5 * (5.0 * c * c2 - 3.0 * c);
        plg[0][4] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0;
        plg[0][5] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0;
        plg[0][6] = (11.0 * c * plg[0][5] - 5.0 * plg[0][4]) / 6.0;
        plg[1][1] = s;
        plg[1][2] = 3.0 * c * s;
        plg[1][3] = 1.5 * (5.0 * c2 - 1.0) * s;
        plg[1][4] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s;
        plg[1][5] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s;
        plg[1][6] = (11.0 * c * plg[1][5] - 6.0 * plg[1][4]) / 5.0;
        plg[2][2] = 3.0 * s2;
        plg[2][3] = 15.0 * s2 * c;
        plg[2][4] = 7.5 * (7.0 * c2 - 1.0) * s2;
        plg[2][5] = 3.0 * c * plg[2][4] - 2.0 * plg[2][3];
        plg[2][6] = (11.0 * c * plg[2][5] - 7.0 * plg[2][4]) / 4.0;
        plg[2][7] = (13.0 * c * plg[2][6] - 8.0 * plg[2][5]) / 5.0;
        plg[3][3] = 15.0 * s2 * s;
        plg[3][4] = 105.0 * s2 * s * c;
        plg[3][5] = (9.0 * c * plg[3][4] - 7.0 * plg[3][3]) / 2.0;
        plg[3][6] = (11.0 * c * plg[3][5] - 8.0 * plg[3][4]) / 3.0;

        self.stloc = (HR * tloc).sin();
        self.ctloc = (HR * tloc).cos();
        self.s2tloc = (2.0 * HR * tloc).sin();
        self.c2tloc = (2.0 * HR * tloc).cos();
        self.s3tloc = (3.0 * HR * tloc).sin();
        self.c3tloc = (3.0 * HR * tloc).cos();

        let doy = f64::from(input.day_of_year);
        let cd32 = (DR * (doy - p[31])).cos();
        let cd18 = (2.0 * DR * (doy - p[17])).cos();
        let cd14 = (DR * (doy - p[13])).cos();
        let cd39 = (2.0 * DR * (doy - p[38])).cos();

        // F10.7 效应。
        let df = input.f107_daily - input.f107_avg;
        self.dfa = input.f107_avg - 150.0;
        let dfa = self.dfa;
        t[0] =
            p[19] * df * (1.0 + p[59] * dfa) + p[20] * df * df + p[21] * dfa + p[29] * dfa.powi(2);
        let f1 = 1.0 + (p[47] * dfa + p[19] * df + p[20] * df * df);
        let f2 = 1.0 + (p[49] * dfa + p[19] * df + p[20] * df * df);

        // 时间无关项。
        t[1] = (p[1] * plg[0][2] + p[2] * plg[0][4] + p[22] * plg[0][6])
            + (p[14] * plg[0][2]) * dfa
            + p[26] * plg[0][1];

        // 对称年变化。
        t[2] = p[18] * cd32;

        // 对称半年变化。
        t[3] = (p[15] + p[16] * plg[0][2]) * cd18;

        // 反对称年变化。
        t[4] = f1 * (p[9] * plg[0][1] + p[10] * plg[0][3]) * cd14;

        // 反对称半年变化。
        t[5] = p[37] * plg[0][1] * cd39;

        // 周日项。
        {
            let t71 = (p[11] * plg[1][2]) * cd14;
            let t72 = (p[12] * plg[1][2]) * cd14;
            t[6] = f2
                * ((p[3] * plg[1][1] + p[4] * plg[1][3] + p[27] * plg[1][5] + t71) * self.ctloc
                    + (p[6] * plg[1][1] + p[7] * plg[1][3] + p[28] * plg[1][5] + t72) * self.stloc);
        }

        // 半日项。
        {
            let t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14;
            let t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14;
            t[7] = f2
                * ((p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * self.c2tloc
                    + (p[8] * plg[2][2] + p[42] * plg[2][4] + t82) * self.s2tloc);
        }

        // 三分日项。
        {
            t[13] = f2
                * ((p[39] * plg[3][3] + (p[93] * plg[3][4] + p[46] * plg[3][6]) * cd14)
                    * self.s3tloc
                    + (p[40] * plg[3][3] + (p[94] * plg[3][4] + p[48] * plg[3][6]) * cd14)
                        * self.c3tloc);
        }

        // 磁活动：Ap 史（switch 9 = −1）或标量日 Ap（switch 9 = +1）。
        if self.storm_time {
            if p[51] != 0.0 {
                let p24 = if p[24] < 1.0E-4 { 1.0E-4 } else { p[24] };
                let mut exp1 = (-10800.0 * p[51].abs()
                    / (1.0
                        + p[138]
                            * (45.0 - (input.geodetic_lat_deg * input.geodetic_lat_deg).sqrt())))
                .exp();
                if exp1 > 0.99999 {
                    exp1 = 0.99999;
                }
                self.apt[0] = sg0(exp1, &input.ap, p24, p[25]);
                t[8] = self.apt[0]
                    * (p[50]
                        + p[96] * plg[0][2]
                        + p[54] * plg[0][4]
                        + (p[125] * plg[0][1] + p[126] * plg[0][3] + p[127] * plg[0][5]) * cd14
                        + (p[128] * plg[1][1] + p[129] * plg[1][3] + p[130] * plg[1][5])
                            * (HR * (tloc - p[131])).cos());
            }
        } else {
            let apd = input.ap[0] - 4.0;
            let p45 = p[44];
            let p44 = if p[43] < 0.0 { 1.0E-5 } else { p[43] };
            self.apdf = apd + (p45 - 1.0) * (apd + ((-p44 * apd).exp() - 1.0) / p44);
            let apdf = self.apdf;
            t[8] = apdf
                * (p[32]
                    + p[45] * plg[0][2]
                    + p[34] * plg[0][4]
                    + (p[100] * plg[0][1] + p[101] * plg[0][3] + p[102] * plg[0][5]) * cd14
                    + (p[121] * plg[1][1] + p[122] * plg[1][3] + p[123] * plg[1][5])
                        * (HR * (tloc - p[124])).cos());
        }

        // 经度、UT 与磁活动耦合项。
        {
            let glon = DGTR * input.geodetic_lon_deg;

            // 经度项。
            t[10] = (1.0 + p[80] * dfa)
                * ((p[64] * plg[1][2]
                    + p[65] * plg[1][4]
                    + p[66] * plg[1][6]
                    + p[103] * plg[1][1]
                    + p[104] * plg[1][3]
                    + p[105] * plg[1][5]
                    + (p[109] * plg[1][1] + p[110] * plg[1][3] + p[111] * plg[1][5]) * cd14)
                    * glon.cos()
                    + (p[90] * plg[1][2]
                        + p[91] * plg[1][4]
                        + p[92] * plg[1][6]
                        + p[106] * plg[1][1]
                        + p[107] * plg[1][3]
                        + p[108] * plg[1][5]
                        + (p[112] * plg[1][1] + p[113] * plg[1][3] + p[114] * plg[1][5]) * cd14)
                        * glon.sin());

            // UT 与 UT/经度混合项。
            t[11] = (1.0 + p[95] * plg[0][1])
                * (1.0 + p[81] * dfa)
                * (1.0 + p[119] * plg[0][1] * cd14)
                * ((p[68] * plg[0][1] + p[69] * plg[0][3] + p[70] * plg[0][5])
                    * (SR * (input.ut_seconds - p[71])).cos());
            t[11] += (p[76] * plg[2][3] + p[77] * plg[2][5] + p[78] * plg[2][7])
                * (SR * (input.ut_seconds - p[79]) + 2.0 * glon).cos()
                * (1.0 + p[137] * dfa);

            // UT/经度/磁活动项。
            if self.storm_time {
                if p[51] != 0.0 {
                    t[12] = self.apt[0]
                        * (1.0 + p[132] * plg[0][1])
                        * ((p[52] * plg[1][2] + p[98] * plg[1][4] + p[67] * plg[1][6])
                            * (glon - DGTR * p[97]).cos())
                        + self.apt[0]
                            * (p[133] * plg[1][1] + p[134] * plg[1][3] + p[135] * plg[1][5])
                            * cd14
                            * (glon - DGTR * p[136]).cos()
                        + self.apt[0]
                            * (p[55] * plg[0][1] + p[56] * plg[0][3] + p[57] * plg[0][5])
                            * (SR * (input.ut_seconds - p[58])).cos();
                }
            } else {
                let apdf = self.apdf;
                t[12] = apdf
                    * (1.0 + p[120] * plg[0][1])
                    * ((p[60] * plg[1][2] + p[61] * plg[1][4] + p[62] * plg[1][6])
                        * (glon - DGTR * p[63]).cos())
                    + apdf
                        * (p[115] * plg[1][1] + p[116] * plg[1][3] + p[117] * plg[1][5])
                        * cd14
                        * (glon - DGTR * p[118]).cos()
                    + apdf
                        * (p[83] * plg[0][1] + p[84] * plg[0][3] + p[85] * plg[0][5])
                        * (SR * (input.ut_seconds - p[75])).cos();
            }
        }

        // 标准开关集下 |sw[i+1]| = 1，故 tinf 为 p[30] 加全部 t[i]。
        t.iter().fold(p[30], |acc, x| acc + x)
    }

    /// 下层大气球谐展开（参考实现 `glob7s`）。
    ///
    /// 参数集标记 `p[99]` 在全部使用到的行上为 0 或 2（等价于参考实现里
    /// "为 0 → 置 2" 的原地写入），故此处只读不写。
    fn glob7s(&self, p: &[f64], input: &Nrlmsise00Input) -> f64 {
        let mut t = [0.0_f64; 14];
        let doy = f64::from(input.day_of_year);
        let cd32 = (DR * (doy - p[31])).cos();
        let cd18 = (2.0 * DR * (doy - p[17])).cos();
        let cd14 = (DR * (doy - p[13])).cos();
        let cd39 = (2.0 * DR * (doy - p[38])).cos();
        let dfa = self.dfa;
        let plg = &self.plg;

        // F10.7。
        t[0] = p[21] * dfa;

        // 时间无关项。
        t[1] = p[1] * plg[0][2]
            + p[2] * plg[0][4]
            + p[22] * plg[0][6]
            + p[26] * plg[0][1]
            + p[14] * plg[0][3]
            + p[59] * plg[0][5];

        // 对称年变化。
        t[2] = (p[18] + p[47] * plg[0][2] + p[29] * plg[0][4]) * cd32;

        // 对称半年变化。
        t[3] = (p[15] + p[16] * plg[0][2] + p[30] * plg[0][4]) * cd18;

        // 反对称年变化。
        t[4] = (p[9] * plg[0][1] + p[10] * plg[0][3] + p[20] * plg[0][5]) * cd14;

        // 反对称半年变化。
        t[5] = (p[37] * plg[0][1]) * cd39;

        // 周日项。
        {
            let t71 = p[11] * plg[1][2] * cd14;
            let t72 = p[12] * plg[1][2] * cd14;
            t[6] = (p[3] * plg[1][1] + p[4] * plg[1][3] + t71) * self.ctloc
                + (p[6] * plg[1][1] + p[7] * plg[1][3] + t72) * self.stloc;
        }

        // 半日项。
        {
            let t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14;
            let t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14;
            t[7] = (p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * self.c2tloc
                + (p[8] * plg[2][2] + p[42] * plg[2][4] + t82) * self.s2tloc;
        }

        // 三分日项。
        t[13] = p[39] * plg[3][3] * self.s3tloc + p[40] * plg[3][3] * self.c3tloc;

        // 磁活动。
        if self.storm_time {
            t[8] = p[50] * self.apt[0] + p[96] * plg[0][2] * self.apt[0];
        } else {
            t[8] = self.apdf * (p[32] + p[45] * plg[0][2]);
        }

        // 经度项。
        {
            let glon = DGTR * input.geodetic_lon_deg;
            let doy_phase = 1.0
                + plg[0][1]
                    * (p[80] * (DR * (doy - p[81])).cos()
                        + p[85] * (2.0 * DR * (doy - p[86])).cos())
                + p[83] * (DR * (doy - p[84])).cos()
                + p[87] * (2.0 * DR * (doy - p[88])).cos();
            t[10] = doy_phase
                * ((p[64] * plg[1][2]
                    + p[65] * plg[1][4]
                    + p[66] * plg[1][6]
                    + p[74] * plg[1][1]
                    + p[75] * plg[1][3]
                    + p[76] * plg[1][5])
                    * glon.cos()
                    + (p[90] * plg[1][2]
                        + p[91] * plg[1][4]
                        + p[92] * plg[1][6]
                        + p[77] * plg[1][1]
                        + p[78] * plg[1][3]
                        + p[79] * plg[1][5])
                        * glon.sin());
        }

        t.iter().sum()
    }

    // ── densm / densu ───────────────────────────────────────────────────

    /// 下层大气（平流层/中层/对流层）温度与密度剖面（参考实现 `densm`）。
    ///
    /// 返回 `(返回值, tz)`：参考实现的返回值在 `xm = 0` 时是温度、否则是密度；
    /// 本实现原样保留该语义并把 `tz` 一并返回（调用方各取所需）。
    fn densm(&self, alt: f64, d0: f64, xm: f64) -> (f64, f64) {
        const MN3: usize = 5;
        const MN2: usize = 4;
        let (tn2, tgn2) = (&self.meso_tn2, &self.meso_tgn2);
        let (tn3, tgn3) = (&self.meso_tn3, &self.meso_tgn3);
        let mut tz = 0.0;
        let mut densm_tmp = d0;
        if alt > ZN2[0] {
            return (if xm == 0.0 { tz } else { d0 }, tz);
        }

        // 平流层/中层温度。
        let z = if alt > ZN2[MN2 - 1] {
            alt
        } else {
            ZN2[MN2 - 1]
        };
        let (z1, z2) = (ZN2[0], ZN2[MN2 - 1]);
        let (t1, t2) = (tn2[0], tn2[MN2 - 1]);
        let zg = zeta(self.re, z, z1);
        let zgdif = zeta(self.re, z2, z1);

        let mut xs = [0.0_f64; 10];
        let mut ys = [0.0_f64; 10];
        let mut y2out = [0.0_f64; 10];
        for (k, (zn, tn)) in ZN2.iter().zip(tn2.iter()).enumerate() {
            xs[k] = zeta(self.re, *zn, z1) / zgdif;
            ys[k] = 1.0 / tn;
        }
        let yd1 = -tgn2[0] / (t1 * t1) * zgdif;
        let yd2 = -tgn2[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)).powi(2);
        spline(&xs, &ys, MN2, yd1, yd2, &mut y2out);
        let x = zg / zgdif;
        let y = splint(&xs, &ys, &y2out, MN2, x);

        tz = 1.0 / y;
        if xm != 0.0 {
            // 平流层/中层密度。
            let glb = self.gsurf / (1.0 + z1 / self.re).powi(2);
            let gamm = xm * glb * zgdif / RGAS;
            let mut expl = gamm * splini(&xs, &ys, &y2out, MN2, x);
            if expl > 50.0 {
                expl = 50.0;
            }
            densm_tmp = densm_tmp * (t1 / tz) * (-expl).exp();
        }

        if alt > ZN3[0] {
            return (if xm == 0.0 { tz } else { densm_tmp }, tz);
        }

        // 对流层/平流层温度。
        let z = alt;
        let (z1, z2) = (ZN3[0], ZN3[MN3 - 1]);
        let (t1, t2) = (tn3[0], tn3[MN3 - 1]);
        let zg = zeta(self.re, z, z1);
        let zgdif = zeta(self.re, z2, z1);
        for (k, (zn, tn)) in ZN3.iter().zip(tn3.iter()).enumerate() {
            xs[k] = zeta(self.re, *zn, z1) / zgdif;
            ys[k] = 1.0 / tn;
        }
        let yd1 = -tgn3[0] / (t1 * t1) * zgdif;
        let yd2 = -tgn3[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)).powi(2);
        spline(&xs, &ys, MN3, yd1, yd2, &mut y2out);
        let x = zg / zgdif;
        let y = splint(&xs, &ys, &y2out, MN3, x);

        tz = 1.0 / y;
        if xm != 0.0 {
            let glb = self.gsurf / (1.0 + z1 / self.re).powi(2);
            let gamm = xm * glb * zgdif / RGAS;
            let mut expl = gamm * splini(&xs, &ys, &y2out, MN3, x);
            if expl > 50.0 {
                expl = 50.0;
            }
            densm_tmp = densm_tmp * (t1 / tz) * (-expl).exp();
        }
        (if xm == 0.0 { tz } else { densm_tmp }, tz)
    }

    /// 热层 Bates 剖面 + 下层 spline 拼接（参考实现 `densu`）。
    ///
    /// 会就地写入 `meso_tn1[0]`/`meso_tgn1[0]`（Bates 剖面在 `za` 处的温度与梯度），
    /// 与参考实现对全局节点的写法一致。
    #[allow(clippy::too_many_arguments)]
    fn densu(
        &mut self,
        alt: f64,
        dlb: f64,
        tinf: f64,
        tlb: f64,
        xm: f64,
        alpha: f64,
        zlb: f64,
        s2: f64,
        zn1: &[f64; 5],
    ) -> (f64, f64) {
        const MN1: usize = 5;
        let za = zn1[0];
        let z = if alt > za { alt } else { za };

        // 位势高度差（相对 ZLB）。
        let zg2 = zeta(self.re, z, zlb);

        // Bates 温度。
        let tt = tinf - (tinf - tlb) * (-s2 * zg2).exp();
        let ta = tt;
        let mut tz = tt;
        let mut densu_temp = tz;

        let mut xs = [0.0_f64; 5];
        let mut ys = [0.0_f64; 5];
        let mut y2out = [0.0_f64; 5];
        let mut zgdif = 0.0;
        let mut z1 = 0.0;
        let mut x = 0.0;
        let mut t1 = 0.0;

        if alt < za {
            // za 以下温度：由 Bates 剖面在 za 处的梯度构造 spline。
            let dta = (tinf - ta) * s2 * ((self.re + zlb) / (self.re + za)).powi(2);
            self.meso_tgn1[0] = dta;
            self.meso_tn1[0] = ta;
            let z = if alt > zn1[MN1 - 1] {
                alt
            } else {
                zn1[MN1 - 1]
            };
            z1 = zn1[0];
            let z2 = zn1[MN1 - 1];
            t1 = self.meso_tn1[0];
            let t2 = self.meso_tn1[MN1 - 1];
            let zg = zeta(self.re, z, z1);
            zgdif = zeta(self.re, z2, z1);
            for (k, (zn, tn)) in zn1.iter().zip(self.meso_tn1.iter()).enumerate() {
                xs[k] = zeta(self.re, *zn, z1) / zgdif;
                ys[k] = 1.0 / tn;
            }
            let yd1 = -self.meso_tgn1[0] / (t1 * t1) * zgdif;
            let yd2 =
                -self.meso_tgn1[1] / (t2 * t2) * zgdif * ((self.re + z2) / (self.re + z1)).powi(2);
            spline(&xs, &ys, MN1, yd1, yd2, &mut y2out);
            x = zg / zgdif;
            let y = splint(&xs, &ys, &y2out, MN1, x);
            tz = 1.0 / y;
            densu_temp = tz;
        }

        if xm == 0.0 {
            return (densu_temp, tz);
        }

        // za 以上的密度。
        let glb = self.gsurf / (1.0 + zlb / self.re).powi(2);
        let gamma = xm * glb / (s2 * RGAS * tinf);
        let mut expl = (-s2 * gamma * zg2).exp();
        if expl > 50.0 {
            expl = 50.0;
        }
        if tt <= 0.0 {
            expl = 50.0;
        }

        let densa = dlb * (tlb / tt).powf(1.0 + alpha + gamma) * expl;
        densu_temp = densa;
        if alt >= za {
            return (densu_temp, tz);
        }

        // za 以下的密度。
        let glb = self.gsurf / (1.0 + z1 / self.re).powi(2);
        let gamm = xm * glb * zgdif / RGAS;
        expl = gamm * splini(&xs, &ys, &y2out, MN1, x);
        if expl > 50.0 {
            expl = 50.0;
        }
        if tz <= 0.0 {
            expl = 50.0;
        }
        densu_temp = densu_temp * (t1 / tz).powf(1.0 + alpha) * (-expl).exp();
        (densu_temp, tz)
    }
}

// ── 纯函数 ─────────────────────────────────────────────────────────────────

/// 位势高度差（参考实现 `zeta`）。
fn zeta(re: f64, zz: f64, zl: f64) -> f64 {
    (zz - zl) * (re + zl) / (re + zz)
}

/// 磁活动 `g0`（参考实现 Eq. A24d）。
fn g0(a: f64, p24: f64, p25: f64) -> f64 {
    a - 4.0 + (p25 - 1.0) * (a - 4.0 + ((-p24.abs() * (a - 4.0)).exp() - 1.0) / p24.abs())
}

/// `sumex`（参考实现 Eq. A24c）。
fn sumex(ex: f64) -> f64 {
    1.0 + (1.0 - ex.powi(19)) / (1.0 - ex) * ex.sqrt()
}

/// `sg0`（参考实现 Eq. A24a），使用 Ap 史的 3 小时分辨率先验。
fn sg0(ex: f64, ap: &[f64; 7], p24: f64, p25: f64) -> f64 {
    (g0(ap[1], p24, p25)
        + (g0(ap[2], p24, p25) * ex
            + g0(ap[3], p24, p25) * ex * ex
            + g0(ap[4], p24, p25) * ex.powi(3)
            + (g0(ap[5], p24, p25) * ex.powi(4) + g0(ap[6], p24, p25) * ex.powi(12))
                * (1.0 - ex.powi(8))
                / (1.0 - ex)))
        / sumex(ex)
}

/// 化学/离解修正（参考实现 `ccor`）。
fn ccor(alt: f64, r: f64, h1: f64, zh: f64) -> f64 {
    let e = (alt - zh) / h1;
    if e > 70.0 {
        return 1.0; // exp(0)
    }
    if e < -70.0 {
        return r.exp();
    }
    let ex = e.exp();
    (r / (1.0 + ex)).exp()
}

/// 双尺度化学/离解修正（参考实现 `ccor2`）。
fn ccor2(alt: f64, r: f64, h1: f64, zh: f64, h2: f64) -> f64 {
    let e1 = (alt - zh) / h1;
    let e2 = (alt - zh) / h2;
    if e1 > 70.0 || e2 > 70.0 {
        return 1.0;
    }
    if e1 < -70.0 && e2 < -70.0 {
        return r.exp();
    }
    let (ex1, ex2) = (e1.exp(), e2.exp());
    (r / (1.0 + 0.5 * (ex1 + ex2))).exp()
}

/// 湍流层顶修正（参考实现 `dnet`）。
fn dnet(dd_in: f64, dm: f64, zhm: f64, xmm: f64, xm: f64) -> f64 {
    let mut dd = dd_in;
    let a0 = zhm / (xmm - xm);
    if !(dm > 0.0 && dd > 0.0) {
        if dd == 0.0 && dm == 0.0 {
            dd = 1.0;
        }
        if dm == 0.0 {
            return dd;
        }
        if dd == 0.0 {
            return dm;
        }
    }
    let ylog = a0 * (dm / dd).ln();
    if ylog < -10.0 {
        return dd;
    }
    if ylog > 10.0 {
        return dm;
    }
    dd * (1.0 + ylog.exp()).powf(1.0 / a0)
}

/// 三次样条二阶导数（参考实现 `spline`，Numerical Recipes 移植）。
fn spline(x: &[f64], y: &[f64], n: usize, yp1: f64, ypn: f64, y2: &mut [f64]) {
    let mut u = [0.0_f64; 10];
    if yp1 > 0.99E30 {
        y2[0] = 0.0;
        u[0] = 0.0;
    } else {
        y2[0] = -0.5;
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
    }
    for i in 1..n - 1 {
        let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (6.0
            * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1])
            / p;
    }
    let (qn, un) = if ypn > 0.99E30 {
        (0.0, 0.0)
    } else {
        (
            0.5,
            (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])),
        )
    };
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
    for k in (0..n - 1).rev() {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
}

/// 三次样条插值（参考实现 `splint`）。
fn splint(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut klo = 0usize;
    let mut khi = n - 1;
    while khi - klo > 1 {
        let k = (khi + klo) / 2;
        if xa[k] > x {
            khi = k;
        } else {
            klo = k;
        }
    }
    let h = xa[khi] - xa[klo];
    let a = (xa[khi] - x) / h;
    let b = (x - xa[klo]) / h;
    a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0
}

/// 三次样条积分（参考实现 `splini`）。
fn splini(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut yi = 0.0;
    let mut klo = 0usize;
    let mut khi = 1usize;
    while x > xa[klo] && khi < n {
        let xx = if khi < n - 1 {
            if x < xa[khi] {
                x
            } else {
                xa[khi]
            }
        } else {
            x
        };
        let h = xa[khi] - xa[klo];
        let a = (xa[khi] - xx) / h;
        let b = (xx - xa[klo]) / h;
        let (a2, b2) = (a * a, b * b);
        yi += ((1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + ((-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi])
                * h
                * h
                / 6.0)
            * h;
        klo += 1;
        khi += 1;
    }
    yi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input(alt_km: f64) -> Nrlmsise00Input {
        Nrlmsise00Input {
            day_of_year: 1,
            ut_seconds: 21600.0,
            altitude_km: alt_km,
            geodetic_lat_deg: 0.0,
            geodetic_lon_deg: 0.0,
            f107_daily: 150.0,
            f107_avg: 150.0,
            ap: [15.0; 7],
        }
    }

    /// 1000 km 及以上返回零密度零温度（模型域外）。
    #[test]
    fn above_ceiling_is_zero() {
        for alt in [1000.0, 1200.0, 5000.0] {
            let out = density(&sample_input(alt), false);
            assert_eq!(out.density_kg_m3, 0.0, "alt={alt}");
            assert_eq!(out.temperature_k, 0.0, "alt={alt}");
        }
    }

    /// 负高度钳到 0 km：与 0 km 结果完全一致。
    #[test]
    fn negative_altitude_clamps_to_zero() {
        let a = density(&sample_input(-50.0), false);
        let b = density(&sample_input(0.0), false);
        assert_eq!(a, b);
        assert!(
            (b.density_kg_m3 - 1.225).abs() < 0.1,
            "海平面密度应约 1.2 kg/m³，got {}",
            b.density_kg_m3
        );
    }

    /// 密度随高度单调衰减（100–900 km），且量级合理。
    #[test]
    fn density_decays_with_altitude() {
        let mut prev = f64::INFINITY;
        for alt in [100.0, 200.0, 300.0, 400.0, 500.0, 700.0, 900.0] {
            let rho = density(&sample_input(alt), false).density_kg_m3;
            assert!(rho > 0.0, "alt={alt}");
            assert!(rho < prev, "alt={alt}: {rho} !< {prev}");
            prev = rho;
        }
    }

    /// `storm_time` 打开 Ap 史后密度与静态结果不同（磁活动项生效）。
    #[test]
    fn storm_time_changes_density() {
        let mut storm = sample_input(400.0);
        storm.ap = [15.0, 130.0, 150.0, 50.0, 20.0, 15.0, 15.0];
        let quiet = density(&sample_input(400.0), false).density_kg_m3;
        let active = density(&storm, true).density_kg_m3;
        assert!(
            (active / quiet - 1.0).abs() > 1e-3,
            "Ap 史驱动的密度差应可分辨: quiet={quiet:e} storm={active:e}"
        );
    }

    /// 暴时判定：全等（含标量广播）为静态，任一元素不同即暴时。
    #[test]
    fn storm_time_rule_is_flatness_of_ap_history() {
        assert!(!storm_time_from_ap(&[15.0; 7]));
        assert!(!storm_time_from_ap(&[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]));
        assert!(storm_time_from_ap(&[
            15.0, 130.0, 150.0, 50.0, 20.0, 15.0, 15.0
        ]));
        // 只有末位不同也算暴时（不依赖具体位置）。
        assert!(storm_time_from_ap(&[
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 16.0
        ]));
    }

    /// 局地太阳时：加入经度偏移并折入 [0, 86400)。
    #[test]
    fn local_solar_time_wraps() {
        assert_eq!(local_solar_time_seconds(0.0, 0.0), 0.0);
        assert_eq!(local_solar_time_seconds(3600.0, 15.0), 7200.0);
        assert_eq!(local_solar_time_seconds(0.0, -15.0), 86400.0 - 3600.0);
        assert_eq!(local_solar_time_seconds(86000.0, 0.0), 86000.0);
        assert_eq!(local_solar_time_seconds(86400.0 + 10.0, 0.0), 10.0);
    }
}
