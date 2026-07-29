//! 混合推进系统建模（化学推进 + 电推进）。
//!
//! 结合脉冲推力（化学推进）和小推力（电推进），实现统一的混合推进系统建模。
//!
//! ## 推进系统参数（来自文献）
//!
//! **化学推进（脉冲）**：
//! - 比冲 Isp = 300-450 s（关宇同 400 s，Orion 310 s）
//! - 推力 T = 6-28 N（关宇同），或更高（Orion 25700 N）
//! - 应用：DRO 逃逸、GEO 插入等大轨道改变
//!
//! **电推进（小推力）**：
//! - 比冲 Isp = 1500-4100 s（SMART-1 1500 s，NSTAR 3100 s，NEXT 4100 s）
//! - 推力 T = 0.1-1 N（Caillau 0.3-10 N，潘迅 1 N，SMART-1 0.07 N）
//! - 应用：轨道精细调整、长期转移

/// 化学推进参数。
#[derive(Debug, Clone, Copy)]
pub struct ChemicalPropulsion {
    /// 最大推力（N）
    pub t_max: f64,
    /// 比冲（s）
    pub isp: f64,
}

impl Default for ChemicalPropulsion {
    fn default() -> Self {
        Self {
            t_max: 10.0, // N（关宇同 6-28 N）
            isp: 400.0,  // s（关宇同 400 s）
        }
    }
}

/// 电推进参数。
#[derive(Debug, Clone, Copy)]
pub struct ElectricPropulsion {
    /// 最大推力（N）
    pub t_max: f64,
    /// 比冲（s）
    pub isp: f64,
}

impl Default for ElectricPropulsion {
    fn default() -> Self {
        Self {
            t_max: 1.0,  // N（潘迅 1 N，Caillau 0.3-10 N）
            isp: 3000.0, // s（NSTAR 3100 s，SMART-1 1500 s）
        }
    }
}

/// 混合推进系统（化学 + 电推进）。
///
/// 支持两种推进方式的组合使用：
/// - 化学推进：瞬时速度改变（impulsive Δv）
/// - 电推进：连续小推力（continuous low-thrust）
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridPropulsion {
    /// 化学推进参数
    pub chemical: ChemicalPropulsion,
    /// 电推进参数
    pub electric: ElectricPropulsion,
}

impl HybridPropulsion {
    /// 创建新的混合推进系统。
    pub fn new(chemical: ChemicalPropulsion, electric: ElectricPropulsion) -> Self {
        Self { chemical, electric }
    }

    /// 计算混合推进加速度。
    ///
    /// # 参数
    /// - `mass`: 航天器质量（kg）
    /// - `chemical_throttle`: 化学推进推力幅值 ∈ [0, 1]
    /// - `electric_throttle`: 电推进推力幅值 ∈ [0, 1]
    /// - `chemical_direction`: 化学推进方向单位向量
    /// - `electric_direction`: 电推进方向单位向量
    ///
    /// # 返回
    /// 总加速度（km/s²）
    pub fn acceleration(
        &self,
        mass: f64,
        chemical_throttle: f64,
        electric_throttle: f64,
        chemical_direction: &[f64; 3],
        electric_direction: &[f64; 3],
    ) -> [f64; 3] {
        // 化学推进加速度（瞬时）
        let a_chem = if chemical_throttle > 0.0 {
            let mag_m_s2 = (self.chemical.t_max / mass) * chemical_throttle;
            let mag_km_s2 = mag_m_s2 / 1000.0;
            [
                mag_km_s2 * chemical_direction[0],
                mag_km_s2 * chemical_direction[1],
                mag_km_s2 * chemical_direction[2],
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        // 电推进加速度（连续）
        let a_elec = if electric_throttle > 0.0 {
            let mag_m_s2 = (self.electric.t_max / mass) * electric_throttle;
            let mag_km_s2 = mag_m_s2 / 1000.0;
            [
                mag_km_s2 * electric_direction[0],
                mag_km_s2 * electric_direction[1],
                mag_km_s2 * electric_direction[2],
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        // 总加速度
        [
            a_chem[0] + a_elec[0],
            a_chem[1] + a_elec[1],
            a_chem[2] + a_elec[2],
        ]
    }

    /// 计算混合推进质量流率。
    ///
    /// # 参数
    /// - `chemical_throttle`: 化学推进推力幅值
    /// - `electric_throttle`: 电推进推力幅值
    ///
    /// # 返回
    /// 质量流率（kg/s）
    pub fn mass_flow_rate(&self, chemical_throttle: f64, electric_throttle: f64) -> f64 {
        let g0 = 9.81; // m/s²

        // 化学推进质量流率
        let mdot_chem = if chemical_throttle > 0.0 {
            -chemical_throttle * self.chemical.t_max / (self.chemical.isp * g0)
        } else {
            0.0
        };

        // 电推进质量流率
        let mdot_elec = if electric_throttle > 0.0 {
            -electric_throttle * self.electric.t_max / (self.electric.isp * g0)
        } else {
            0.0
        };

        mdot_chem + mdot_elec
    }

    /// 计算化学推进脉冲 Δv。
    ///
    /// # 参数
    /// - `mass_before`: 脉冲前质量（kg）
    /// - `propellant_mass`: 燃料质量（kg）
    ///
    /// # 返回
    /// 速度增量 Δv（km/s）
    pub fn chemical_delta_v(&self, mass_before: f64, propellant_mass: f64) -> f64 {
        let g0 = 9.81; // m/s²
        let mass_after = mass_before - propellant_mass;

        // 火箭方程：Δv = Isp * g0 * ln(m_before / m_after)
        let dv_m_s = self.chemical.isp * g0 * (mass_before / mass_after).ln();
        dv_m_s / 1000.0 // 转换为 km/s
    }

    /// 计算电推进燃料消耗。
    ///
    /// # 参数
    /// - `delta_v`: 速度增量（km/s）
    /// - `mass_before`: 初始质量（kg）
    ///
    /// # 返回
    /// 燃料质量（kg）
    pub fn electric_propellant_mass(&self, delta_v: f64, mass_before: f64) -> f64 {
        let g0 = 9.81; // m/s²
        let dv_m_s = delta_v * 1000.0;

        // 火箭方程：m_prop = m_before * (1 - exp(-Δv / (Isp * g0)))
        mass_before * (1.0 - (-dv_m_s / (self.electric.isp * g0)).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chemical_propulsion_default() {
        let chem = ChemicalPropulsion::default();
        assert_eq!(chem.t_max, 10.0);
        assert_eq!(chem.isp, 400.0);
    }

    #[test]
    fn test_electric_propulsion_default() {
        let elec = ElectricPropulsion::default();
        assert_eq!(elec.t_max, 1.0);
        assert_eq!(elec.isp, 3000.0);
    }

    #[test]
    fn test_hybrid_acceleration() {
        let hybrid = HybridPropulsion::default();
        let mass = 1500.0; // kg

        // 只使用电推进
        let a = hybrid.acceleration(
            mass,
            0.0, // chemical_throttle
            1.0, // electric_throttle
            &[1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
        );

        // 电推进加速度：a = T/m = 1.0 N / 1500 kg = 6.67e-4 m/s² = 6.67e-7 km/s²
        let expected = 1.0 / 1500.0 / 1000.0;
        assert!((a[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mass_flow_rate() {
        let hybrid = HybridPropulsion::default();

        // 只使用电推进
        let mdot = hybrid.mass_flow_rate(0.0, 1.0);

        // 电推进质量流率：mdot = -T / (Isp * g0) = -1.0 / (3000 * 9.81)
        let expected = -1.0 / (3000.0 * 9.81);
        assert!((mdot - expected).abs() < 1e-10);
    }

    #[test]
    fn test_chemical_delta_v() {
        let hybrid = HybridPropulsion::default();
        let mass_before = 1500.0; // kg
        let propellant_mass = 100.0; // kg

        let dv = hybrid.chemical_delta_v(mass_before, propellant_mass);

        // Δv = Isp * g0 * ln(m_before / m_after) = 400 * 9.81 * ln(1500/1400) ≈ 268 m/s ≈ 0.268 km/s
        assert!(dv > 0.26 && dv < 0.28);
    }

    #[test]
    fn test_electric_propellant_mass() {
        let hybrid = HybridPropulsion::default();
        let delta_v = 1.0; // km/s
        let mass_before = 1500.0; // kg

        let prop = hybrid.electric_propellant_mass(delta_v, mass_before);

        // m_prop = m_before * (1 - exp(-Δv / (Isp * g0)))
        // = 1500 * (1 - exp(-1000 / (3000 * 9.81))) ≈ 50 kg
        assert!(prop > 45.0 && prop < 55.0);
    }
}
