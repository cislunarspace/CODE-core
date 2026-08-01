//! 同伦法求解器（L2-L1 同伦 + Sigmoid 平滑）。
//!
//! 参考 Caillau et al. (2012) 和 Fahey (2024) 的方法，实现从能量最优到燃料最优的
//! 平滑过渡。
//!
//! ## 核心思想
//!
//! 1. **能量最优（L2 范数）**：`J = ∫ u² dt`，控制连续，易于求解
//! 2. **燃料最优（L1 范数）**：`J = ∫ u dt`，bang-bang 控制，非光滑
//! 3. **同伦函数**：`J_λ = ∫ [(1-λ)u² + λu] dt`，λ: 0 → 1
//! 4. **Sigmoid 平滑**：`u = 1/2 [1 + tanh(S/ε)]`，ε → 0

use crate::multiple_shooting::{multiple_shooting_correct, MultipleShootingRustResult};
use e2m2e_forces::forces::augmented_state::ThrustParams;
use e2m2e_forces::forces::compiled::CompiledForce;

/// 同伦法求解器配置。
pub struct HomotopyConfig {
    /// 初始同伦参数 λ₀ = 0（能量最优）
    pub lambda_init: f64,
    /// 最终同伦参数 λ_f = 1（燃料最优）
    pub lambda_final: f64,
    /// 同伦步长
    pub lambda_step: f64,
    /// Sigmoid 平滑参数 ε
    pub epsilon: f64,
    /// 最大迭代次数（每个 λ 步）
    pub max_iter: usize,
    /// 收敛容差
    pub tolerance: f64,
    /// 积分相对容差
    pub rtol: f64,
}

impl Default for HomotopyConfig {
    fn default() -> Self {
        Self {
            lambda_init: 0.0,
            lambda_final: 1.0,
            lambda_step: 0.1,
            epsilon: 0.01,
            max_iter: 50,
            tolerance: 1e-6,
            rtol: 1e-10,
        }
    }
}

/// 同伦法求解结果。
pub struct HomotopyResult {
    /// 最终的 patch points（时间）
    pub t_patch: Vec<f64>,
    /// 最终的 patch points（状态，7D：[r, v, m]）
    pub state_patch: Vec<[f64; 7]>,
    /// 是否收敛
    pub converged: bool,
    /// 总迭代次数
    pub total_iterations: usize,
    /// 最终残差
    pub max_residual: f64,
    /// 同伦路径（λ 值序列）
    pub lambda_history: Vec<f64>,
    /// 每个 λ 步的残差历史
    pub residual_history: Vec<f64>,
}

/// 同伦法问题定义：打包在多个函数间重复出现的参数。
pub struct HomotopyProblem<'a> {
    /// 力模型列表
    pub forces: &'a [CompiledForce],
    /// 传播系 origin
    pub observer: &'a str,
    /// patch points 时间
    pub t_patch: &'a [f64],
    /// patch points 状态（7D）
    pub state_patch: &'a [[f64; 7]],
}

/// 同伦法求解器。
///
/// 使用 L2-L1 同伦从能量最优过渡到燃料最优。
pub struct HomotopySolver {
    config: HomotopyConfig,
}

impl HomotopySolver {
    /// 创建新的同伦法求解器。
    pub fn new(config: HomotopyConfig) -> Self {
        Self { config }
    }

    /// 求解燃料最优控制。
    ///
    /// # 参数
    /// - `ctx`: 问题定义（力模型、传播系、patch points）
    /// - `thrust`: 推力配置
    /// - `verbose`: 是否输出进度
    ///
    /// # 返回
    /// 同伦法求解结果
    pub fn solve(
        &self,
        ctx: &HomotopyProblem<'_>,
        thrust: &ThrustParams,
        verbose: bool,
    ) -> Result<HomotopyResult, String> {
        let mut lambda = self.config.lambda_init;
        let mut lambda_history = vec![lambda];
        let mut residual_history = Vec::new();
        let mut total_iterations = 0;

        // 1. 求解能量最优问题（λ = 0）
        if verbose {
            eprintln!("同伦法求解：λ = {:.2}（能量最优）", lambda);
        }

        let mut current_result = self.solve_with_lambda(ctx, thrust, lambda, verbose)?;

        total_iterations += current_result.iterations;
        residual_history.push(current_result.max_residual);

        // 2. 同伦迭代（λ: 0 → 1）
        while lambda < self.config.lambda_final {
            lambda = (lambda + self.config.lambda_step).min(self.config.lambda_final);
            lambda_history.push(lambda);

            if verbose {
                eprintln!("同伦法求解：λ = {:.2}", lambda);
            }

            // 使用上一步的解作为初始猜测
            // 注意：当前使用 6D 状态（忽略质量），后续需要扩展到 7D
            let prev_state_patch: Vec<[f64; 7]> = current_result
                .state_patch
                .iter()
                .map(|s| {
                    // 从 6D 状态恢复 7D（假设质量恒定）
                    let mut s7 = [0.0; 7];
                    s7[0..6].copy_from_slice(s);
                    s7[6] = 1500.0; // 默认质量 1500 kg
                    s7
                })
                .collect();

            let ctx_next = HomotopyProblem {
                forces: ctx.forces,
                observer: ctx.observer,
                t_patch: &current_result.t_patch,
                state_patch: &prev_state_patch,
            };
            current_result = self.solve_with_lambda(&ctx_next, thrust, lambda, verbose)?;

            total_iterations += current_result.iterations;
            residual_history.push(current_result.max_residual);

            // 检查收敛
            if current_result.max_residual < self.config.tolerance {
                if verbose {
                    eprintln!(
                        "同伦法收敛：λ = {:.2}, 残差 = {:.2e}",
                        lambda, current_result.max_residual
                    );
                }
                break;
            }
        }

        Ok(HomotopyResult {
            t_patch: current_result.t_patch,
            state_patch: current_result
                .state_patch
                .iter()
                .map(|s| {
                    // 从 6D 状态恢复 7D（假设质量恒定）
                    let mut s7 = [0.0; 7];
                    s7[0..6].copy_from_slice(s);
                    s7[6] = 1500.0; // 默认质量 1500 kg
                    s7
                })
                .collect(),
            converged: current_result.converged,
            total_iterations,
            max_residual: current_result.max_residual,
            lambda_history,
            residual_history,
        })
    }

    /// 使用指定的 λ 值求解。
    ///
    /// 这里使用多重打靶法求解，控制律使用 Sigmoid 平滑。
    fn solve_with_lambda(
        &self,
        ctx: &HomotopyProblem<'_>,
        _thrust: &ThrustParams,
        _lambda: f64,
        verbose: bool,
    ) -> Result<MultipleShootingRustResult, String> {
        // TODO: 实现带同伦参数的多重打靶求解
        // 当前使用标准多重打靶作为占位
        // 实际实现需要：
        // 1. 修改多重打靶以支持 7D 状态向量
        // 2. 添加 Sigmoid 平滑控制律
        // 3. 计算协态变量（costate）

        // 占位：使用 6D 状态向量（忽略质量）
        let state_patch_6d: Vec<[f64; 6]> = ctx
            .state_patch
            .iter()
            .map(|s| [s[0], s[1], s[2], s[3], s[4], s[5]])
            .collect();

        multiple_shooting_correct(
            ctx.forces,
            ctx.observer,
            ctx.t_patch,
            &state_patch_6d,
            false, // 固定时间
            false, // 不固定首节点（homotopy 占位实现，保持原有语义）
            self.config.max_iter,
            self.config.tolerance,
            self.config.rtol,
            None,
            verbose,
        )
    }
}

/// Sigmoid 平滑控制律。
///
/// u = 1/2 [1 + tanh(S/ε)]
///
/// 其中 S 是切换函数（switching function），由协态决定。
///
/// # 参数
/// - `s`: 切换函数值
/// - `epsilon`: 平滑参数
///
/// # 返回
/// 平滑后的控制值 u ∈ [0, 1]
pub fn sigmoid_control(s: f64, epsilon: f64) -> f64 {
    0.5 * (1.0 + (s / epsilon).tanh())
}

/// 计算切换函数（switching function）。
///
/// S = λ_v^T û - λ_m / (Isp * g0)
///
/// 其中：
/// - λ_v：速度协态
/// - û：推力方向单位向量
/// - λ_m：质量协态
/// - Isp：比冲
/// - g0：标准重力加速度
///
/// # 参数
/// - `lambda_v`: 速度协态（3D）
/// - `u_hat`: 推力方向单位向量（3D）
/// - `lambda_m`: 质量协态
/// - `isp`: 比冲（s）
///
/// # 返回
/// 切换函数值 S
pub fn switching_function(lambda_v: &[f64; 3], u_hat: &[f64; 3], lambda_m: f64, isp: f64) -> f64 {
    let g0 = 9.81; // m/s²
    lambda_v[0] * u_hat[0] + lambda_v[1] * u_hat[1] + lambda_v[2] * u_hat[2] - lambda_m / (isp * g0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_control() {
        // 测试 Sigmoid 控制律
        assert!((sigmoid_control(0.0, 0.01) - 0.5).abs() < 1e-10);
        assert!((sigmoid_control(1.0, 0.01) - 1.0).abs() < 1e-3);
        assert!((sigmoid_control(-1.0, 0.01) - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_switching_function() {
        // 测试切换函数
        let lambda_v = [1.0, 0.0, 0.0];
        let u_hat = [1.0, 0.0, 0.0];
        let lambda_m = 0.1;
        let isp = 3000.0;

        let s = switching_function(&lambda_v, &u_hat, lambda_m, isp);

        // S = 1.0 - 0.1 / (3000 * 9.81) ≈ 1.0
        assert!(s > 0.99 && s < 1.01);
    }
}
