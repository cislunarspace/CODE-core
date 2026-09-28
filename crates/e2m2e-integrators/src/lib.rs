//! e2m2e 积分器 crate：`e2m2e._integrators` 扩展的 PyO3 绑定与编译入口。
//!
//! 纯数学积分器（Butcher 表、RK/ABM/Cowell、solve_ivp）在
//! `e2m2e-propagation` crate，力模型在 `e2m2e-forces`，SPICE FFI 在
//! `e2m2e-spice`；本 crate 只做 PyO3 绑定与 shooting 算法。
//!
//! 仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。

use pyo3::prelude::*;

pub mod bindings;
pub mod center_manifold;
#[cfg(feature = "spice")]
pub mod differential_correction;
pub mod family;
pub mod family_generation;
pub mod force_parse;
#[cfg(feature = "spice")]
pub mod frame_convert;

pub mod hjb;
#[cfg(feature = "spice")]
pub mod lowthrust;
#[cfg(feature = "spice")]
pub mod multiple_shooting;
pub mod normal_form;
pub mod nsga2;
pub mod planar_pal;
pub mod qf_cm;
#[cfg(feature = "spice")]
pub mod segmented_shooting;

use e2m2e_propagation::multistep_methods::MultistepMethod;
use e2m2e_propagation::rk_methods::RkMethod;

/// 解析 abi-version.txt 的纯数字内容为 u32（const fn，编译期求值）。
const fn parse_abi_version(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut result: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b >= b'0' && b <= b'9' {
            result = result * 10 + (b - b'0') as u32;
        }
        i += 1;
    }
    result
}

/// 从 abi-version.txt 读取（单一来源），build.rs 同步生成 Python 侧 _rust_abi.py。
///
/// # 版本沿革
///
/// abi-version 只在**新增/改 pyfunction 边界** 时 bump（Rust 内部函数签名
/// 变更不 bump（它们不是 Python 可见的 ABI）。每次 bump 须在本节补行：
///
/// - **v1**（5b616cc）：初始 ABI 版本戳 + 统一网关 ``_check_rust_abi`` 。
/// - **v2** （3b28353）：新增 ``propagate_with_state_py`` （EphemerisDynamics
///   纯状态 Rust 路径）。
/// - **v3** （ff63403）：新增 ``transfer_grid_search_serial_py`` +
///   ``TransferPointResult`` pyclass（转移网格搜索串行评估）。
/// - **v4**：新增 ``spice_spkezr`` + ``spice_pxform`` （Rust CSPICE
///   实例诊断查询 API）。
/// - **v5**：多重与分段打靶结果将公开 ``converged`` 替换为
///   ``status`` / ``cause`` / ``message`` 三元组。
/// - **v6**（606847c）：新增 ``propagate_segments_py`` （分段打靶逐段
///   积分下沉）与 ``frame_convert`` 批量入口（坐标/历元/星历批量转换）。
/// - **v7**：新增 ``pal_f_df_tangent_py`` + ``pal_newton_step_py``
///   （伪弧长延拓数值内核：F/dF/切向量计算与 PAL 牛顿迭代）。
/// - **v8**：新增 ``planar_full_period_pal_py`` 与
///   ``PlanarPalRustResult`` ，为 SPO/LPO 平面全周期伪弧长延拓提供 Rust
///   数值内核。
/// - **v9**：新增 ``qlaw_propagate_py`` 与
///   ``qlaw_segment_direction_py`` （Q-law 低推力初猜的反馈积分与 Q 函数
///   评估内核）。
/// - **v10**：新增 ``nsga2_*_py`` （NSGA-II 约束排序、选择、
///   SBX 交叉与多项式变异算子）。
/// - **v11**：新增低推力打靶批量评估与配点缺陷批量评估入口，
///   将低推力直接法的重复数值评估下沉 Rust。
/// - **v12**：新增 WSB 三维网格搜索与低能转移流形截面态配对入口。
/// - **v13**：新增 ``differential_correction_cr3bp_py`` ，将 CR3BP
///   单段微分修正的残差、雅可比、Newton 修正与收敛状态机下沉 Rust。
/// - **v14**：新增 ``collinear_center_modes_py`` 、
///   ``lissajous_bounded_trajectory_py`` 与 ``orbit_family_metric_py`` ，将
///   Lissajous 中心模态轨迹和族几何度量下沉 Rust。
/// - **v15**：新增 ``generate_cr3bp_family_py`` ，将七类轨道族的
///   种子、延拓、筛选与结构化终止收进单次 Rust 调用。
/// - **v16**：新增 ``manifold_seeds_py`` 与 ``manifold_propagate_py`` ，
///   将不变流形种子生成与批量传播调度下沉 Rust。
/// - **v17**：新增 ``poly_poisson_py`` / ``poly_simplify_py`` /
///   ``polylist_simplify_py`` / ``keys_by_order_py`` / ``trim_degree_py`` ，
///   将 normal_form 数值多项式核完整下沉 Rust。
/// - **v18**：新增 ``qf_to_cm_py`` 与 ``cm_to_qf_py`` ，将 QF↔CM
///   高阶 Lie 流（12 实维分裂复积分）下沉 Rust，关闭复值积分例外。
/// - **v19**：新增 ``center_manifold_reduce_py`` ，将中心流形两步
///   Lie 同调化简（频域 W、Poisson 链、虚/实基底变换）完整下沉 Rust。
/// - **v20**：新增 ``generate_cr3bp_family_windows_py`` ，按 Jacobi
///   能量窗口批量生成轨道族（延拓 trace 只走一次，各窗口分别筛选成员）。
/// - **v21**：新增 ``solve_hjb_py`` （HJB 结构网格求解通用入口，
///   动力学标识 + 参数表）与 ``solve_planar_lowthrust_hjb_py`` （geo-nrho
///   既有签名的兼容包装）。
/// - **v22**：新增 ``propagate_compiled_ias15_py`` ，
///   ``propagate_compiled_stm_py`` 增可选 ``sens_params`` （力模型参数
///   敏感度变分方程，5.8.8；当时未在此补行，此处补记）。
/// - **v23**：新增 ``solve_ivp_events_kernel_py`` （事件路径 EOM 内核分派：
///   动力学标识 + 参数表，复用 e2m2e-forces 的 CR3BP/BCR4BP EOM/STM 内核，
///   issue #594）。
/// - **v24**：RO（共振轨道族）加入 ``generate_cr3bp_family_py`` /
///   ``generate_cr3bp_family_windows_py`` 的族分派（两入口各增可选关键字
///   参数 ``resonance_p`` / ``resonance_q``），``orbit_family_metric_py``
///   增 ``earth-distance`` 度量（issue #627）。
/// - **v25**：新增 ``nrlmsise00_density_py``（NRLMSISE-00 密度/温度查询绑定，
///   供 ``NRLMSISE00Atmosphere.density`` 调用；issue #637）。
/// - **v26**：新增 ``propagate_kepler_py``（conic 档二体 Kepler 封闭解传播
///   + 解析 STM，issue #739）。
///
/// 1→3 跳号实为 1→2→3 两次单步 bump，分别在上述两 commit；不存在跳过的
/// 中间版本。ADR 0018 记录的 ∂a/∂v 雅可比接口扩是 Rust 内部签名变更，未 bump。
const RUST_PY_ABI: u32 = parse_abi_version(include_str!("../abi-version.txt"));

/// 返回 Rust↔Python ABI 版本号（编译期常量，反映此 .pyd/.so 真实状态）。
#[pyfunction]
fn _py_abi_version() -> u32 {
    RUST_PY_ABI
}

/// 占位函数，用于验证 FFI 路径端到端通畅。
#[pyfunction]
fn hello_integrators() -> PyResult<String> {
    Ok("hello from e2m2e-integrators".to_string())
}

/// 把 Rust 内部 [`e2m2e_forces::PropagateError`] 翻译成 Python 异常。
///
/// 所有内部 ``PropagateError`` → ``e2m2e.exceptions.PropagationFailure``
/// （``E2M2EError`` 子类）。消息前缀都加 ``prefix`` （形如
/// "CR3BP propagation failed: ..."）。Python 侧据此按类型捕获，不再依赖
/// 错误消息字符串前缀匹配，改 Rust 措辞不影响 ``except PropagationFailure`` 。
pub(crate) fn propagate_error_to_pyerr(
    py: Python<'_>,
    prefix: &str,
    e: impl std::fmt::Display,
) -> PyErr {
    let msg = format!("{prefix}: {e}");
    match py
        .import("e2m2e.exceptions")
        .and_then(|m| m.getattr("PropagationFailure"))
        .and_then(|cls| cls.call1((msg.clone(),)))
    {
        Ok(instance) => PyErr::from_value(instance),
        Err(_) => pyo3::exceptions::PyRuntimeError::new_err(msg),
    }
}

#[pymodule]
fn _integrators(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_py_abi_version, m)?)?;
    m.add_function(wrap_pyfunction!(hello_integrators, m)?)?;
    m.add_function(wrap_pyfunction!(normal_form::project_hamiltonian_qf_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        normal_form::build_cr3bp_hamiltonian_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        center_manifold::center_manifold_reduce_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(normal_form::poly_poisson_py, m)?)?;
    m.add_function(wrap_pyfunction!(normal_form::poly_simplify_py, m)?)?;
    m.add_function(wrap_pyfunction!(normal_form::polylist_simplify_py, m)?)?;
    m.add_function(wrap_pyfunction!(normal_form::keys_by_order_py, m)?)?;
    m.add_function(wrap_pyfunction!(normal_form::trim_degree_py, m)?)?;
    m.add_function(wrap_pyfunction!(qf_cm::qf_to_cm_py, m)?)?;
    m.add_function(wrap_pyfunction!(qf_cm::cm_to_qf_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::steppers::rk_step, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::steppers::solve_ivp_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::steppers::solve_ivp_events_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::steppers::solve_ivp_events_kernel_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(bindings::steppers::multistep_step, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::steppers::cowell_step, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lambert::lambert_izzo_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lambert::lambert_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lambert::propagate_kepler_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::transfer_search::porkchop_grid_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::transfer_search::porkchop_grid_states_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::force_models::spherical_harmonic_accel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::force_models::solid_tide_step1,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::force_models::solid_tide_step2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(bindings::force_models::pole_tide, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_cr3bp_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_segments_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_cr3bp_stm_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_bcr4bp_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_bcr4bp_stm_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_cr3bp_megno_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_bcr4bp_megno_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::three_body::propagate_geocentric_fate_map_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::geometry::compute_distance_series_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::geometry::compute_min_distance_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::geometry::detect_intersection_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::geometry::detect_local_minimum_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(bindings::geometry::check_collision_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::qlaw::qlaw_propagate_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::qlaw::qlaw_segment_direction_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        lowthrust::lowthrust_shooting_evaluate_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        lowthrust::lowthrust_collocation_defects_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        lowthrust::lowthrust_discrete_collocation_defects_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        lowthrust::lowthrust_variable_time_collocation_defects_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(nsga2::nsga2_sort_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        nsga2::nsga2_environmental_selection_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(nsga2::nsga2_tournament_selection_py, m)?)?;
    m.add_function(wrap_pyfunction!(nsga2::nsga2_variation_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::pal::pal_f_df_tangent_py, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::pal::pal_newton_step_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::transfer_search::wsb_search_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::transfer_search::low_energy_patch_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(bindings::manifold::manifold_seeds_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::manifold::manifold_propagate_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::transfer_search::transfer_grid_search_serial_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::transfer_search::transfer_grid_search_py,
        m
    )?)?;
    m.add_class::<bindings::transfer_search::WsbCandidate>()?;
    m.add_class::<bindings::transfer_search::LowEnergyPatchCandidate>()?;
    m.add_class::<bindings::transfer_search::TransferPointResult>()?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::spice::spice_poc_body_position,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::spice_furnsh, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::spice_unload, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::spice_spkezr, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::spice_pxform, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::force_models::third_body_acceleration,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::force_models::indirect_term_acceleration,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::force_models::gravity_field_acceleration,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::force_models::srp_acceleration,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::compiled::propagate_compiled, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_compiled_lowthrust,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_compiled_lowthrust_sensitivity,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::enable_ephem_cache, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::disable_ephem_cache, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(bindings::spice::ephem_ffi_call_count, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::spice::reset_ephem_ffi_call_count,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::augmented_eom_7d_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_with_stm_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_with_state_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_compiled_stm_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::compiled::propagate_compiled_ias15_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        bindings::force_models::nrlmsise00_density_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        differential_correction::differential_correction_cr3bp_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        multiple_shooting::multiple_shooting_correct_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_class::<multiple_shooting::MultipleShootingRustResult>()?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        segmented_shooting::segmented_shooting_correct_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_class::<segmented_shooting::SegmentedShootingResult>()?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        frame_convert::batch_synodic_to_j2000_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(
        frame_convert::batch_j2000_to_synodic_py,
        m
    )?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(frame_convert::batch_body_states_py, m)?)?;
    #[cfg(feature = "spice")]
    m.add_function(wrap_pyfunction!(frame_convert::batch_et_to_utc_py, m)?)?;
    m.add_function(wrap_pyfunction!(planar_pal::planar_full_period_pal_py, m)?)?;
    m.add_class::<planar_pal::PlanarPalRustResult>()?;
    m.add_function(wrap_pyfunction!(family::collinear_center_modes_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        family::lissajous_bounded_trajectory_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(family::orbit_family_metric_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        family_generation::generate_cr3bp_family_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        family_generation::generate_cr3bp_family_windows_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(hjb::solve_hjb_py, m)?)?;
    m.add_function(wrap_pyfunction!(hjb::solve_planar_lowthrust_hjb_py, m)?)?;
    m.add_class::<RkMethod>()?;
    m.add_class::<MultistepMethod>()?;
    m.add_class::<bindings::steppers::StepResult>()?;
    m.add_class::<bindings::steppers::MultistepResult>()?;
    m.add_class::<bindings::steppers::CowellResult>()?;

    // Rust 物理常量同源核对入口：把 e2m2e-propagation 从
    // constants.toml 生成的常量以 `_propagation_constants` 子模块挂出，
    // 供 Python 侧逐位对拍。
    m.add_submodule(&e2m2e_propagation::_propagation_constants_module_bound(
        m.py(),
    )?)?;

    Ok(())
}
