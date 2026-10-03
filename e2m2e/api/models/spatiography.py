"""地月空间分区分析（spatiography）请求/响应模型。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from .shared import ResultResponse, _ApiModel

__all__ = [
    "SpatiographyScalesRequest",
    "SpatiographyScalesResponse",
    "SpatiographyClassifyRequest",
    "SpatiographyClassifyResponse",
    "SpatiographyIntervalClassifyRequest",
    "SpatiographyIntervalClassifyResponse",
    "SpatiographyBoundariesRequest",
    "SpatiographyBoundariesResponse",
    "SpatiographyAtlasRequest",
    "SpatiographyAtlasResponse",
    "SpatiographyMapRequest",
    "SpatiographyMapResponse",
]


class SpatiographyScalesRequest(_ApiModel):
    """分区解析尺度计算输入（spatiography，ADR 0041）。"""

    system: Literal["earth_moon"] = Field(
        default="earth_moon",
        description="天体系统；当前仅支持地月（Primer §5 口径，Simon 1994 月根数）",
    )
    elements: list[str] = Field(
        default_factory=list,
        description="要计算的量（scales 键名）子集；空 = 全部。含 laplace_radius_geolunar、"
        "hill_radius_moon、soi_laplace_moon、battin_moon_earthward_km 等",
    )


class SpatiographyScalesResponse(ResultResponse):
    """分区解析尺度输出（Rosengren et al. 2026 §5 闭式边界，物理单位 km）。"""

    scales: dict[str, float]
    libration_points_km: dict[str, list[float]] = Field(
        description="L1–L5 精确解位置（会合系质心原点，km，z=0）；精确求根口径"
        "（论文表值 57868/64347 km 为级数近似注记）"
    )
    jacobi_criticals: dict[str, float] = Field(
        description="平动点处临界 Jacobi 值 C1..C5（Parker 约定，无量纲）"
    )
    resonance_ladder: list[dict[str, Any]] = Field(
        description="共振名义中心阶梯（Table 1/2 全表）：label/k/k_body/a_km/"
        "a_over_a_moon 或 rho_over_moon_radius/period_days/secondary"
    )
    constants_used: dict[str, float] = Field(description="Primer 常数集（数值）")
    citation: str = Field(description="常数集出处")
    details: dict[str, Any]


class SpatiographyClassifyRequest(_ApiModel):
    """分区区域分类输入。"""

    states: list[list[float]] = Field(
        min_length=1,
        description="状态列表，每项 [x,y,z,vx,vy,vz]；坐标系与单位由 frame 声明",
    )
    frame: Literal["synodic_barycentric_km", "synodic_barycentric_nd"] = Field(
        description="状态的数据系标签（ADR 0040 state_frame 词汇，本工具首批启用"
        " synodic_barycentric_nd）：synodic_barycentric_km = 地月会合旋转系、质心原点、"
        "物理单位 km/km/s；synodic_barycentric_nd = 同系无量纲（长度 a☾、速度 a☾·n，"
        "Primer 常数口径）"
    )
    reference: Literal["table1", "table4"] = Field(
        default="table1",
        description="分区口径：table1 = 论文 Table 1 五省语义；table4 = 附录 B 六"
        "制图带（deliberate-overlap，相邻区端部有意重叠）",
    )
    include_overlaps: bool = Field(
        default=True,
        description="重叠带返回多标签（False 时按优先序取主标签）",
    )


class SpatiographyClassifyResponse(ResultResponse):
    """分区区域分类输出。"""

    zone_ids: list[list[int]] = Field(
        description="逐状态区域 id 列表（重叠带多值、升序），名称见 legend"
    )
    legend: dict[str, str] = Field(
        description="区域 id → 名称（terrestrial / cislunar_inner_secular /"
        " cislunar_outer_resonant / circumlunar / translunar / heliocentric；"
        "cislunar 为狭义带级名，非伞式）"
    )
    diagnostics: list[dict[str, Any]] = Field(
        description="逐状态诊断：r_geocentric_km / rho_selenocentric_km /"
        " a_geocentric_km / a_over_a_moon / jacobi_constant / topology_case /"
        " open_necks"
    )
    details: dict[str, Any]


class SpatiographyIntervalClassifyRequest(_ApiModel):
    """带不确定度的分区区域分类输入（状态盒，issue #785）。"""

    state: list[float] = Field(
        min_length=6,
        max_length=6,
        description="标称 6 维状态 [x,y,z,vx,vy,vz]；坐标系与单位由 frame 声明",
    )
    half_widths: list[Annotated[float, Field(ge=0.0)]] = Field(
        min_length=6,
        max_length=6,
        description="对角盒各分量半宽 [hx,hy,hz,hvx,hvy,hvz]，与 state 同 frame"
        " 同单位；各分量 ≥ 0，全零时退化为点判定（与 spatiography_classify"
        " 逐位一致）",
    )
    frame: Literal["synodic_barycentric_km", "synodic_barycentric_nd"] = Field(
        description="状态的数据系标签（ADR 0040 state_frame 词汇，本工具首批启用"
        " synodic_barycentric_nd）：synodic_barycentric_km = 地月会合旋转系、质心原点、"
        "物理单位 km/km/s；synodic_barycentric_nd = 同系无量纲（长度 a☾、速度 a☾·n，"
        "Primer 常数口径）"
    )
    reference: Literal["table1", "table4"] = Field(
        default="table1",
        description="分区口径：table1 = 论文 Table 1 五省语义；table4 = 附录 B 六"
        "制图带（deliberate-overlap，相邻区端部有意重叠）",
    )
    truncation_order: int = Field(
        default=3,
        ge=1,
        le=12,
        description="微分代数截断阶（issue #784 DA 原语）：判据链按该阶 Taylor"
        " 多项式展开后保守包围，截断余项 O(|h|^(k+1))；1–12",
    )


class SpatiographyIntervalClassifyResponse(ResultResponse):
    """带不确定度的分区区域分类输出（区间界 + possible/certain 双标签）。"""

    r_geocentric_km: list[float] = Field(description="地心距区间 [lo, hi]（km）")
    rho_selenocentric_km: list[float] = Field(description="月心距区间 [lo, hi]（km）")
    a_geocentric_km: list[float] = Field(
        description="地心 osculating 半长轴区间 [lo, hi]（km）；inf 表示含逃逸态"
    )
    a_over_a_moon: list[float] = Field(description="a/a☾ 区间 [lo, hi]；inf 表示含逃逸态")
    jacobi_constant: list[float] = Field(description="Jacobi 常数区间 [lo, hi]")
    topology_case_min: int = Field(description="Hill 拓扑 Case 区间下界（1..5）")
    topology_case_max: int = Field(description="Hill 拓扑 Case 区间上界（1..5）")
    ambiguous_critical_values: list[str] = Field(
        description="被区间严格跨越的临界 Jacobi 值（C1..C4 子集；跨界歧义显式列出）"
    )
    open_necks: list[str] = Field(
        description="已开启颈口（取 Jacobi 下界的最开情形，possible 口径）"
    )
    zone_ids_possible: list[int] = Field(description="区间可能触及的分区 id（升序），名称见 legend")
    zone_ids_certain: list[int] = Field(
        description="区间整体落入的分区 id（升序；zone_ids_possible 的子集）"
    )
    legend: dict[str, str] = Field(
        description="区域 id → 名称（同 spatiography_classify；cislunar 为狭义带级名，非伞式）"
    )
    details: dict[str, Any]


class SpatiographyBoundariesRequest(_ApiModel):
    """分区边界几何输入（可视化数据层）。"""

    kind: Literal["synodic_planar", "ae_curves"] = Field(
        default="synodic_planar",
        description="synodic_planar = 地月会合旋转系（质心原点、z=0 平面）边界圆族"
        "与 Battin 非对称曲线、L1–L5；ae_curves = 地心 osculating (a,e) 根数平面"
        "走廊曲线族（掠地线/Hill 远点线/月 Hill 相遇走廊/GEO 穿越线/共振竖线/"
        "Tisserand 等值线；crossing diagnostics 而非物理面）",
    )
    boundary_set: list[str] | None = Field(
        default=None,
        description="元素/曲线族名子集；空 = 全部（名称见 kind 对应支持清单）",
    )
    resolution: int = Field(
        default=720,
        ge=8,
        le=4096,
        description="曲线离散点数（synodic_planar 为闭合曲线点数；ae_curves 为每条曲线采样点数）",
    )


class SpatiographyBoundariesResponse(ResultResponse):
    """分区边界几何输出（前端只做归一与绘制，不做数值计算）。"""

    elements: list[dict[str, Any]] = Field(
        description="边界元素：kind=circle（center_km/radius_km/points_km）|"
        " polyline（center_km/points_km）| point（center_km，会合系质心原点 km）|"
        " curve_ae（points_ae=[a_km,e]）| vertical_ae（a_km）"
    )
    state_frame: Literal["synodic_barycentric_km", "element_space_ae"] = Field(
        description="几何的数据系标签：synodic_planar 输出为 synodic_barycentric_km"
        "（地月会合旋转系、质心原点、物理 km）；ae_curves 输出在根数空间，标签为"
        " element_space_ae（横轴 a 单位 km、纵轴 e 无量纲，ADR 0041 登记的新词汇）"
    )
    details: dict[str, Any]


class SpatiographyMapRequest(_ApiModel):
    """六域两层天图输入（Primer §7.3 / Table 4，ADR 0041 Phase 3c）。

    网格初值按命名场景（2027-08-02 日全食历元、(Ω,ω,M) 固定角、
    i = 月轨面、反 aligned 拱线）生成；逐格传播 EM/EMS 点质量模型
    （MEGNO + 命运两层）。网格分辨率与积分窗按需收缩——CI 走抽查
    小网格，全量制图走 scripts/ 手动（ADR 0037 预算口径）。
    """

    zone: Literal["SC", "CR", "CG", "IT", "OT", "TF"] = Field(
        description="制图域（Table 4 行序）：SC = secular cislunar；"
        "CR = cislunar resonant；CG = circumlunar gateway；IT/OT = 内/"
        "外 translunar；TF = translunar fringe"
    )
    model: Literal["em", "ems"] = Field(
        default="em",
        description="动力学模型：em = 椭圆点质量地月（星历初值后孤立"
        "演化）；ems = +太阳点质量（架构持续性检验）",
    )
    n_a: int = Field(default=12, ge=2, le=400, description="a/a☾ 轴格点数")
    n_e: int = Field(default=8, ge=2, le=400, description="e 轴格点数")
    e_min: float = Field(default=0.0, ge=0.0, lt=1.0, description="偏心率下限")
    e_max: float = Field(default=0.9, gt=0.0, lt=1.0, description="偏心率上限")
    span_years: float | None = Field(
        default=None,
        gt=0.0,
        le=60.0,
        description="积分窗（年）；None = Table 4 区带缺省（SC/CR/CG 19、IT 38、OT/TF 57）",
    )
    rtol: float = Field(default=1e-9, gt=1e-14, lt=1e-4, description="积分相对容差")
    max_step_hours: float = Field(
        default=6.0, gt=0.1, le=48.0, description="最大步长（小时；近点漏检防护）"
    )
    stop_on_terminal: bool = Field(default=True, description="终端事件（再入/撞月/逃逸）早停")
    ybar_ordered_band: float = Field(
        default=0.2, gt=0.0, description="|Ȳ−2| ≤ 带内记 ordered（Phase 3b 定标）"
    )
    ybar_chaotic_excess: float = Field(
        default=1.0, gt=0.0, description="Ȳ ≥ 2+excess 记 chaotic（Phase 3b 定标）"
    )


class SpatiographyMapResponse(ResultResponse):
    """六域两层天图输出（Ȳ 场 + 命运场 + 诊断量；大数组可走 sidecar 帧）。"""

    zone: str = Field(description="制图域名")
    model: str = Field(description="动力学模型（em/ems）")
    span_years: float = Field(description="实际积分窗（年）")
    a_over_a_moon: list[float] = Field(description="a/a☾ 轴（升序）")
    e_grid: list[float] = Field(description="偏心率轴（升序）")
    ybar_field: list[list[float | None]] | None = Field(
        description="Ȳ 场（n_a×n_e，行序 = a 轴）；终端短路格为 None；"
        "sidecar 帧路径下为 None 占位（帧序第 1 帧）"
    )
    fate_ids: list[list[int]] | None = Field(
        description="命运类 id 场（索引 fate_legend；sidecar 帧路径下为 None 占位，"
        "帧序第 2 帧，以 f32 编码的类 id）"
    )
    t_escape_years_field: list[list[float | None]] | None = Field(
        description="首次逃逸时刻场（年，未逃逸 None；帧序第 3 帧）"
    )
    min_r_sel_km_field: list[list[float | None]] | None = Field(
        description="最小月心距场（km；帧序第 4 帧）"
    )
    min_r_geo_km_field: list[list[float | None]] | None = Field(
        description="最小地心距场（km；帧序第 5 帧）"
    )
    fate_legend: dict[str, str] = Field(description="命运类 id → 名称（八类）")
    thresholds: dict[str, float] = Field(description="命运分类阈值（登记自由参数回显）")
    scenario: dict[str, Any] = Field(description="命名场景回显（历元、固定角、出处）")
    diagnostic_focus: str = Field(description="该区诊断聚焦（Table 4 Description 口径）")
    details: dict[str, Any]


class SpatiographyAtlasRequest(_ApiModel):
    """共振图集输入（Primer §4.2–§4.4 / §5.3，ADR 0041 Phase 3a）。"""

    products: list[str] = Field(
        default_factory=lambda: ["gallardo_widths", "secular_loci", "vzlk_portrait"],
        description="输出产品子集：gallardo_widths = Gallardo 半解析共振半宽包络"
        "（式 100–104，Fig. 8）；secular_loci = 拱线驻定 loci（式 75–78，Fig. 5）；"
        "vzlk_portrait = vZLK 相图（式 64–68）与时间尺度（式 69–71）",
    )
    resonance_pairs: list[list[int]] | None = Field(
        default=None,
        description="Gallardo 包络的 [k, k_body] 互素对（k 为卫星侧整数）；"
        "None = Table 1 内月 9 条 + 1:1 + 外月 9 条",
    )
    e_min: float = Field(default=0.0, ge=0.0, lt=1.0, description="包络偏心率下限")
    e_max: float = Field(default=0.9, gt=0.0, lt=1.0, description="包络偏心率上限")
    n_e: int = Field(default=19, ge=3, le=201, description="包络偏心率切片数")
    varpi_offset_deg: float = Field(
        default=180.0,
        description="卫星近日点黄经相对月球近日点黄经的夹角（缺省 180° 反平行，"
        "与 §7.3 制图切片同约定）",
    )
    n_sigma: int = Field(default=72, ge=12, le=720, description="共振角采样点数")
    n_lambda: int = Field(default=180, ge=24, le=1440, description="λ☾ 每 2π 的采样数")
    locus_e_slices: list[float] = Field(
        default_factory=lambda: [0.0, 0.3, 0.6],
        description="secular loci 的偏心率切片",
    )
    a_over_a_moon_min: float = Field(
        default=1.02, gt=1.0, description="translunar loci 半长轴下限（a/a☾）"
    )
    a_over_a_moon_max: float = Field(
        default=3.9, gt=1.0, description="translunar loci 半长轴上限（a/a☾，地球 Hill 界内）"
    )
    n_locus: int = Field(default=73, ge=5, le=1001, description="loci 半长轴采样数")
    vzlk_c1: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="vZLK 相图第一积分 c1 = (1−e²)cos²I（式 67）；c1 < 0.6 存在分离线",
    )


class SpatiographyAtlasResponse(ResultResponse):
    """共振图集输出（元素空间曲线 + vZLK 标量，前端只做归一与绘制）。"""

    elements: list[dict[str, Any]] = Field(
        description="曲线元素：kind=envelope_ae（points=[a_km,e]，半宽包络上下沿）|"
        " vertical_ae（a_km，名义中心竖线）| locus_ai（points=[a_km,I_deg]，"
        "拱线驻定 loci）| portrait_curve（points=[omega_deg,y]，c2 等值线，"
        "y=sqrt(1−e²)；附带 c2 等值）"
    )
    state_frames: dict[str, str] = Field(
        description="kind → 数据系标签：envelope_ae/vertical_ae = element_space_ae；"
        "locus_ai = element_space_ai（新词汇，ADR 0041 Phase 3 登记）；"
        "portrait_curve = vzlk_phase_plane（新词汇，同上）"
    )
    vzlk: dict[str, float] = Field(
        description="vZLK 标量：critical_inclination_deg（式 64）、nu_vzlk_rad_s 与"
        " t_vzlk_days（式 69/71，取 a=a☾ 处）等"
    )
    details: dict[str, Any]
