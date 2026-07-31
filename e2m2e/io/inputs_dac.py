"""DFH inputs-dac.txt 输入文件生成器（功能码 1 设计 / 2 控制 / 4 预报）。

文件为定行号格式：功能码 1/2/3 时 245 行，功能码 4（轨道预报）时 274
行；``//`` 之后是注释。布局固定——6 个轨道设计块 + 控制段 + 转移段 +
（功能 4 追加的）预报段。

生成策略照 MATLAB ``fmt_inputs_design.m`` / ``fmt_inputs_control.m`` /
``fmt_inputs_propagate.m``：以 golden 模板（``data/inputs-dac.golden``，
245 行）为底，只重建与所请求功能相关的参数块，其余各行逐字节保留。
这样 DFH exe 看到的上下文与黄金样本验证时完全一致。转移参数块的重建
留待后续阶段。功能 4 预报段的行布局例外：MATLAB 版本多插一行分隔符
（275 行），exe 按定行号读会错位，这里按说明文档与 exe 实跑修正为
274 行。

数值渲染规则照 MATLAB ``valrow.m`` / ``epoch_row.m``：

- 标量：整数值按 ``%d``，其余按 ``%g``（6 位有效数字）；
- 历元行：6 个分量各按 ``%g``、双空格分隔，``//`` 位于第 24 列；
- 设计块数值行 ``//`` 位于第 30 列；摄动开关块第 31 列；DYB 块第 30 列。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_DYB",
    "DEFAULT_PERTURBATION",
    "format_inputs_control",
    "format_inputs_design",
    "format_inputs_propagate",
    "write_inputs_dac",
]

_GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "inputs-dac.golden"

#: 摄动开关默认值（与 MATLAB fmt_perturb_block.m 一致）
DEFAULT_PERTURBATION: dict[str, int] = {
    "sun_body": 1,
    "planets": 1,
    "earth_nonspherical": 1,
    "moon_nonspherical": 1,
    "solar_radiation": 2,
    "atmosphere": 0,
    "relativity": 0,
    "tide": 1,
    "coupling": 1,
}

#: DYB 系数默认值：DYB(1)=等效面质比 0.01 (m2/kg)，其余为相对 DYB(1) 的比值
DEFAULT_DYB: list[float] = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_PERTURB_LABELS = [
    "太阳的第三体引力摄动开关（=0，关闭；=1，打开）",
    "大行星的第三体引力摄动开关（=0，关闭；=1，打开）",
    "地球的非球形引力摄动开关（=0，关闭；=1，打开）",
    "月球的非球形引力摄动开关（=0，关闭；=1，打开）",
    "光压摄动参数（=0，关闭；=1，炮弹模型；=2，ECOM模型）",
    "大气摄动开关（=0，关闭；=1，打开）",
    "广义相对论效应修正（=0，关闭；=1，打开）",
    "地球的潮汐摄动（=0，关闭；=1，打开）",
    "地球非球形和大天体的耦合项（=0，关闭；=1，打开）",
]

_DYB_LABELS = [
    "等效面质比，也是DYB模型D方向上的常量分量DYB(1)（单位：m2/kg）",
    "DYB(2)相对DYB(1)的比值",
    "DYB(3)相对DYB(1)的比值",
    "DYB(4)相对DYB(1)的比值",
    "DYB(5)相对DYB(1)的比值",
    "DYB(6)相对DYB(1)的比值",
    "DYB(7)相对DYB(1)的比值",
    "DYB(8)相对DYB(1)的比值",
    "DYB(9)相对DYB(1)的比值",
]

#: 各轨道类型设计块在 golden 中的行范围（0 起始切片）与类型编号
_DESIGN_BLOCKS: dict[str, tuple[slice, int]] = {
    "DRO": (slice(4, 29), 1),
    "NRHO": (slice(30, 56), 2),
    "HALO": (slice(57, 83), 3),
    "LISSAJOUS": (slice(84, 112), 4),
    "L4": (slice(113, 140), 5),
    "L5": (slice(141, 168), 6),
}


def _fmt_g(v: float) -> str:
    """MATLAB ``%g``：6 位有效数字。"""
    return f"{float(v):g}"


def _valrow(value: Any, slash_col: int, comment: str) -> str:
    """格式化一行数值：``<value><填充>//<comment>``，``//`` 起始于第
    ``slash_col`` 列（1 起始计数）。语义同 MATLAB ``valrow.m``。"""
    if isinstance(value, str):
        valstr = value
    elif isinstance(value, (list, tuple)):
        valstr = "  ".join(_fmt_g(v) for v in value)
    else:
        v = float(value)
        valstr = str(int(round(v))) if v == round(v) and abs(v) < 1e15 else _fmt_g(v)
    return valstr.ljust(slash_col - 1) + "//" + comment


def _epoch_row(epoch: Any, comment: str) -> str:
    """历元行：``<年  月  日  时  分  秒>//<comment>``，``//`` 位于第 24 列。"""
    valstr = "  ".join(_fmt_g(v) for v in epoch)
    return valstr.ljust(23) + "//" + comment


def _perturb_block(perturbation: dict[str, int] | None, prefix: str = "") -> list[str]:
    """9 行摄动开关块，``//`` 位于第 31 列。

    ``prefix`` 为注释标签前缀（控制段的"用于轨道控制量计算的"等）。
    """
    vals = dict(DEFAULT_PERTURBATION)
    if perturbation:
        unknown = set(perturbation) - set(DEFAULT_PERTURBATION)
        if unknown:
            raise ValueError(f"未知摄动开关字段: {sorted(unknown)}")
        vals.update(perturbation)
    return [
        str(int(vals[key])).ljust(30) + "//" + prefix + label
        for key, label in zip(DEFAULT_PERTURBATION, _PERTURB_LABELS, strict=True)
    ]


def _dyb_block(dyb: Any, suffix: str = "") -> list[str]:
    """9 行 DYB 系数块，``%0.2f`` 格式，``//`` 位于第 30 列。

    ``suffix`` 为注释标签后缀（控制段的"（控制计算理论模型）"等）。
    """
    dyb = list(dyb)
    if len(dyb) != 9:
        raise ValueError(f"dyb 必须为 9 个分量，当前 {len(dyb)} 个")
    return [
        f"{float(v):.2f}".ljust(29) + "//" + label + suffix
        for v, label in zip(dyb, _DYB_LABELS, strict=True)
    ]


def _golden_lines() -> list[str]:
    return _GOLDEN_PATH.read_text(encoding="utf-8").splitlines()


def _require(opts: dict[str, Any], names: list[str]) -> None:
    missing = [n for n in names if opts.get(n) is None]
    if missing:
        raise ValueError(f"缺少必填参数: {', '.join(missing)}")


def _common_tail(
    perturbation: dict[str, int] | None,
    dyb: Any,
    earth_degree: int,
    moon_degree: int,
    output_step: float,
) -> list[str]:
    """设计块/预报段共用的 21 行：9 摄动 + 9 DYB + 2 阶次 + 1 输出步长。"""
    return [
        *_perturb_block(perturbation),
        *_dyb_block(dyb),
        _valrow(earth_degree, 30, "地球非球形引力位阶次数"),
        _valrow(moon_degree, 30, "月球非球形引力位阶次数"),
        _valrow(f"{float(output_step):.1f}", 24, "用户指定的星历输出间隔（单位：sec）"),
    ]


def _build_design_block(sel: str, o: dict[str, Any]) -> list[str]:
    """构建一个轨道设计块（仅内容行，不含前导 ``####`` 分隔符）。"""
    block: list[str] = []
    if sel == "DRO":
        _require(o, ["amplitude", "phase", "epoch", "duration"])
        block.append(
            _valrow(
                _fmt_g(o["amplitude"]),
                30,
                "待设计DRO轨道的振幅参数（单位：km，取值在月球半径~11万公里之间）",
            )
        )
        block.append(
            _valrow(_fmt_g(o["phase"]), 30, "待设计DRO轨道的初始相位参数（取值在0~1之间）")
        )
        block.append(
            _epoch_row(o["epoch"], "待设计DRO轨道设计的起始历元（年-月-日-时-分-秒格式的UTC）")
        )
        block.append(
            _valrow(
                _fmt_g(o["duration"]),
                30,
                "待设计DRO轨道的维持时间（单位：年；取值大于0，不大于20）",
            )
        )
    elif sel == "NRHO":
        _require(o, ["collinear_point", "north_south", "perilune_height", "epoch", "duration"])
        block.append(_valrow(o["collinear_point"], 30, "共线平动点编号（取值为1或2）"))
        block.append(
            _valrow(o["north_south"], 30, "北或南NRHO轨道（=1，北NRHO轨道；=2，南NRHO轨道）")
        )
        # 说明文档：第 33 行为两列——近月点高度 + 初始相位（0.01~0.99，
        # 近月点高度较低时初始相位需要明显大于 0）。相位缺列时 DFH 读入
        # 垃圾值，设计出的轨道数日内外即逃逸（实测全部变体如此）
        phase = 0.5 if o["phase"] is None else float(o["phase"])
        if not 0.01 <= phase <= 0.99:
            raise ValueError(f"NRHO phase 应在 0.01~0.99 之间，实际为 {phase}")
        block.append(
            f"{_fmt_g(o['perilune_height'])}  {_fmt_g(phase)}".ljust(30)
            + "//待设计NRHO轨道的近地点高度（单位：km，取值范围在100~10000之间）"
        )
        block.append(
            _epoch_row(o["epoch"], "待设计NRHO轨道设计的起始历元（年-月-日-时-分-秒格式的UTC）")
        )
        block.append(
            _valrow(
                _fmt_g(o["duration"]),
                30,
                "待设计NRHO轨道的维持时间（单位：年；取值大于0，不大于20）",
            )
        )
    elif sel == "HALO":
        _require(o, ["collinear_point", "amplitude", "phase", "epoch", "duration"])
        block.append(_valrow(o["collinear_point"], 30, "共线平动点编号（取值为1或2）"))
        block.append(
            _valrow(
                _fmt_g(o["amplitude"]),
                30,
                "待设计halo轨道的白道面外振幅参数（单位：km，取值在-73000~73000之间）",
            )
        )
        block.append(
            _valrow(_fmt_g(o["phase"]), 30, "待设计halo轨道的初始相位参数（取值在0~1之间）")
        )
        block.append(
            _epoch_row(o["epoch"], "待设计halo轨道设计的起始历元（年-月-日-时-分-秒格式的UTC）")
        )
        block.append(
            _valrow(
                _fmt_g(o["duration"]),
                30,
                "待设计halo轨道的维持时间（单位：年；取值大于0，不大于20）",
            )
        )
    elif sel == "LISSAJOUS":
        _require(
            o,
            [
                "collinear_point",
                "amplitude_in",
                "amplitude_out",
                "phase_in",
                "phase_out",
                "epoch",
                "duration",
            ],
        )
        block.append(_valrow(o["collinear_point"], 30, "共线平动点编号（取值为1, 2或3）"))
        block.append(
            _valrow(
                _fmt_g(o["amplitude_in"]),
                30,
                "待设计Lissajous轨道的白道面内振幅参数（单位：km。对L1、L2点，此值不超过7600km；对L3点，此值不超过100000km）",
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["amplitude_out"]),
                30,
                "待设计Lissajous轨道的白道面外振幅参数（单位：km。对L1、L2点，此值不超过7600km；对L3点，此值不超过100000km）",
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["phase_in"]),
                30,
                "待设计Lissajous轨道在白道面内的初始相位参数（取值在0~1之间）",
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["phase_out"]),
                30,
                "待设计Lissajous轨道在白道面外的初始相位参数（取值在0~1之间）",
            )
        )
        block.append(
            _epoch_row(
                o["epoch"], "待设计Lissajous轨道设计的起始历元（年-月-日-时-分-秒格式的UTC）"
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["duration"]),
                30,
                "待设计Lissajous轨道的维持时间（单位：年；取值大于0，不大于20）",
            )
        )
    else:  # L4 / L5
        _require(o, ["amplitude_in", "amplitude_out", "phase_in", "phase_out", "epoch", "duration"])
        block.append(
            _valrow(
                _fmt_g(o["amplitude_in"]),
                30,
                f"待设计{sel}点轨道的白道面内振幅参数（单位：km；此值不超过10000km）",
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["amplitude_out"]),
                30,
                f"待设计{sel}点轨道的白道面外振幅参数（单位：km；此值不超过76000km）",
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["phase_in"]),
                30,
                f"待设计{sel}点轨道在白道面内的初始相位参数（取值在0~1之间）",
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["phase_out"]),
                30,
                f"待设计{sel}点轨道在白道面外的初始相位参数（取值在0~1之间）",
            )
        )
        # golden 在 L4/L5 块的历元/维持时间标签中使用 "Lissajous" 字样，保留
        block.append(
            _epoch_row(
                o["epoch"], "待设计Lissajous轨道设计的起始历元（年-月-日-时-分-秒格式的UTC）"
            )
        )
        block.append(
            _valrow(
                _fmt_g(o["duration"]),
                30,
                "待设计Lissajous轨道的维持时间（单位：年；取值大于0，不大于20）",
            )
        )

    block += _common_tail(
        o["perturbation"], o["dyb"], o["earth_degree"], o["moon_degree"], o["output_step"]
    )
    return block


def format_inputs_design(
    orbit_type: str,
    *,
    amplitude: float | None = None,
    phase: float | None = None,
    collinear_point: int | None = None,
    north_south: int | None = None,
    perilune_height: float | None = None,
    amplitude_in: float | None = None,
    amplitude_out: float | None = None,
    phase_in: float | None = None,
    phase_out: float | None = None,
    epoch: Any = None,
    duration: float | None = None,
    perturbation: dict[str, int] | None = None,
    dyb: Any = None,
    earth_degree: int = 10,
    moon_degree: int = 10,
    output_step: float = 60.0,
) -> list[str]:
    """生成功能码 1（任务轨道设计）的 245 行 inputs-dac.txt 内容。

    以 golden 模板为底，只重建 ``orbit_type`` 对应的设计块与 L1/L3 两行，
    其余各行逐字节保留。返回行列表（不含行尾符），用
    :func:`write_inputs_dac` 写盘。
    """
    sel = orbit_type.upper()
    if sel not in _DESIGN_BLOCKS:
        raise ValueError(f"orbit_type 必须为 DRO/NRHO/Halo/Lissajous/L4/L5，当前 {orbit_type!r}")
    blk_slice, type_num = _DESIGN_BLOCKS[sel]

    opts = {
        "amplitude": amplitude,
        "phase": phase,
        "collinear_point": collinear_point,
        "north_south": north_south,
        "perilune_height": perilune_height,
        "amplitude_in": amplitude_in,
        "amplitude_out": amplitude_out,
        "phase_in": phase_in,
        "phase_out": phase_out,
        "epoch": epoch,
        "duration": duration,
        "perturbation": perturbation,
        "dyb": DEFAULT_DYB if dyb is None else dyb,
        "earth_degree": earth_degree,
        "moon_degree": moon_degree,
        "output_step": output_step,
    }
    new_block = _build_design_block(sel, opts)
    if len(new_block) != blk_slice.stop - blk_slice.start:
        raise RuntimeError(
            f"设计块 {sel} 生成了 {len(new_block)} 行，应为 {blk_slice.stop - blk_slice.start} 行"
        )

    g = _golden_lines()
    g[0] = _valrow(1, 30, "软件的功能参数（=1，任务轨道设计；=2，任务轨道控制；=3，转移轨道设计）")
    g[2] = _valrow(
        type_num,
        29,
        "待设计任务轨道的类型参数（=1，DRO；=2，NRHO；=3，HALO；=4，Lissajous；=5，L4点；=6，L5点）",
    )
    g[blk_slice] = new_block
    return g


def format_inputs_propagate(
    *,
    epoch: Any,
    duration: float,
    initial_state: Any,
    perturbation: dict[str, int] | None = None,
    dyb: Any = None,
    earth_degree: int = 10,
    moon_degree: int = 10,
    output_step: float = 60.0,
) -> list[str]:
    """生成功能码 4（轨道预报）的 274 行 inputs-dac.txt 内容。

    保留 golden 前 244 行（L1 改写为功能码 4），丢弃原 L245 的 END 标记，
    追加 30 行预报段：21 行力模型 + 历元 + 时长 + 初始状态 6 分量 + END。

    行布局以说明文档为准（L245~265 力模型同 9~29 行、L266 起始历元、
    L267 预报时长、L268~273 初始状态），并经 DFH_DAC.exe 实跑验证。
    注意 MATLAB ``fmt_inputs_propagate.m`` 在预报段前多插一行 ``####``
    分隔符，导致其后全部下移一行（275 行布局）——exe 按定行号读取会
    错位报 list-directed I/O 错误，此处不移植该 bug。

    Args:
        epoch: 起始历元 UTC 六分量 ``[年, 月, 日, 时, 分, 秒]``
        duration: 预报时长（天）
        initial_state: 初始状态 6 分量（GCRS 位置 km + 速度 m/s）
    """
    state = list(initial_state)
    if len(state) != 6:
        raise ValueError(f"initial_state 必须为 6 个分量，当前 {len(state)} 个")

    block = _common_tail(
        perturbation, DEFAULT_DYB if dyb is None else dyb, earth_degree, moon_degree, output_step
    )
    block.append(_epoch_row(epoch, "轨道预报的起始历元（年-月-日-时-分-秒格式的UTC）"))
    block.append(_valrow(_fmt_g(duration), 30, "轨道预报的时间长度（单位：day）"))
    pos_labels = [
        "初始位置X分量（单位：km）",
        "初始位置Y分量（单位：km）",
        "初始位置Z分量（单位：km）",
    ]
    vel_labels = [
        "初始速度X分量（单位：m/s）",
        "初始速度Y分量（单位：m/s）",
        "初始速度Z分量（单位：m/s）",
    ]
    for i in range(3):
        block.append(_valrow(_fmt_g(state[i]), 30, pos_labels[i]))
    for i in range(3, 6):
        block.append(_valrow(_fmt_g(state[i]), 30, vel_labels[i - 3]))
    block.append("END OF THE INPUT FILE")
    if len(block) != 30:
        raise RuntimeError(f"预报段生成了 {len(block)} 行，应为 30 行")

    g = _golden_lines()
    g[0] = _valrow(4, 30, "软件的功能参数（=1，任务轨道设计；=2，任务轨道控制；=3，转移轨道设计）")
    return g[:244] + block


def format_inputs_control(
    *,
    control_mode: int = 1,
    is_nrho: int = 0,
    special_mode: int = 1,
    control_interval: float = 30.0,
    feedback_arc: float = 28.0,
    special_crossings: int = 3,
    num_controls: int = 120,
    num_monte_carlo: int = 5,
    output_step: float = 86400.0,
    position_accuracy: float = 1500.0,
    velocity_accuracy: float = 0.002,
    thrust_angle_err: float = 0.333,
    thrust_mean: float = 10.0,
    thrust_rel_err: float = 0.003,
    thrust_abs_err: float = 0.033,
    thrust_min: float = 0.1,
    thrust_max: float = 100.0,
    thrust_total: float = 1000.0,
    ecom_error_level: float = 0.10,
    perturbation: dict[str, int] | None = None,
    dyb: Any = None,
    earth_degree: int = 2,
    moon_degree: int = 2,
    real_perturbation: dict[str, int] | None = None,
    real_dyb: Any = None,
    real_earth_degree: int = 10,
    real_moon_degree: int = 10,
) -> list[str]:
    """生成功能码 2（任务轨道控制）的 245 行 inputs-dac.txt 内容。

    以 golden 模板为底，只重建控制段（L169-216，48 行）与 L1 功能码，
    其余各行逐字节保留。行布局与注释文本照 MATLAB
    ``fmt_inputs_control.m``：理论（控制量计算）与实际（真实动力学）两套
    力模型块 + 测控/推力误差参数 + 控制模式与时间参数。

    参数默认值与 MATLAB ``control_orbit.m`` 一致（ControlMode=1 目标点
    宽松、120 个控制周期、5 次蒙特卡洛等）；``real_*`` 未给时与实际模型
    相同的摄动开关/阶次 10×10 对齐。
    """
    theory_pert = _perturb_block(perturbation, prefix="用于轨道控制量计算的")
    real_pert = _perturb_block(real_perturbation, prefix="用于模拟实际力模型的")

    dyb_vals = list(DEFAULT_DYB if dyb is None else dyb)
    real_dyb_vals = list(DEFAULT_DYB if real_dyb is None else real_dyb)
    if len(dyb_vals) != 9 or len(real_dyb_vals) != 9:
        raise ValueError("dyb/real_dyb 必须为 9 个分量")

    # L188-196：DYB 系数（第 2..9 行带"（控制计算理论模型）"后缀；
    # 第 1 行独立标签、无后缀，照 MATLAB fmt_inputs_control.m 显式重建）
    dyb_block = _dyb_block(dyb_vals, suffix="（控制计算理论模型）")
    dyb_block[0] = (
        f"{float(dyb_vals[0]):.2f}".ljust(29)
        + "//等效面质比，也是DYB模型D方向上的常量分量DYB(1)（单位：m2/kg）"
    )

    block: list[str] = [
        "#" * 94,  # L169 分隔行
        *theory_pert,  # L170-178
        *real_pert,  # L179-187
        *dyb_block,  # L188-196
        _valrow(f"{ecom_error_level:.2f}", 30, "ECOM模型三个方向上各系数的误差量级（百分比/100）"),  # L197
        _valrow(earth_degree, 30, "地球非球形引力位阶次数（控制计算理论模型）"),  # L198
        _valrow(moon_degree, 30, "月球非球形引力位阶次数（控制计算理论模型）"),  # L199
        _valrow(real_earth_degree, 30, "地球非球形引力位阶次数（模拟的实际力模型）"),  # L200
        _valrow(real_moon_degree, 30, "月球非球形引力位阶次数（模拟的实际力模型）"),  # L201
        _valrow(f"{position_accuracy:.1f}", 30, "轨道测控的位置精度（1-sigma，单位：m）"),  # L202
        _valrow(f"{velocity_accuracy:g}", 30, "轨道测控的速度精度（1-sigma，单位：m/s）"),  # L203
        _valrow(f"{thrust_angle_err:g}", 30, "轨控发动机的角度精度（1-sigma，单位：deg）"),  # L204
        _valrow(f"{thrust_mean:.1f}", 30, "轨控发动机的中点值（单位：m/s）"),  # L205
        _valrow(  # L206
            f"{thrust_rel_err:g}",
            30,
            "轨控发动机喷气速度量大小的相对精度（1-sigma，百分比/100，当喷气量大于中点值时）",
        ),
        _valrow(  # L207
            f"{thrust_abs_err:g}",
            30,
            "轨控发动机喷气速度量大小的误差精度（1-sigma，单位：m/s，当喷气量小于中点值时）",
        ),
        _valrow(f"{thrust_min:g}", 30, "轨控发动机的最小开机速度增量（单位：m/s）"),  # L208
        _valrow(f"{thrust_max:.1f}", 30, "轨控发动机的最大开机速度增量（单位：m/s）"),  # L209
        _valrow(f"{thrust_total:.1f}", 30, "轨控发动机允许的所有速度增量（单位：m/s）"),  # L210
        _valrow(  # L211：控制模式 + NRHO 标志（双空格分隔）
            [control_mode, is_nrho],
            30,
            "轨控模式（=1，目标点宽松；=2，目标点紧致；=3，特征点；=4，目标点宽松+角动量管理；=5，目标点紧致+角动量管理；=6，特征点+角动量管理）目标轨道是否为NRHO轨道（=1，是；=0，否）",
        ),
        _valrow(  # L212
            special_mode,
            30,
            "特征点的控制模式参数（=1，Lissajous类型轨道；=2，halo类型轨道）",
        ),
        _valrow(f"{control_interval:.1f}", 30, "各控制方案的控制时间间隔（单位：day）"),  # L213
        _valrow(  # L214
            f"{feedback_arc:.1f}",
            30,
            "目标点紧致控制方案的反馈弧段长度（包括不考虑和考虑角动量管理两种情形，单位：day）",
        ),
        _valrow(  # L215
            special_crossings,
            30,
            "特征点控制方案中使用的与x-z平面的交点次数。",
        ),
        _valrow(  # L216：控制次数 + 蒙特卡洛次数 + 输出间隔（一行三值）
            f"{num_controls:d}      {num_monte_carlo:d}       {output_step:.1f}",
            30,
            "用户指定的控制次数（总控制时间长度=（控制次数-1）*控制时间间隔）、MonteCarlo仿真次数、输出星历的时间间隔（单位s）",
        ),
    ]
    if len(block) != 48:
        raise RuntimeError(f"控制段生成了 {len(block)} 行，应为 48 行")

    g = _golden_lines()
    g[0] = _valrow(2, 30, "软件的功能参数（=1，任务轨道设计；=2，任务轨道控制；=3，转移轨道设计）")
    g[168:216] = block
    return g


def write_inputs_dac(lines: list[str], path: str | Path) -> None:
    """将行列表逐行以 CRLF、UTF-8（无 BOM）写入磁盘。"""
    Path(path).write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
