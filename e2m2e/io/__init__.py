"""DFH 软件输入输出文件格式读写。

DFH（``DFH_DAC.exe``，闭源 Fortran）地月空间轨道设计软件的文本格式互操作
模块，用于黄金样本回归与 MATLAB/qiao 工作流互换。覆盖：

- ``EPHEMERIDES_*.TXT`` 星历读写；
- ``SK_STATISTIC.TXT`` 蒙特卡洛统计解析与写入；
- ``MANEUVERS.TXT`` 机动序列解析与写入；
- ``RESULTS_HMN/LGA/WSB.TXT`` 转移设计结果解析；
- ``inputs-dac.txt`` 输入文件生成（功能码 1 设计、2 控制、4 预报）；
- DFH 摄动开关 → e2m2e 力模型配置映射。
"""

from .ephemeris import EphemerisTable, parse_ephemeris, read_ephemeris, write_ephemeris
from .force_mapping import PLANET_BODIES, dfh_perturbation_to_force_config
from .inputs_dac import (
    DEFAULT_DYB,
    DEFAULT_PERTURBATION,
    format_inputs_control,
    format_inputs_design,
    format_inputs_propagate,
    write_inputs_dac,
)
from .maneuvers import ManeuverTable, parse_maneuvers, read_maneuvers, write_maneuvers
from .results import (
    HmnResult,
    MultiOrbitResult,
    OrbitSegment,
    parse_results_hmn,
    parse_results_multi,
    read_results_hmn,
    read_results_lga,
    read_results_wsb,
)
from .sk_statistic import SKStatistic, parse_sk_statistic, read_sk_statistic, write_sk_statistic

__all__ = [
    "EphemerisTable",
    "parse_ephemeris",
    "read_ephemeris",
    "write_ephemeris",
    "ManeuverTable",
    "parse_maneuvers",
    "read_maneuvers",
    "write_maneuvers",
    "SKStatistic",
    "parse_sk_statistic",
    "read_sk_statistic",
    "write_sk_statistic",
    "HmnResult",
    "MultiOrbitResult",
    "OrbitSegment",
    "parse_results_hmn",
    "parse_results_multi",
    "read_results_hmn",
    "read_results_lga",
    "read_results_wsb",
    "DEFAULT_DYB",
    "DEFAULT_PERTURBATION",
    "format_inputs_control",
    "format_inputs_design",
    "format_inputs_propagate",
    "write_inputs_dac",
    "PLANET_BODIES",
    "dfh_perturbation_to_force_config",
]
