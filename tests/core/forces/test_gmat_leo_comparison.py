"""GMAT LEO 对比脚本生成与基本行为测试。

验证脚本格式、力模型配置与报告解析。
"""

from pathlib import Path

import pytest


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gmat_leo_output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_generate_gmat_script_creates_files(output_dir: Path) -> None:
    """脚本生成器应创建 .script 文件和报告路径。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
    from generate_gmat_leo_script import generate_gmat_script

    script_path = generate_gmat_script(output_dir)

    assert script_path.exists()
    assert script_path.name == "leo_reference_gmat.script"
    content = script_path.read_text(encoding="utf-8")
    assert "Create Spacecraft LEOSat" in content
    assert "Create ForceModel LEOForceModel" in content
    assert "Create Propagator LEOPropagator" in content
    assert "Create ReportFile LEOReport" in content
    assert "CoordinateSystem = EarthICRF" in content
    assert "Propagate LEOPropagator(LEOSat)" in content


def test_generate_gmat_script_force_model_config(output_dir: Path) -> None:
    """生成的脚本应包含正确的力模型参数。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
    from generate_gmat_leo_script import generate_gmat_script

    script_path = generate_gmat_script(
        output_dir,
        forcemodel={
            "gravity_degree": 10,
            "gravity_order": 10,
            "drag_model": "Exponential",
            "srp": True,
        },
    )
    content = script_path.read_text(encoding="utf-8")

    assert "GravityField.Earth.Degree              = 10" in content
    assert "GravityField.Earth.Order               = 10" in content
    assert "Drag                                   = Exponential" in content
    assert "SRP                                    = On" in content
    assert "SRP.ShadowModel                        = None" in content


def test_compare_script_warns_when_gmat_report_missing(
    output_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """缺少 GMAT 报告时 compare 脚本应打印友好提示并退出。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
    from compare_with_gmat import main

    sys.argv = [
        "compare_with_gmat",
        "--output-dir",
        str(output_dir),
        "--gmat-report",
        str(output_dir / "missing.txt"),
    ]

    main()

    captured = capsys.readouterr()
    assert "GMAT report not found" in captured.out
    assert "gmat -s" in captured.out


@pytest.mark.spice
def test_propagate_e2m2e_runs_with_full_force_model(output_dir: Path, spice_kernel_path: str) -> None:
    """e2m2e 传播函数应能用完整力模型跑完 1 天。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
    from compare_with_gmat import _propagate_e2m2e

    data = _propagate_e2m2e(output_dir, include_drag=True, include_srp=True)

    assert data["time"].shape == (200,)
    assert data["states"].shape == (200, 6)
    assert data["time"][-1] - data["time"][0] > 86000.0


def test_parse_gmat_report_parses_space_delimited_rows(output_dir: Path) -> None:
    """GMAT 报告解析器应能解析带表头的空格分隔数据。"""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))
    from compare_with_gmat import _parse_gmat_report

    report_file = output_dir / "dummy_report.txt"
    report_file.write_text(
        "UTCGregorian EarthICRF.X EarthICRF.Y EarthICRF.Z EarthICRF.VX EarthICRF.VY EarthICRF.VZ\n"
        "2025-06-21T11:00:06 6771.0 0.0 0.0 0.0 4.768 6.016\n"
        "2025-06-21T11:01:06 6760.0 10.0 0.0 0.1 4.768 6.016\n",
        encoding="utf-8",
    )

    data = _parse_gmat_report(report_file)

    assert data["states"].shape == (2, 6)
    assert data["states"][0, 0] == pytest.approx(6771.0)
    assert data["states"][1, 3] == pytest.approx(0.1)
