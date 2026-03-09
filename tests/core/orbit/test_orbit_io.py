"""
Orbit 文件读写测试模块

测试 Orbit.save_to_file() 和 Orbit.load_from_file() 的基本行为，
重点覆盖 numpy 布尔值序列化问题以及保存后的属性恢复。
"""

import json

import numpy as np
import pytest

from e2m2e import Orbit


@pytest.fixture
def sample_orbit():
    """创建用于文件读写测试的轨道对象"""
    orbit = Orbit(
        states=[
            [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.8, 0.1, 0.0, -0.1, 0.1, 0.0],
            [0.6, 0.0, 0.0, 0.0, -0.2, 0.0],
            [0.8, -0.1, 0.0, 0.1, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],
        ],
        times=[0.0, 0.5, 1.0, 1.5, 2.0],
    )
    orbit.period = 2.0
    orbit.family_type = "lyapunov"
    orbit.is_periodic = np.bool_(True)
    orbit.periodicity_error = 1e-12
    orbit.metadata["description"] = "test orbit"
    orbit.metadata["tags"] = ["unit-test"]
    return orbit


class TestOrbitSaveToFile:
    """测试轨道保存功能"""

    def test_save_to_file_creates_parent_directory(self, tmp_path, sample_orbit):
        """保存时应自动创建父目录"""
        output_file = tmp_path / "nested" / "orbit.json"

        sample_orbit.save_to_file(str(output_file))

        assert output_file.exists()

    def test_save_to_file_serializes_numpy_bool(self, tmp_path, sample_orbit):
        """numpy.bool_ 类型应被转换为原生 bool 后写入 JSON"""
        output_file = tmp_path / "orbit.json"

        sample_orbit.save_to_file(str(output_file))

        with output_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        assert data["properties"]["is_periodic"] is True
        assert isinstance(data["properties"]["is_periodic"], bool)

    def test_save_to_file_writes_metadata_timestamp(self, tmp_path, sample_orbit):
        """保存结果应包含顶层时间戳和 metadata 中的保存时间戳"""
        output_file = tmp_path / "orbit.json"

        sample_orbit.save_to_file(str(output_file))

        with output_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        assert "timestamp" in data
        assert data["metadata"]["saved_timestamp"] == data["timestamp"]


class TestOrbitLoadFromFile:
    """测试轨道加载功能"""

    def test_load_from_file_restores_saved_properties(self, tmp_path, sample_orbit):
        """从文件加载后应恢复关键轨道属性"""
        output_file = tmp_path / "orbit.json"
        sample_orbit.save_to_file(str(output_file))

        loaded_orbit = Orbit.load_from_file(str(output_file))

        np.testing.assert_allclose(loaded_orbit.states, sample_orbit.states)
        np.testing.assert_allclose(loaded_orbit.times, sample_orbit.times)
        np.testing.assert_allclose(loaded_orbit.mean_state, sample_orbit.mean_state)
        assert loaded_orbit.period == sample_orbit.period
        assert loaded_orbit.family_type == sample_orbit.family_type
        assert loaded_orbit.is_periodic is True
        assert loaded_orbit.periodicity_error == sample_orbit.periodicity_error
        assert loaded_orbit.metadata["description"] == sample_orbit.metadata["description"]
        assert loaded_orbit.metadata["tags"] == sample_orbit.metadata["tags"]

    def test_load_from_file_restores_precomputed_extrema_and_amplitudes(
        self, tmp_path, sample_orbit
    ):
        """从文件加载后应保留保存时的极值和振幅信息"""
        output_file = tmp_path / "orbit.json"
        sample_orbit.save_to_file(str(output_file))

        loaded_orbit = Orbit.load_from_file(str(output_file))

        assert loaded_orbit.amplitudes == sample_orbit.amplitudes
        assert loaded_orbit.extrema == sample_orbit.extrema