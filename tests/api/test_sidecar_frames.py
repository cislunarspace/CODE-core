"""sidecar 二进制帧编解码测试。

帧契约的字节级黄金断言在本文件：magic/dtype/ndim/shape/数据逐字段
逐项对照，实现漂移会在这里被抓住。
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.interface]

from e2m2e.api.sidecar.frames import (  # noqa: E402
    MAGIC,
    FrameError,
    decode_frame,
    encode_frame,
)


def test_golden_bytes_f32_1d():
    """黄金字节串：f32 一维 [1.0, -2.0] 逐字节对照 ADR 0035 §3。"""
    buf = encode_frame(np.array([1.0, -2.0], dtype=np.float32), "f32")
    expected = (
        bytes([0x45, 0x32, 0x4D, 0x32])  # u32 magic 0x324D3245，小端
        + bytes([0x00])  # u8 dtype 0 = f32
        + bytes([0x01])  # u8 ndim = 1
        + bytes([0x02, 0x00, 0x00, 0x00])  # u32 shape[0] = 2，小端
        + np.array([1.0, -2.0], dtype="<f4").tobytes()
    )
    assert buf == expected


def test_golden_bytes_f64_2d():
    """黄金字节串：f64 (1, 2) 数组，验证多维 shape 顺序（C 连续）。"""
    arr = np.array([[6.936e-3, 0.0]], dtype=np.float64)
    buf = encode_frame(arr, "f64")
    assert buf[:4] == bytes([0x45, 0x32, 0x4D, 0x32])
    assert buf[4] == 0x01  # dtype 1 = f64
    assert buf[5] == 0x02  # ndim = 2
    assert buf[6:14] == bytes([1, 0, 0, 0, 2, 0, 0, 0])  # shape (1, 2)
    assert buf[14:] == arr.astype("<f8").tobytes()
    assert len(buf) == 6 + 8 + 16


@pytest.mark.parametrize("dtype_code,np_dtype", [(0, np.float32), (1, np.float64)])
@pytest.mark.parametrize("shape", [(3,), (2, 6), (2, 3, 4)])
def test_roundtrip(dtype_code, np_dtype, shape):
    arr = np.arange(int(np.prod(shape)), dtype=np_dtype).reshape(shape)
    dtype = "f32" if dtype_code == 0 else "f64"
    buf = encode_frame(arr, dtype)
    out, out_dtype, consumed = decode_frame(buf)
    assert out_dtype == dtype
    assert out.dtype == np_dtype
    assert out.shape == shape
    np.testing.assert_array_equal(out, arr)
    assert consumed == len(buf)


def test_decode_recovers_stream_position():
    """帧尾由 shape 决定，帧后尾随字节（下一帧或 JSON 行）合法；解码返回
    消费字节数，JSON 行流可据此恢复（ADR 0035 §3）。"""
    frame = encode_frame(np.ones((2, 6)), "f64")
    rest = b'\n{"status": "ok"}'
    _, _, consumed = decode_frame(frame + rest)
    assert consumed == len(frame)


def test_decode_bad_magic():
    buf = bytearray(encode_frame(np.zeros(3), "f32"))
    buf[0] ^= 0xFF
    with pytest.raises(FrameError, match="magic"):
        decode_frame(bytes(buf))


def test_decode_bad_dtype_code():
    buf = bytearray(encode_frame(np.zeros(3), "f32"))
    buf[4] = 7
    with pytest.raises(FrameError, match="dtype"):
        decode_frame(bytes(buf))


def test_decode_truncated():
    """帧长与 shape 声明不符（数据段截断/头不完整）→ FrameError。"""
    good = encode_frame(np.zeros((2, 6)), "f64")
    with pytest.raises(FrameError):
        decode_frame(good[:-1])  # 数据段截断
    with pytest.raises(FrameError):
        decode_frame(good[:5])  # 头都不完整


def test_encode_rejects_zero_dim_and_unknown_dtype():
    scalar = np.float64(1.0)
    with pytest.raises(FrameError, match="ndim"):
        encode_frame(scalar, "f64")  # 标量 ndim=0，契约要求 ≥1
    with pytest.raises(FrameError, match="dtype"):
        encode_frame(np.zeros(2), "f16")


def test_encode_normalizes_dtype_and_order():
    """大端/非连续输入被规范化为小端 C 连续，字节输出不受输入布局影响。"""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f8")
    assert encode_frame(arr, "f64") == encode_frame(np.ascontiguousarray(arr.astype("<f8")), "f64")
    assert MAGIC == 0x324D3245
