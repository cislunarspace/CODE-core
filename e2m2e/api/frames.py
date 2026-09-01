"""二进制帧编解码（ADR 0035 §3）。

帧格式是 tod ↔ e2m2e 的跨仓库持久契约，逐字段定义（多字节整数一律小端）：

    偏移  长度     类型      含义
    0     4        u32       magic = 0x324D3245（ASCII "E2M2" 小端；字节 45 32 4D 32）
    4     1        u8        dtype：0 = float32，1 = float64
    5     1        u8        ndim：维度数，≥ 1
    6     4·ndim   u32[]     shape：各维元素数（不是字节数）
    6+4·ndim  —   —         原始数组字节：C 连续、小端，长度 = prod(shape) × 元素宽度

没有 version 字段：magic 兼职版本锚点，不兼容改动时换 magic（须走新 ADR）。
本模块是帧契约的唯一实现点，不得引入 ADR 之外的字段。位于 api/ 包根而
非 sidecar 子包：执行核心的画布帧抽取（execution.py）与 sidecar 协议共
用同一编解码（#601）。
"""

from __future__ import annotations

import struct

import numpy as np

__all__ = ["MAGIC", "FrameError", "encode_frame", "decode_frame"]

# ASCII "E2M2" 的小端 u32。hexdump 可直接认出（ADR 0035 决策）。
MAGIC = 0x324D3245

# dtype 码表（u8）：0 = f32，1 = f64。请求方声明，编码与解码共用。
_DTYPE_CODES = {"f32": 0, "f64": 1}
_CODE_DTYPES = {code: name for name, code in _DTYPE_CODES.items()}
_NUMPY_DTYPES = {"f32": "<f4", "f64": "<f8"}

_HEADER = struct.Struct("<IBB")  # magic, dtype, ndim
_U32 = struct.Struct("<I")


class FrameError(ValueError):
    """帧编解码失败（坏 magic、坏 dtype、长度与声明不符等）。"""


def encode_frame(array: np.ndarray, dtype: str) -> bytes:
    """把数组编码为一帧。

    Args:
        array: 任意布局的数组（内部规范化为 C 连续、小端）。
        dtype: ``"f32"`` 或 ``"f64"``，与数组当前 dtype 独立（按声明转换）。
    """
    if dtype not in _DTYPE_CODES:
        raise FrameError(f"未知 dtype {dtype!r}（支持：f32/f64）")
    arr = np.asarray(array)
    if arr.ndim < 1:
        raise FrameError("ndim 必须 ≥ 1（标量不能单独成帧）")
    arr = np.ascontiguousarray(arr, dtype=_NUMPY_DTYPES[dtype])
    header = _HEADER.pack(MAGIC, _DTYPE_CODES[dtype], arr.ndim)
    shape = b"".join(_U32.pack(dim) for dim in arr.shape)
    return header + shape + arr.tobytes()


def decode_frame(buf: bytes | memoryview) -> tuple[np.ndarray, str, int]:
    """从 ``buf`` 开头解码一帧。

    Returns:
        (数组, dtype 名, 消费字节数)。消费字节数让调用方在帧流后恢复
        JSON 行流（JSON 行的 ``binary_frames`` 声明了帧数，帧后无分隔符）。
    """
    view = memoryview(buf)
    if len(view) < _HEADER.size:
        raise FrameError(f"帧头不完整：{len(view)} 字节 < {_HEADER.size}")
    magic, dtype_code, ndim = _HEADER.unpack_from(view)
    if magic != MAGIC:
        raise FrameError(f"帧 magic 不符：0x{magic:08X}（期望 0x{MAGIC:08X}）")
    dtype = _CODE_DTYPES.get(dtype_code)
    if dtype is None:
        raise FrameError(f"未知 dtype 码 {dtype_code}（支持：0=f32, 1=f64）")
    if ndim < 1:
        raise FrameError(f"ndim 必须 ≥ 1，帧内为 {ndim}")
    offset = _HEADER.size
    shape_end = offset + 4 * ndim
    if len(view) < shape_end:
        raise FrameError("shape 段不完整")
    shape = tuple(_U32.unpack_from(view, offset + 4 * i)[0] for i in range(ndim))
    n_elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
    item_size = 4 if dtype == "f32" else 8
    data_len = n_elements * item_size
    data_end = shape_end + data_len
    if len(view) < data_end:
        raise FrameError(f"数据段不完整：{len(view) - shape_end} 字节 < 声明 {data_len}")
    arr = np.frombuffer(view[shape_end:data_end], dtype=_NUMPY_DTYPES[dtype]).reshape(shape)
    return arr, dtype, data_end


def read_raw_frame(stream) -> bytes:
    """从二进制流读取一帧完整字节（含帧头），原样返回。

    供 worker stdout 泵转发帧字节用（#607）：父进程不需要解码数组，
    只需按契约切出完整帧。magic 与长度校验失败抛 :class:`FrameError`。
    """
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise FrameError(f"帧头不完整：{len(header)} 字节 < {_HEADER.size}")
    magic, dtype_code, ndim = _HEADER.unpack(header)
    if magic != MAGIC:
        raise FrameError(f"帧 magic 不符：0x{magic:08X}（期望 0x{MAGIC:08X}）")
    if ndim < 1:
        raise FrameError(f"ndim 必须 ≥ 1，帧内为 {ndim}")
    shape_bytes = stream.read(4 * ndim)
    if len(shape_bytes) < 4 * ndim:
        raise FrameError("shape 段不完整")
    shape = _U32.unpack_from(shape_bytes)
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    data_len = n_elements * (4 if dtype_code == 0 else 8)
    payload = stream.read(data_len)
    if len(payload) < data_len:
        raise FrameError(f"数据段不完整：{len(payload)} 字节 < 声明 {data_len}")
    return header + shape_bytes + payload
