"""探针：按 ``orb_id`` 读取 LLNL 百万条地月轨迹数据集的单条轨道。

数据集由 LLNL 用 SSAPy 生成，一百万条轨道共用一个历元
1980-01-01 00:00:00 TT（GCRF 惯性系），每条最长传播六年，力模型为
EGM2008 180 阶地球、GRGM1200A 月球、太阳点质量与地球/太阳辐射压。
发布分两层：描述表 ``descriptive_data.csv``（722 MB、一百万行、33 列）与
20000 个时间序列 HDF5（每个文件 50 条轨道，全量约 8.7 TB）。登记与用法
评估见 issue #807，本脚本是该 issue 的配套诊断工具。

GDO 对单连接限速很紧（实测 8 至 36 KB/s），全量下载不可行，单文件也要
十几分钟。脚本因此不落整文件，而是把 URL 包成按 HTTP Range 分块的只读
文件对象交给 h5py，只取需要的元数据块与时间序列切片。同一条轨道的
``orb_id`` 在描述表里一行、在 HDF5 里是一个组，脚本把两边都读出来并核对
同名量的口径，供第七类人工对照使用（见 ADR 0055）。

已知口径陷阱，脚本会在输出里复述。描述表的 ``v0`` 列逐行等于 ``r0`` 列，
初速度只能取 HDF5 的 ``v`` 序列第 0 帧。停算码 ``ejection`` 出现 0 至 4
五个取值而论文只列了四个标签，编码含义待确认。每条轨道在 HDF5 里占两个
组，``<orb_id>`` 组放时间序列，``<orb_id>.0`` 组只放 ``vmin``、``vmax``
与 ``r_vmin``、``r_vmax``，读的时候两个组都要取。

依赖：``numpy`` 随项目安装，``h5py`` 不在项目依赖里，属手工诊断的临时
叠加，需要时安装::

    uv pip install --python .venv/bin/python h5py

用法：:

    .venv/bin/python scripts/probe_llnl_cislunar_orbit.py --orb-id 10000
    .venv/bin/python scripts/probe_llnl_cislunar_orbit.py --orb-id 10000 --list-keys
    .venv/bin/python scripts/probe_llnl_cislunar_orbit.py --orb-id 10000 \\
        --series r v --frames 0:24 --save /tmp/orb_10000.npz
    .venv/bin/python scripts/probe_llnl_cislunar_orbit.py --orb-id 10000 --no-remote

描述表默认按环境变量 ``E2M2E_LLNL_CSV`` 或 ``~/Downloads/descriptive_data.csv``
查找，取不到就跳过本地核对，只读远端。``--series all`` 取该组 16 条逐小时
序列，也可以按名字取组内任意键（``covariances``、``nearby_orbits_r``、
``threebody_r`` 等），先用 ``--list-keys`` 看有哪些。组内的标量数据集不按帧
取用，点名它们会被跳过。``--frames`` 用 ``START:STOP[:STEP]`` 或单帧号限定
切片，最多三段且步长非零，单帧号不支持负数，负数区间写成两段形式。序列只
按需取用，不落整文件。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

DEFAULT_SERVER = "https://gdo-cislunar.llnl.gov/cislunar_data_six_year_1.0GEO_to_18.2GEO/"
ORBITS_PER_FILE = 50
ORBIT_COUNT = 1_000_000
EPOCH = "1980-01-01 00:00:00 TT"

# 每小时一帧的序列，顺序照组内约定排列，便于对照
TIME_SERIES = (
    "r",
    "v",
    "ra",
    "dec",
    "range",
    "pm_ra",
    "pm_dec",
    "pm_r",
    "M_v",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "argument_of_periapsis",
    "longitude_of_ascending_node",
    "true_anomaly",
    "true_longitude",
)

# 主组的标量元数据
SCALAR_KEYS = (
    "orb_id",
    "ejection",
    "lifetime",
    "period",
    "perigee",
    "apogee",
    "r_earth_min",
    "r_earth_max",
    "r_moon_min",
    "std_divergence",
    "median_divergence",
    "mean_divergence",
    "max_divergence",
    "threebody_std_divergence",
    "threebody_median_divergence",
    "threebody_mean_divergence",
    "threebody_max_divergence",
)

# <orb_id>.0 组里的最小/最大速度与其位置
COMPANION_KEYS = ("vmin", "vmax", "r_vmin", "r_vmax")

# 与描述表同名的量：(CSV 列、HDF5 键)
SCALAR_PAIRS = (
    ("ejection", "ejection"),
    ("lifetime", "lifetime"),
    ("period", "period"),
    ("perigee", "perigee"),
    ("apogee", "apogee"),
    ("vmin", "vmin"),
    ("vmax", "vmax"),
    ("std_divergence", "std_divergence"),
    ("median_divergence", "median_divergence"),
    ("mean_divergence", "mean_divergence"),
    ("max_divergence", "max_divergence"),
    ("threebody_std_divergence", "threebody_std_divergence"),
    ("threebody_median_divergence", "threebody_median_divergence"),
    ("threebody_mean_divergence", "threebody_mean_divergence"),
    ("threebody_max_divergence", "threebody_max_divergence"),
)

# 与描述表同名的向量：(CSV 列、HDF5 键)
VECTOR_PAIRS = (
    ("r0", "r_initial"),
    ("r_vmin", "r_vmin"),
    ("r_vmax", "r_vmax"),
)


class HttpRangeFile(io.RawIOBase):
    """把远端 URL 包成可 seek 的只读文件对象，按块缓存 HTTP Range 请求。

    供 h5py 的 fileobj 驱动使用，只取真正读到的块，避免下载整文件。
    """

    def __init__(
        self,
        url: str,
        *,
        block_size: int,
        timeout: float,
        retries: int,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._url = url
        self._block_size = block_size
        self._timeout = timeout
        self._retries = retries
        self._verbose = verbose
        self._pos = 0
        self._blocks: dict[int, bytes] = {}
        self._size = self._probe_size()
        self.fetched = 0

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"seek whence 不支持：{whence}")
        self._pos = max(0, pos)
        return self._pos

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        if not len(view) or self._pos >= self._size:
            return 0
        end = min(self._pos + len(view), self._size)
        data = self._read_range(self._pos, end)
        view[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        out = bytearray()
        while len(out) < size and self._pos < self._size:
            chunk = self._read_range(self._pos, min(self._pos + size - len(out), self._size))
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
        return bytes(out)

    def _read_range(self, start: int, end: int) -> bytes:
        """取 [start, end) 的字节，按块拼装。"""
        out = bytearray()
        index = start // self._block_size
        cursor = start
        while cursor < end:
            block_start = index * self._block_size
            block = self._block(index)
            offset = cursor - block_start
            if offset >= len(block):
                break
            take = min(end - cursor, len(block) - offset)
            out += block[offset : offset + take]
            cursor += take
            index += 1
        return bytes(out)

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached
        low = index * self._block_size
        high = min(low + self._block_size - 1, self._size - 1)
        data = self._request(low, high)
        if len(self._blocks) > 4096:
            self._blocks.clear()
        self._blocks[index] = data
        self.fetched += len(data)
        return data

    def _request(self, low: int, high: int) -> bytes:
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                time.sleep(min(2.0**attempt, 30.0))
            if self._verbose:
                print(f"  取字节 {low}-{high}", file=sys.stderr, flush=True)
            request = urllib.request.Request(self._url, headers={"Range": f"bytes={low}-{high}"})
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return response.read()
            except Exception as exc:  # noqa: BLE001 - 远端限速与断连都按重试处理
                last = exc
        raise RuntimeError(f"区间请求失败（{low}-{high}，已重试 {self._retries} 次）：{last}")

    def _probe_size(self) -> int:
        request = urllib.request.Request(self._url, method="HEAD")
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            length = response.headers.get("Content-Length")
        if not length:
            raise RuntimeError(f"HEAD 未返回 Content-Length：{self._url}")
        return int(length)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 orb_id 读取 LLNL 百万条地月轨迹数据集（描述表 + 远端 HDF5 按需读取）",
    )
    parser.add_argument("--orb-id", type=int, required=True, help="轨道编号，0 至 999999")
    parser.add_argument("--csv", type=pathlib.Path, default=None, help="descriptive_data.csv 路径")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="GDO 数据集目录 URL")
    parser.add_argument(
        "--series",
        nargs="*",
        default=[],
        metavar="NAME",
        help="要取的时间序列名，取 all 表示该组全部序列",
    )
    parser.add_argument(
        "--frames",
        default=None,
        metavar="START:STOP[:STEP]",
        help="序列切片，单帧号或 START:STOP[:STEP]，默认取全长",
    )
    parser.add_argument("--list-keys", action="store_true", help="只列出组内键与形状")
    parser.add_argument("--no-remote", action="store_true", help="只读本地描述表，不访问远端")
    parser.add_argument(
        "--save",
        type=pathlib.Path,
        default=None,
        help="把读到的数据写到 npz，路径缺 .npz 后缀时自动补上",
    )
    parser.add_argument("--block-size", type=int, default=1 << 16, help="远端读取的块大小（字节）")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时（秒）")
    parser.add_argument("--retries", type=int, default=3, help="单次请求重试次数")
    parser.add_argument("--verbose", action="store_true", help="打印每次区间请求")
    return parser.parse_args(argv)


def resolve_csv_path(explicit: pathlib.Path | None) -> pathlib.Path:
    """返回描述表路径，取不到时返回不存在的候选路径，由调用方决定是否跳过。"""
    if explicit is not None:
        return explicit
    from_env = os.environ.get("E2M2E_LLNL_CSV")
    if from_env:
        return pathlib.Path(from_env).expanduser()
    return pathlib.Path.home() / "Downloads" / "descriptive_data.csv"


def seek_near_row(handle: Any, guess: int) -> None:
    """把句柄定位到 ``guess`` 字节所在行的行首，找不到换行就回到第一条数据行。"""
    if guess <= 0:
        handle.seek(0)
        handle.readline()
        return
    start = max(0, guess - 4096)
    handle.seek(start)
    tail = handle.read(guess - start)
    newline = tail.rfind(b"\n")
    if newline < 0:
        handle.seek(0)
        handle.readline()
        return
    handle.seek(start + newline + 1)


def csv_row_for(path: pathlib.Path, orb_id: int) -> dict[str, str] | None:
    """在描述表里取 ``orb_id`` 那一行，用行号与字节位置的近似线性关系先定位。"""
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = next(csv.reader([handle.readline().decode("utf-8")]))
        guess = size * orb_id // ORBIT_COUNT
        for use_guess in (True, False):
            seek_near_row(handle, guess if use_guess else 0)
            for line in handle:
                if not line.strip():
                    continue
                row = next(csv.reader([line.decode("utf-8")]))
                if len(row) != len(header):
                    continue
                try:
                    current = int(row[0])
                except ValueError:
                    continue
                if current == orb_id:
                    return dict(zip(header, row, strict=True))
                if current > orb_id:
                    break
    return None


def parse_vector(text: str) -> np.ndarray:
    """解析描述表里的 ``[a b c]`` 向量。"""
    return np.array([float(item) for item in text.strip().strip("[]").split()])


def parse_frames(text: str | None) -> slice:
    """解析 ``START:STOP[:STEP]`` 或单帧号，段数、整数与步长都校验。"""
    if not text:
        return slice(None)
    parts = text.split(":")
    if len(parts) > 3:
        raise SystemExit(f"--frames 最多三段 START:STOP[:STEP]，收到 {text}")
    numbers: list[int | None] = []
    for part in parts:
        if not part:
            numbers.append(None)
            continue
        try:
            numbers.append(int(part))
        except ValueError:
            raise SystemExit(f"--frames 的每段必须是整数或留空，收到 {text}") from None
    if len(numbers) == 1:
        frame = numbers[0]
        if frame is None:  # pragma: no cover - 单段留空时 split 给出的是单个空段
            raise SystemExit(f"--frames 的单帧号不能留空，收到 {text}")
        if frame < 0:
            raise SystemExit(f"--frames 的单帧号不支持负数，负数区间写成两段形式，收到 {text}")
        return slice(frame, frame + 1)
    start = numbers[0]
    stop = numbers[1]
    step = numbers[2] if len(numbers) > 2 else None
    if step == 0:
        raise SystemExit(f"--frames 的步长不能为 0，收到 {text}")
    return slice(start, stop, step)


def h5_url(server: str, orb_id: int) -> str:
    start = (orb_id // ORBITS_PER_FILE) * ORBITS_PER_FILE
    name = f"orb_id_{start}_to_{start + ORBITS_PER_FILE - 1}.h5"
    return urllib.parse.urljoin(server if server.endswith("/") else server + "/", name)


@contextlib.contextmanager
def open_remote_h5(url: str, args: argparse.Namespace) -> Iterator[tuple[Any, HttpRangeFile]]:
    """打开远端 HDF5，关闭时一并释放分块读取器。"""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise SystemExit(
            "缺少 h5py，远端读取需要它。安装：\n  uv pip install --python .venv/bin/python h5py"
        ) from exc
    reader = HttpRangeFile(
        url,
        block_size=args.block_size,
        timeout=args.timeout,
        retries=args.retries,
        verbose=args.verbose,
    )
    with h5py.File(reader, "r") as handle:
        yield handle, reader


def read_scalars(group: Any, keys: Iterable[str]) -> dict[str, float]:
    """读取组内标量，缺失的键跳过。"""
    out: dict[str, float] = {}
    for key in keys:
        obj = group.get(key)
        if obj is not None and obj.shape == ():
            out[key] = float(obj[()])
    return out


def report_csv(row: dict[str, str] | None, path: pathlib.Path, orb_id: int) -> None:
    print(f"描述表：{path}")
    if row is None:
        print(f"  没有找到 orb_id {orb_id} 的行，跳过本地核对")
        return
    ejection = row["ejection"]
    print(
        "  停算码 {ejection}，寿命 {life:.4f} 年，周期 {period:.4f} 天，"
        "半长轴 {a:.4f} GEO，偏心率 {ecc:.4f}，倾角 {inc:.2f} 度".format(
            ejection=ejection,
            life=float(row["lifetime"]),
            period=float(row["period"]),
            a=float(row["a"]) / 4.216e7,
            ecc=float(row["e"]),
            inc=np.degrees(float(row["i"])),
        )
    )
    print(f"  初始位置 r0（m）：{parse_vector(row['r0']).tolist()}")
    print("  提示：v0 列与 r0 列相同，是本数据集发布表的已知缺陷，初速度取 HDF5 的 v[0]。")
    if ejection not in {"0", "2", "3", "4"}:
        print("  提示：该停算码不在本次抽样见过的 0/2/3/4 之列，编码含义待确认。")


def report_h5_metadata(group: Any, companion: Any) -> dict[str, Any]:
    scalars = read_scalars(group, SCALAR_KEYS)
    print("HDF5 元数据：")
    for key in SCALAR_KEYS:
        if key in scalars:
            print(f"  {key} = {scalars[key]:.10g}")
    for key in COMPANION_KEYS:
        if companion is not None and key in companion:
            value = np.asarray(companion[key][()])
            if value.shape == ():
                print(f"  {key} = {float(value):.10g}")
                scalars[key] = float(value)
            else:
                print(f"  {key} = {value.tolist()}")
    if "r_initial" in group:
        r_initial = np.asarray(group["r_initial"][()])
        print(f"  r_initial（m）= {r_initial.tolist()}")
        print(f"    地心距 {np.linalg.norm(r_initial) / 1e3:.1f} km")
        scalars["r_initial"] = r_initial
    return scalars


def format_size(count: float) -> str:
    if count >= 1e6:
        return f"{count / 1e6:.2f} MB"
    if count >= 1e3:
        return f"{count / 1e3:.1f} kB"
    return f"{count:.0f} B"


def report_pulled_series(
    series: dict[str, np.ndarray],
    full_shapes: dict[str, tuple[int, ...]],
    lifetime_seconds: float | None,
    frames: slice,
) -> None:
    if not series:
        return
    print("时间序列：")
    for name, array in series.items():
        shape = full_shapes.get(name, array.shape)
        cadence = ""
        if len(shape) >= 1 and shape[0] > 1 and lifetime_seconds:
            cadence = f"，平均间距 {lifetime_seconds / (shape[0] - 1):.1f} s"
        print(f"  {name} 全长 {shape}，本次 {array.shape}{cadence}")
    first = next(iter(series))
    values = np.asarray(series[first])
    if values.size == 0:
        print(f"  首末帧：{first} 本次切片为空，没有可展示的帧")
    else:
        print(f"  首帧 {first} = {values[0].tolist()}")
        print(f"  末帧 {first} = {values[-1].tolist()}")
    print(f"  切片 {frames}")


def compare(csv_row: dict[str, str] | None, scalars: dict[str, Any]) -> None:
    if csv_row is None:
        return
    print("口径核对（描述表对 HDF5）：")
    for column, key in SCALAR_PAIRS:
        if key not in scalars:
            continue
        left = float(csv_row[column])
        right = float(scalars[key])
        rel = abs(left - right) / abs(right) if right else float("nan")
        verdict = "一致" if left == right or rel < 1e-12 else f"不一致，相对差 {rel:.3e}"
        print(f"  {column}: 描述表 {left:.10g}，HDF5 {right:.10g}，{verdict}")
    for column, key in VECTOR_PAIRS:
        if key not in scalars:
            continue
        left = parse_vector(csv_row[column])
        right = np.asarray(scalars[key])
        scale = max(float(np.max(np.abs(right))), 1.0)
        rel = float(np.max(np.abs(left - right))) / scale
        if rel < 1e-7:
            verdict = "一致（差值为描述表有效数字截断）"
        else:
            verdict = f"最大分量差 {rel * scale:.6g} m，相对 {rel:.3e}"
        print(f"  {column}: {verdict}")


def report_list_keys(group: Any, companion: Any) -> None:
    print("组内键：")
    for name in sorted(group.keys()):
        obj = group[name]
        kind = "组" if not hasattr(obj, "shape") else f"{obj.shape} {obj.dtype}"
        print(f"  {name:28s} {kind}")
    if companion is not None:
        print(f"伴随组（{companion.name}）：")
        for name in sorted(companion.keys()):
            obj = companion[name]
            print(f"  {name:28s} {obj.shape} {obj.dtype}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.orb_id < ORBIT_COUNT:
        raise SystemExit(f"--orb-id 应在 0 至 {ORBIT_COUNT - 1} 之间")

    print(f"数据集：LLNL 百万条地月轨迹，历元 {EPOCH}，GCRF，单条最长六年（issue #807）")
    file_name = pathlib.Path(h5_url(args.server, args.orb_id)).name
    print(f"轨道：orb_id {args.orb_id}，所在文件 {file_name}")
    print()

    csv_path = resolve_csv_path(args.csv)
    csv_row: dict[str, str] | None = None
    if csv_path.exists():
        csv_row = csv_row_for(csv_path, args.orb_id)
        report_csv(csv_row, csv_path, args.orb_id)
    else:
        print(f"描述表：未找到 {csv_path}（--csv 可指定路径，跳过本地核对）")
    if args.no_remote:
        return 0

    frames = parse_frames(args.frames)
    url = h5_url(args.server, args.orb_id)
    series_names = list(TIME_SERIES) if args.series == ["all"] else list(args.series)

    with open_remote_h5(url, args) as (handle, reader):
        key = str(args.orb_id)
        if key not in handle:
            raise SystemExit(f"文件里没有组 {key}，可用组示例：{sorted(handle.keys())[:6]}")
        group = handle[key]
        companion = handle.get(f"{key}.0")
        if args.list_keys:
            report_list_keys(group, companion)
            return 0

        print()
        scalars = report_h5_metadata(group, companion)
        if companion is not None:
            for key_name in ("r_vmin", "r_vmax"):
                if key_name in companion:
                    scalars[key_name] = np.asarray(companion[key_name][()])
        lifetime_seconds = scalars.get("lifetime", 0.0) * 365.25 * 86400.0

        pulled: dict[str, np.ndarray] = {}
        full_shapes: dict[str, tuple[int, ...]] = {}
        for name in series_names:
            if name not in group:
                print(f"  序列 {name} 不在组内，跳过")
                continue
            if group[name].shape == ():
                print(f"  {name} 是标量数据集，不能按帧取用，跳过")
                continue
            full_shapes[name] = tuple(group[name].shape)
            pulled[name] = np.asarray(group[name][frames])
        print()
        report_pulled_series(pulled, full_shapes, lifetime_seconds or None, frames)
        print()
        compare(csv_row, scalars)
        print()
        print(f"远端取用字节：{format_size(reader.fetched)}（块大小 {args.block_size}）")

        if args.save is not None:
            save = args.save
            if not str(save).endswith(".npz"):
                save = pathlib.Path(f"{save}.npz")
            scalar_meta = {k: float(v) for k, v in scalars.items() if np.ndim(v) == 0}
            meta = {
                "orb_id": args.orb_id,
                "url": url,
                "epoch": EPOCH,
                "frames": str(frames),
                "scalars": scalar_meta,
            }
            np.savez(save, meta=json.dumps(meta, ensure_ascii=False), **pulled)
            print(f"已写出 {save}（{format_size(save.stat().st_size)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
