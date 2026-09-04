"""基线数据集显式导入（ADR 0036 数据集；ADR 0047 决策 4 出包）。

基线分发包不再随 wheel 分发：数据集以 GitHub Release 资产（束文件
zip）提供，调用方下载解压后经 :func:`import_baseline` 展开为逐成员的
v2 轨道记录（束 id 即 ``family_id``，成员 ``record_id`` 确定性命名）
写入自己显式创建的库。束是 v1 传输格式，结构化读取，不经 v2 校验。
"""

from __future__ import annotations

import io
import json
from importlib.resources.abc import Traversable

import numpy as np

from .bundle import expand_bundle
from .record import CatalogFilter, validate_record_id
from .store import CatalogStore

__all__ = ["BASELINE_TAG", "import_baseline"]

#: 基线记录统一携带的标签
BASELINE_TAG = "baseline"


def import_baseline(store: CatalogStore, source_dir: Traversable) -> int:
    """把基线分发包目录展开为成员记录写入库，返回实际写入的记录数。

    ``source_dir`` 必填：Release 资产解压后的束目录（JSON + NPZ），
    不再有包内默认源（ADR 0047）。逐束对位 ``family_id``：库中已有
    该族且基线版本一致时跳过（用户对基线成员的删除得到尊重）；版本
    不一致时先删该族旧成员再全量展开（成员数变化不留陈旧尾巴）。
    JSON 声明了 ``arrays`` 而目录内缺少 NPZ 时抛错（数据集不完整，
    不导入残缺束）。
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"基线源目录不存在：{source_dir!s}；请从 GitHub Release 下载基线"
            "数据集资产并解压后传入（ADR 0047）"
        )
    imported = 0
    for item in source_dir.iterdir():
        if not item.name.endswith(".json"):
            continue
        bundle_meta = json.loads(item.read_text(encoding="utf-8"))
        family_id = validate_record_id(bundle_meta["record_id"])
        existing = store.query(CatalogFilter(family_id=family_id))
        if existing and _family_baseline_version(store, existing[0]["record_id"]) == (
            _baseline_version(bundle_meta)
        ):
            continue
        for summary in existing:
            store.delete(summary["record_id"])
        npz_src = source_dir / f"{family_id}.npz"
        # JSON 声明了段数组时 NPZ 必须存在：静默跳过会造出查得到、
        # 拿不到数据的残缺记录，且 baseline_version 对位后永不修复
        if bundle_meta.get("arrays") and not npz_src.is_file():
            raise FileNotFoundError(
                f"基线束 {family_id} 的 JSON 声明了 arrays 但源目录缺少"
                f" {family_id}.npz，数据集不完整（生成见 make catalog-baseline）"
            )
        # Traversable 无路径语义（源可能是 zip 内成员），经字节流加载
        with np.load(io.BytesIO(npz_src.read_bytes())) as npz:
            bundle_arrays = {key: npz[key] for key in npz.files}
        for meta, arrays in expand_bundle(bundle_meta, bundle_arrays):
            store.put(meta, arrays)
            imported += 1
    return imported


def _family_baseline_version(store: CatalogStore, record_id: str) -> str | None:
    return _baseline_version(store.get_meta(record_id))


def _baseline_version(meta: dict) -> str | None:
    version = meta.get("scalars", {}).get("baseline_version")
    return str(version) if version is not None else None
