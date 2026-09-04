"""CR3BP 基线数据集首用导入（ADR 0036 决策 5；ADR 0045 决策 8）。

包内 ``e2m2e/data/catalog_baseline/`` 是一批 v1 族束（分发包，见
``bundle.py``），``tags=["baseline"]``、``scalars.baseline_version`` 标
基线版本、束 id 确定性命名如 ``baseline-halo-l2``。首用导入把每束
展开为逐成员的 v2 轨道记录（束 id 即 ``family_id``，成员
``record_id`` 确定性命名），经存储引擎逐条写入；之后一切走既有
``catalog_query``（整族按 ``family_id`` 过滤）。
"""

from __future__ import annotations

import io
import json
from importlib.resources.abc import Traversable

import numpy as np

from .bundle import expand_bundle
from .record import CatalogFilter, validate_record_id
from .store import CatalogStore

__all__ = ["BASELINE_TAG", "baseline_source_dir", "import_baseline"]

#: 基线记录统一携带的标签
BASELINE_TAG = "baseline"


def baseline_source_dir() -> Traversable:
    """包内基线数据目录（importlib.resources 定位，不猜安装路径）。"""
    from importlib.resources import files

    return files("e2m2e.data").joinpath("catalog_baseline")


def import_baseline(store: CatalogStore, source_dir: Traversable | None = None) -> int:
    """把包内基线分发包展开为成员记录写入用户库，返回实际写入的记录数。

    逐束对位 ``family_id``：库中已有该族且基线版本一致时跳过（用户对
    基线成员的删除得到尊重）；版本不一致时先删该族旧成员再全量展开
    （成员数变化不留陈旧尾巴）。束是 v1 传输格式，结构化读取，不经
    v2 校验。JSON 声明了 ``arrays`` 而包内缺少 NPZ 时抛错（数据集不
    完整，不导入残缺束）。包内无基线数据（如开发环境未生成）时静默
    跳过。``source_dir`` 供测试注入合成基线源。
    """
    src = source_dir if source_dir is not None else baseline_source_dir()
    if not src.is_dir():
        return 0
    imported = 0
    for item in src.iterdir():
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
        npz_src = src / f"{family_id}.npz"
        # JSON 声明了段数组时 NPZ 必须存在：静默跳过会造出查得到、
        # 拿不到数据的残缺记录，且 baseline_version 对位后永不修复
        if bundle_meta.get("arrays") and not npz_src.is_file():
            raise FileNotFoundError(
                f"基线束 {family_id} 的 JSON 声明了 arrays 但包内缺少"
                f" {family_id}.npz，数据集不完整（生成见 make catalog-baseline）"
            )
        # Traversable 无路径语义（包资源可能是 zip 内成员），经字节流加载
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
