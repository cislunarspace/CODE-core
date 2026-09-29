"""决策变量块登记器（纯布局，不含成本/权重语义）。"""

from __future__ import annotations

__all__ = ["VariableLayout"]


class VariableLayout:
    """按登记顺序分配决策变量块偏移的登记器。

    框架与具体问题（SF 的冲量/v∞/TOF 块、MGA 精化的历元/TOF 块）共享的
    变量布局原语：``register`` 顺序切分 ``[0, n_var)``，返回各块的切片；
    成本函数、权重等转录语义由调用方自持，不在本类。

    用法::

        layout = VariableLayout()
        imp = layout.register("impulses", 3 * n)   # slice(0, 3n)
        tof = layout.register("tof", n_legs)        # slice(3n, 3n + n_legs)
        layout.n_var                                # 3n + n_legs
    """

    def __init__(self) -> None:
        self._slices: dict[str, slice] = {}
        self._n_var: int = 0

    def register(self, name: str, size: int) -> slice:
        """登记一个变量块，返回其 ``slice(start, start + size)``。

        Raises:
            ValueError: 名字重复或 ``size`` 为负整数。
        """
        if name in self._slices:
            raise ValueError(f"变量块 {name!r} 已登记，不能重复登记")
        n = int(size)
        if n != size or n < 0:
            raise ValueError(f"变量块大小必须为非负整数，得到 {size!r}")
        block = slice(self._n_var, self._n_var + n)
        self._slices[name] = block
        self._n_var += n
        return block

    @property
    def n_var(self) -> int:
        """已登记变量总数。"""
        return self._n_var

    def __getitem__(self, name: str) -> slice:
        """取已登记变量块的切片；未登记名字抛 ``KeyError``。"""
        return self._slices[name]
