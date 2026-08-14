"""力模型子包异常。

所有异常均继承 :class:`e2m2e.exceptions.E2M2EError`，并同时保留
对原有 Python 内置异常的多重继承，使既有 ``except ValueError`` /
``except TypeError`` 等子句继续生效。
"""

from ...exceptions import E2M2EError


class CoordinateTransformError(E2M2EError, ValueError):
    """坐标转换失败时抛出。"""

    pass


class RelativisticCorrectionError(E2M2EError):
    """相对论修正力模型专用异常。"""

    pass


class NotSerializableError(E2M2EError, TypeError):
    """力模型无法序列化为配置（如含任意 Python callable）时抛出。"""

    pass
