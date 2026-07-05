"""e2m2e 统一异常层次。

所有 e2m2e 抛出的异常都以 :class:`E2M2EError` 为共同基类，
便于调用方用 ``except E2M2EError`` 统一捕获库内部错误。

各子包仍可把自身异常多重继承自 Python 内置异常（如 ``ValueError``、
``RuntimeError``、``TypeError``），以保留原有的语义与既有 ``except``
子句的兼容性。
"""


class E2M2EError(Exception):
    """所有 e2m2e 异常的共同基类。"""

    pass
