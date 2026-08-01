"""转移模块导出卫生测试。"""


def test_module_does_not_export_nlp_result():
    """NLPOptimizationResult 不应再从 e2m2e.algorithm.transfer 公共导出。"""
    import e2m2e.algorithm.transfer

    assert not hasattr(e2m2e.algorithm.transfer, "NLPOptimizationResult")


def test_module_does_not_export_optimizer_adapters():
    """已删除的 optimizer adapter 类不应残留导出。"""
    import e2m2e.algorithm.transfer

    assert not hasattr(e2m2e.algorithm.transfer, "TransferOptimizer")
    assert not hasattr(e2m2e.algorithm.transfer, "SciPyTransferOptimizer")
    assert not hasattr(e2m2e.algorithm.transfer, "COPTTransferOptimizer")
