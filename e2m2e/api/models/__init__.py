"""公开数据模型（Pydantic，全手写）子包。

输入/输出/错误模型按主题拆分在各子模块（#750），经本 ``__init__`` 逐名
re-export，公开导入路径 ``e2m2e.api.models`` 不变（ADR 0014 修订）。
输入/输出/错误模型精雕参数单位、默认值、取值域。Pydantic 只在
api/ 边界，算法层用 numpy/dataclass。每个 Facade 方法一个
Request/Response，外加统一错误模型 ``OrbitError``。
"""

from .catalog import (
    CatalogDeleteRequest,
    CatalogDeleteResponse,
    CatalogExportRequest,
    CatalogExportResponse,
    CatalogGetRequest,
    CatalogQueryRequest,
    CatalogQueryResponse,
    CatalogRecordResponse,
    CatalogRecordSummary,
    CatalogSweepPointOutcome,
    CatalogSweepRequest,
    CatalogSweepResponse,
    CatalogTagRequest,
    CatalogTagResponse,
    CatalogTerminologyResponse,
)
from .control import ControlOrbitRequest, ControlOrbitResponse
from .design import DesignOrbitRequest, DesignOrbitResponse
from .family import FamilyGenerationRequest, FamilyGenerationResponse
from .mga import (
    MgaChainCandidate,
    MgaFlybyInfo,
    MissionArchitectureSearchRequest,
    MissionArchitectureSearchResponse,
)
from .propagation import PropagationRequest, PropagationResponse
from .ranges import NumericRange
from .ranges import RangeSpec as RangeSpec
from .ranges import ValidRangesResponse as ValidRangesResponse
from .shared import OrbitError, ResultResponse, propagation_failure_details
from .spacetime import SpacetimeTransformRequest, SpacetimeTransformResponse
from .spatiography import (
    SpatiographyAtlasRequest,
    SpatiographyAtlasResponse,
    SpatiographyBoundariesRequest,
    SpatiographyBoundariesResponse,
    SpatiographyClassifyRequest,
    SpatiographyClassifyResponse,
    SpatiographyMapRequest,
    SpatiographyMapResponse,
    SpatiographyScalesRequest,
    SpatiographyScalesResponse,
)
from .transfer import (
    BplaneInfo as BplaneInfo,
)
from .transfer import (
    BplaneTarget as BplaneTarget,
)
from .transfer import (
    DepartureAsymptote as DepartureAsymptote,
)
from .transfer import (
    ManeuverEvent as ManeuverEvent,
)
from .transfer import (
    TransferCandidate as TransferCandidate,
)
from .transfer import (
    TransferDesignRequest,
    TransferDesignResponse,
)

__all__ = [
    "OrbitError",
    "propagation_failure_details",
    "NumericRange",
    "ResultResponse",
    "DesignOrbitRequest",
    "DesignOrbitResponse",
    "ControlOrbitRequest",
    "ControlOrbitResponse",
    "TransferDesignRequest",
    "TransferDesignResponse",
    "MissionArchitectureSearchRequest",
    "MissionArchitectureSearchResponse",
    "MgaChainCandidate",
    "MgaFlybyInfo",
    "PropagationRequest",
    "PropagationResponse",
    "SpacetimeTransformRequest",
    "SpacetimeTransformResponse",
    "FamilyGenerationRequest",
    "FamilyGenerationResponse",
    "CatalogQueryRequest",
    "CatalogRecordSummary",
    "CatalogQueryResponse",
    "CatalogGetRequest",
    "CatalogRecordResponse",
    "CatalogDeleteRequest",
    "CatalogDeleteResponse",
    "CatalogTagRequest",
    "CatalogTagResponse",
    "CatalogTerminologyResponse",
    "CatalogExportRequest",
    "CatalogExportResponse",
    "CatalogSweepRequest",
    "CatalogSweepPointOutcome",
    "CatalogSweepResponse",
    "SpatiographyScalesRequest",
    "SpatiographyScalesResponse",
    "SpatiographyClassifyRequest",
    "SpatiographyClassifyResponse",
    "SpatiographyBoundariesRequest",
    "SpatiographyBoundariesResponse",
    "SpatiographyAtlasRequest",
    "SpatiographyAtlasResponse",
    "SpatiographyMapRequest",
    "SpatiographyMapResponse",
]
