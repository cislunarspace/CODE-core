"""算法层组件目录。"""

from .components import Component

ALGORITHM_COMPONENTS = [
    Component(
        name="CR3BP_System",
        module_path="e2m2e.algorithm.dynamics.cr3bp_system",
        layer="algorithm",
        description="CR3BP system definition (mass parameter, libration points, Jacobi constant)",
    ),
    Component(
        name="Dynamics",
        module_path="e2m2e.algorithm.dynamics.dynamics",
        layer="algorithm",
        description="Generic dynamics base class",
    ),
    Component(
        name="CR3BP_Dynamics",
        module_path="e2m2e.algorithm.dynamics.dynamics",
        dependencies=["Dynamics", "CR3BP_System"],
        layer="algorithm",
        description="CR3BP equations of motion and STM computation",
    ),
    Component(
        name="EphemerisSystem",
        module_path="e2m2e.algorithm.dynamics.ephemeris_system",
        dependencies=["SPICEManager"],
        layer="algorithm",
        description="Ephemeris system configuration",
    ),
    Component(
        name="EphemerisDynamics",
        module_path="e2m2e.algorithm.dynamics.ephemeris_dynamics",
        dependencies=["Dynamics", "EphemerisSystem"],
        layer="algorithm",
        description="Ephemeris N-body dynamics (legacy, internal to multiple shooting)",
    ),
    Component(
        name="DifferentialCorrection",
        module_path="e2m2e.algorithm.solver.differential_correction",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="Differential correction problem construction entry",
    ),
    Component(
        name="CorrectionConfig",
        module_path="e2m2e.algorithm.family.strategies.base",
        layer="algorithm",
        description="Immutable correction strategy configuration",
    ),
    Component(
        name="symmetric_2d_fixed_x0",
        module_path="e2m2e.algorithm.family.strategies.symmetric_2d",
        dependencies=["CorrectionConfig"],
        layer="algorithm",
        description="2D symmetric fixed-x0 correction strategy",
    ),
    Component(
        name="symmetric_xz_fixed_z0",
        module_path="e2m2e.algorithm.family.strategies.symmetric_3d",
        dependencies=["CorrectionConfig"],
        layer="algorithm",
        description="XZ-symmetric fixed-z0 correction strategy",
    ),
    Component(
        name="Continuation",
        module_path="e2m2e.algorithm.solver.continuation",
        dependencies=["DifferentialCorrection"],
        layer="algorithm",
        description="Orbit family continuation",
    ),
    Component(
        name="StabilityAnalysis",
        module_path="e2m2e.algorithm.stability",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="Floquet stability analysis",
    ),
    Component(
        name="MultipleShooting",
        module_path="e2m2e.algorithm.solver.multiple_shooting",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="Multiple shooting problem construction entry",
    ),
    Component(
        name="compute_F_and_dF_symmetric_xz_plane",
        module_path="e2m2e.algorithm.solver.continuation",
        dependencies=["CR3BP_Dynamics"],
        layer="algorithm",
        description="XZ-symmetric constraint function and Jacobian",
    ),
]
