"""
Orbital MCP Server - Model Context Protocol server for orbital mechanics calculations.

This server exposes tools for CR3BP dynamics, orbit analysis, and trajectory propagation
using the e2m2e library.
"""

import asyncio
import json
from typing import Any

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent

# Import e2m2e core modules
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.system import CR3BP_System
from e2m2e.core.orbit import Orbit

# Initialize FastMCP server
mcp = FastMCP(name="orbital-mcp")


# =============================================================================
# MCP Tools - CR3BP Dynamics
# =============================================================================

@mcp.tool(
    name="propagate_trajectory",
    description="""Propagate a trajectory in the Circular Restricted Three-Body Problem (CR3BP).
    
    Input:
    - mu: Mass parameter of the system (float, e.g., 0.012150585609 for Earth-Moon)
    - state: Initial 6D state vector [x, y, z, vx, vy, vz] in normalized units
    - t_span: Time interval as [t_start, t_end], e.g., [0, 1.0]
    - t_eval: Optional list of times at which to store the solution
    - with_stm: If True, compute State Transition Matrix along the trajectory
    
    Output:
    - Dictionary containing trajectory data with times, states, and optionally STM
    """
)
async def propagate_trajectory(
    mu: float,
    state: list[float],
    t_span: list[float],
    t_eval: list[float] | None = None,
    with_stm: bool = False
) -> dict[str, Any]:
    """Propagate a trajectory in CR3BP."""
    try:
        # Validate inputs
        if len(state) != 6:
            return {"error": "State must be a 6D vector [x, y, z, vx, vy, vz]"}
        if len(t_span) != 2:
            return {"error": "t_span must be [t_start, t_end]"}
        
        # Create system and dynamics
        system = CR3BP_System(mu=mu)
        dynamics = CR3BP_Dynamics(system)
        
        # Propagate
        result = dynamics.propagate(
            initial_state=state,
            t_span=t_span,
            t_eval=t_eval,
            with_stm=with_stm
        )
        
        # Extract results
        t = result.t.tolist()
        y = result.y.T.tolist()  # Shape: (n_points, 6)
        
        response = {
            "success": True,
            "n_points": len(t),
            "times": t,
            "states": y,
            "initial_state": state,
            "t_span": t_span
        }
        
        # Include STM if requested
        if with_stm and dynamics.last_stm is not None:
            response["has_stm"] = True
            # STM at final time (monodromy matrix if t_span[1] is period)
            response["final_stm"] = dynamics.last_stm.tolist()
        
        return response
        
    except Exception as e:
        return {"error": f"Propagation failed: {str(e)}", "success": False}


@mcp.tool(
    name="compute_stm",
    description="""Compute the State Transition Matrix (STM) for CR3BP dynamics.
    
    The STM maps small variations in initial state to variations at a later time:
    delta_state(t) = STM * delta_state(t0)
    
    Input:
    - mu: Mass parameter of the system
    - state: Initial 6D state vector [x, y, z, vx, vy, vz]
    - t: Time at which to compute STM (from t0=0)
    
    Output:
    - The 6x6 State Transition Matrix
    """
)
async def compute_stm(
    mu: float,
    state: list[float],
    t: float
) -> dict[str, Any]:
    """Compute State Transition Matrix at time t."""
    try:
        if len(state) != 6:
            return {"error": "State must be a 6D vector"}
        
        system = CR3BP_System(mu=mu)
        dynamics = CR3BP_Dynamics(system)
        
        stm = dynamics.compute_state_transition_matrix(
            initial_state=state,
            t=t
        )
        
        return {
            "success": True,
            "state": state,
            "time": t,
            "stm": stm.tolist()
        }
        
    except Exception as e:
        return {"error": f"STM computation failed: {str(e)}", "success": False}


@mcp.tool(
    name="compute_jacobi",
    description="""Compute the Jacobi constant for a given state in CR3BP.
    
    The Jacobi constant is a conserved quantity in CR3BP, proportional to -2*V where V is the potential.
    Used for orbit classification and family continuation.
    
    Input:
    - mu: Mass parameter of the system
    - state: 6D state vector [x, y, z, vx, vy, vz]
    
    Output:
    - Jacobi constant value (energy-like quantity, lower = more energetic)
    """
)
async def compute_jacobi(
    mu: float,
    state: list[float]
) -> dict[str, Any]:
    """Compute Jacobi constant for a given state."""
    try:
        if len(state) != 6:
            return {"error": "State must be a 6D vector"}
        
        system = CR3BP_System(mu=mu)
        dynamics = CR3BP_Dynamics(system)
        
        jacobi = dynamics.compute_jacobi_constant(state)
        
        return {
            "success": True,
            "state": state,
            "jacobi_constant": float(jacobi)
        }
        
    except Exception as e:
        return {"error": f"Jacobi computation failed: {str(e)}", "success": False}


@mcp.tool(
    name="check_crossing",
    description="""Check if a trajectory crosses a Poincaré section.
    
    Poincaré sections are useful for analyzing periodic orbits and chaos.
    
    Input:
    - mu: Mass parameter of the system
    - state: 6D state vector at current time
    - plane: The cross-section plane ('x', 'y', 'z', 'vx', 'vy', or 'vz')
    - value: The value at which the plane is defined
    
    Output:
    - Dictionary with crossing information
    """
)
async def check_crossing(
    mu: float,
    state: list[float],
    plane: str,
    value: float
) -> dict[str, Any]:
    """Check if state crosses a Poincaré section."""
    try:
        if plane not in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
            return {"error": "Plane must be one of: x, y, z, vx, vy, vz"}
        
        system = CR3BP_System(mu=mu)
        dynamics = CR3BP_Dynamics(system)
        
        is_crossing, crossing_state, crossing_time = dynamics.check_cross_section(
            state=state,
            plane=plane,
            value=value
        )
        
        return {
            "success": True,
            "is_crossing": bool(is_crossing),
            "plane": plane,
            "value": value,
            "crossing_state": crossing_state.tolist() if is_crossing else None
        }
        
    except Exception as e:
        return {"error": f"Crossing check failed: {str(e)}", "success": False}


# =============================================================================
# MCP Tools - Orbit Analysis
# =============================================================================

@mcp.tool(
    name="analyze_orbit",
    description="""Analyze an orbit to extract key properties.
    
    Input:
    - states: List of 6D state vectors [[x,y,z,vx,vy,vz], ...]
    - times: List of corresponding times
    - family_type: Optional string describing orbit family ('dro', 'ro', 'halo', 'lyapunov', etc.)
    
    Output:
    - Dictionary with period, amplitudes, stability index, and Jacobi constant
    """
)
async def analyze_orbit(
    states: list[list[float]],
    times: list[float],
    family_type: str = "unknown"
) -> dict[str, Any]:
    """Analyze an orbit to get period, amplitudes, stability."""
    try:
        import numpy as np
        
        if len(states) < 2 or len(states) != len(times):
            return {"error": "Invalid states or times array"}
        
        # Create Orbit object
        orbit = Orbit(states=states, times=times)
        orbit.family_type = family_type
        
        # Compute properties
        period = orbit.get_period()
        
        # Get amplitudes in each direction
        amp_x = orbit.get_amplitude("x")
        amp_y = orbit.get_amplitude("y")
        amp_z = orbit.get_amplitude("z")
        
        # Compute stability (may fail for non-periodic)
        try:
            orbit.compute_stability()
            stability_indices = orbit.stability_indices.tolist() if orbit.stability_indices is not None else None
        except:
            stability_indices = None
        
        # Compute Jacobi for middle state
        from e2m2e.core.dynamics import CR3BP_Dynamics
        from e2m2e.core.system import CR3BP_System
        
        mid_state = states[len(states)//2]
        try:
            system = CR3BP_System()  # Default Earth-Moon
            dynamics = CR3BP_Dynamics(system)
            jacobi = dynamics.compute_jacobi_constant(mid_state)
        except:
            jacobi = None
        
        return {
            "success": True,
            "n_points": len(states),
            "period": float(period) if period else None,
            "amplitudes": {
                "x": float(amp_x) if amp_x else None,
                "y": float(amp_y) if amp_y else None,
                "z": float(amp_z) if amp_z else None
            },
            "stability_indices": stability_indices,
            "jacobi_constant": float(jacobi) if jacobi else None,
            "family_type": family_type
        }
        
    except Exception as e:
        return {"error": f"Orbit analysis failed: {str(e)}", "success": False}


@mcp.tool(
    name="get_orbit_period",
    description="""Calculate the period of an orbit from state data.
    
    Input:
    - states: List of 6D state vectors
    - times: Corresponding time values
    
    Output:
    - The orbital period
    """
)
async def get_orbit_period(
    states: list[list[float]],
    times: list[float]
) -> dict[str, Any]:
    """Get period of an orbit."""
    try:
        if len(states) != len(times):
            return {"error": "states and times must have same length"}
        
        orbit = Orbit(states=states, times=times)
        period = orbit.get_period()
        
        return {
            "success": True,
            "period": float(period) if period else None,
            "n_points": len(states)
        }
        
    except Exception as e:
        return {"error": f"Period calculation failed: {str(e)}", "success": False}


@mcp.tool(
    name="get_orbit_amplitude",
    description="""Calculate the amplitude of an orbit in a specified direction.
    
    Input:
    - states: List of 6D state vectors
    - direction: 'x', 'y', or 'z'
    
    Output:
    - The amplitude (max - min) / 2 in the specified direction
    """
)
async def get_orbit_amplitude(
    states: list[list[float]],
    direction: str
) -> dict[str, Any]:
    """Get amplitude of an orbit in specified direction."""
    try:
        if direction not in ['x', 'y', 'z']:
            return {"error": "Direction must be 'x', 'y', or 'z'"}
        
        # Extract position states
        pos_indices = {'x': 0, 'y': 1, 'z': 2}
        idx = pos_indices[direction]
        positions = [s[idx] for s in states]
        
        amplitude = (max(positions) - min(positions)) / 2
        center = (max(positions) + min(positions)) / 2
        
        return {
            "success": True,
            "direction": direction,
            "amplitude": float(amplitude),
            "center": float(center),
            "min": float(min(positions)),
            "max": float(max(positions))
        }
        
    except Exception as e:
        return {"error": f"Amplitude calculation failed: {str(e)}", "success": False}


# =============================================================================
# MCP Resources - Provide orbit family data
# =============================================================================

@mcp.resource(
    uri="resource://earth_moon_system",
    name="EarthMoonSystem",
    description="Basic parameters for Earth-Moon CR3BP system"
)
def get_earth_moon_system() -> str:
    """Return Earth-Moon system parameters."""
    system = CR3BP_System()  # Default Earth-Moon
    return json.dumps({
        "name": "Earth-Moon",
        "mu": system.mu,
        "L1": float(system.L1) if hasattr(system, 'L1') else None,
        "L2": float(system.L2) if hasattr(system, 'L2') else None,
        "L3": float(system.L3) if hasattr(system, 'L3') else None,
    }, indent=2)


@mcp.resource(
    uri="resource://dro_families",
    name="DROFamilies",
    description="Distant Retrograde Orbit family data"
)
def get_dro_families() -> str:
    """Return available DRO family data files."""
    import os
    from pathlib import Path
    
    # Look for DRO data in e2m2e output
    base_path = Path(__file__).parent.parent / "output" / "dro"
    
    files = []
    if base_path.exists():
        for f in base_path.glob("*.json"):
            files.append({
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size
            })
    
    return json.dumps({
        "count": len(files),
        "files": files
    }, indent=2)


# =============================================================================
# MCP Prompts - Reusable prompt templates
# =============================================================================

@mcp.prompt(
    name="orbit_analysis",
    description="Analyze an orbit for periodic behavior and stability"
)
def orbit_analysis_prompt(states: str, times: str) -> str:
    """Prompt for analyzing orbit periodicity and stability."""
    return f"""Analyze the following orbit data to determine:
1. Is the orbit periodic?
2. What is its period?
3. What are the amplitudes in x, y, z?
4. Is the orbit stable (linear stability)?

States (6D): {states}
Times: {times}

Provide a comprehensive analysis."""


@mcp.prompt(
    name="transfer_design",
    description="Design a transfer between two orbits"
)
def transfer_design_prompt(
    origin_orbit: str,
    target_orbit: str,
    constraints: str = ""
) -> str:
    """Prompt for designing orbital transfers."""
    return f"""Design a two-impulse or multi-impulse transfer:
    
Origin: {origin_orbit}
Target: {target_orbit}
Constraints: {constraints}

Consider:
- ΔV requirements
- Transfer geometry
- Time of flight
- Mid-course corrections

Provide detailed transfer trajectory design."""


# =============================================================================
# Server Entry Point
# =============================================================================

def main():
    """Main entry point for the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
