"""Tests for the Rust integrator extension."""


def test_hello_integrators_smoke():
    """Smoke test: the Rust extension module imports and responds."""
    from e2m2e._integrators import hello_integrators

    assert hello_integrators() == "hello from e2m2e-integrators"
