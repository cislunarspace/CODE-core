Atmosphere Models
=================

e2m2e provides pluggable atmosphere density models, dependency-injected into
``DragModel``. The current implementation is a US Standard Atmosphere 1976
piecewise-exponential model covering 0–1000 km altitude.

**ExponentialAtmosphere**: USSA76 piecewise-exponential densities via
``density(altitude)``, with first-order F10.7 / Ap corrections.

Within each layer ``ρ(h) = ρ₀ · exp(-(h - h₀) / H)``; layer scale heights derive
from adjacent breakpoint density ratios — continuous and monotonically
decreasing. F10.7 (solar radio flux) and Ap (geomagnetic index) scale the base
density linearly.

.. code-block:: python

   from e2m2e.algorithm.forces.atmosphere import ExponentialAtmosphere

   # Defaults: F10.7=150 sfu (moderate solar), Ap=15 (moderate geomagnetic)
   atm = ExponentialAtmosphere()

   rho_surface = atm.density(0.0)      # 1.225 kg/m³
   rho_100km   = atm.density(100.0)    # 5.604e-7 kg/m³
   rho_400km   = atm.density(400.0)    # 2.803e-12 kg/m³
   rho_1000km  = atm.density(1000.0)   # 0.0 (beyond model ceiling)

   atm_high = ExponentialAtmosphere(f107=200, ap=50)
   rho_high = atm_high.density(400.0)  # higher than defaults
