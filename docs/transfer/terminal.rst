Terminal Conditions
===================

Terminal conditions define what departure and arrival must satisfy in transfer
design.

TerminalCondition
~~~~~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.terminal.TerminalCondition` is the abstract
base defining how departure/arrival states are obtained:

.. code-block:: python

   from e2m2e.algorithm.transfer.terminal import TerminalCondition

   class TerminalCondition(ABC):
       def get_initial_state(self) -> np.ndarray:
           """Return departure state [x, y, z, vx, vy, vz]"""
           ...

       def get_arrival_state(
           self, t_ins: float, dynamics: CR3BP_Dynamics
       ) -> tuple[np.ndarray, np.ndarray]:
           """Return arrival position & velocity, two (3,) arrays"""
           ...

OrbitTerminal
~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.terminal.OrbitTerminal` sits on an ``Orbit``:

- Departure state = orbit's first point
- Arrival state obtained by propagation (``dynamics`` passed at call time)

Constructor takes only the orbit; whether it serves as departure or arrival is
decided by its position at optimizer assembly.

.. code-block:: python

   from e2m2e.algorithm.transfer.terminal import OrbitTerminal

   departure = OrbitTerminal(dro_orbit)
   arrival = OrbitTerminal(ro_orbit)

StateTerminal
~~~~~~~~~~~~~

:class:`~e2m2e.algorithm.transfer.terminal.StateTerminal` pins state & time:

- Both departure & arrival are fixed values
- No dependence on propagation

.. code-block:: python

   from e2m2e.algorithm.transfer.terminal import StateTerminal

   departure = StateTerminal(state=np.array([r, 0, 0, 0, v, 0]), time=0.0)

Integration with optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inject terminals via ``Transfer.set_orbit()``:

.. code-block:: python

   from e2m2e.algorithm.transfer import Transfer

   transfer = Transfer(dynamics)
   transfer.set_orbit(start=dro_orbit, end=ro_orbit)

   # Internally creates OrbitTerminals for both ends
