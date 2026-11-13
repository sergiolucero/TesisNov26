"""Forward physics for the stochastic pump-and-treat optimization.

The modules here are deliberately small and dependency-light (numpy/scipy)
so that the optimization layer can import them without forking the trusted
1D PINN/FNO scripts.

Modules
-------
fv2d        : 2D cell-centred finite-volume flow + transport with wells
references  : analytic references (Thiem, Ogata--Banks)
"""

from .fv2d import (  # noqa: F401
    Aquifer,
    FlowModel,
    FlowResponses,
    Grid,
    TransportResult,
    advection_operator,
    face_dispersion,
    flow_responses,
    gaussian_well_weights,
    transport_run,
)
from .references import ogata_banks, thiem_drawdown  # noqa: F401

__all__ = [
    "Aquifer",
    "FlowModel",
    "FlowResponses",
    "Grid",
    "TransportResult",
    "advection_operator",
    "face_dispersion",
    "flow_responses",
    "gaussian_well_weights",
    "transport_run",
    "ogata_banks",
    "thiem_drawdown",
]
