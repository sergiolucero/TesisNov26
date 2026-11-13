"""Stochastic optimization layer for the pump-and-treat problem.

Modules
-------
random_fields : KL expansion, log-normal K, scenario archive
wells         : hexagonal lattice, CVT, layout diagnostics
objective     : J, G, H, cleanup time, CVaR helpers
saa           : scenario archive, sample-average approximation
chance        : exact empirical / CVaR / sigmoid reformulations
adjoint       : discrete adjoint and gradient checks
optimize      : pattern search, SLSQP+adjoint, multi-start
report        : manifests, metrics, figures
config        : YAML configuration and quick overrides
"""

__all__ = [
    "random_fields", "wells", "objective", "saa", "chance",
    "adjoint", "optimize", "report", "config",
]
