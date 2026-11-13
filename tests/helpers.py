"""Shared tiny problem builders for the unit tests."""

import numpy as np

from physics.fv2d import Aquifer, Grid
from stoch.random_fields import KLExpansion, ScenarioGenerator
from stoch.saa import Archive, ForwardConfig


def tiny_archive(nsteps=40, dt=10.0, n_scen=2, seed0=10, n_wells=2):
    grid = Grid(16, 8, 64.0, 32.0)
    kl = KLExpansion(grid, sigma=0.4, ell_x=15.0, ell_y=12.0,
                     n_kl=0, target_variance=0.95)
    gen = ScenarioGenerator(grid, kl)
    scenarios = [gen.sample(seed0 + i) for i in range(n_scen)]
    aq = Aquifer(K=scenarios[0].K, phi=0.3, R=1.0, b=10.0,
                 alpha_l=5.0, alpha_t=0.5)
    fwd = ForwardConfig(
        dt=dt, nsteps=nsteps, grad=1e-3,
        source_xy=(18.0, 16.0), source_q=20.0, source_c=100.0,
        source_days=50.0,
        compliance_xy=((36.0, 16.0),),
        obs_xy=((28.0, 16.0),),
        eps_well=1.0,
    )
    if n_wells == 2:
        wells = np.array([[26.0, 10.0], [26.0, 22.0]])
    else:
        wells = np.array([[26.0, 16.0]])
    return Archive(grid, aq, wells, scenarios, fwd)
