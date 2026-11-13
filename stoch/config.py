"""Configuration loading and the quick-mode overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

__all__ = ["default_config", "load_config", "apply_quick"]


def default_config() -> dict:
    return {
        "domain": {"lx": 200.0, "ly": 100.0, "nx": 80, "ny": 40},
        "aquifer": {
            "k0": 10.0, "b": 10.0, "phi": 0.30, "R": 1.0,
            "alpha_l": 5.0, "alpha_t": 0.5, "decay": 0.0,
        },
        "flow": {"grad": 1e-3},
        "random_field": {
            "sigma": 0.5, "ell_x": 20.0, "ell_y": 20.0,
            "n_kl": 0, "target_variance": 0.95,
            "random_source": False, "random_gradient": False,
        },
        "wells": {
            "n": 9, "method": "hex", "q_max": 50.0, "Q_max": 200.0,
            "eps": 1.5, "plume_threshold": 0.05,
        },
        "source": {"x": 30.0, "y": 50.0, "q": 10.0, "c": 100.0,
                   "days": 365.0},
        "compliance": {
            "x": [185.0], "y": [50.0], "creg": 1.0, "window_start": 0.8,
        },
        "observation": {"x": [100.0], "y": [50.0], "h_min": -10.0},
        "time": {"dt": 36.5, "nsteps": 100},
        "saa": {"M_train": 40, "M_test": 80, "alpha": 0.05, "alpha_h": 0.01},
        "cost": {
            "c_e": 1.0, "c_w": 1.0, "c_m": 1.0, "c_tau": 1.0,
            "c_fail": 10.0, "eps_time": 0.05, "time_mode": "soft",
        },
        "optim": {
            "method": "pattern", "n_starts": 3, "max_iter": 60,
            "max_evals": 1500, "rho_g": 1000000.0, "rho_h": 1000000.0,
            "chance": "cvar", "seed": 0,
        },
        "seed": {"train": 1000, "test": 500000},
        "plot": True,
    }


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def apply_quick(cfg: dict) -> dict:
    """Small, fast settings for a smoke run (not physically converged)."""
    quick = {
        "domain": {"nx": 30, "ny": 15},
        "random_field": {"n_kl": 0, "target_variance": 0.95},
        "wells": {"n": 4},
        "time": {"dt": 36.5, "nsteps": 40},
        "saa": {"M_train": 8, "M_test": 16},
        "optim": {"max_iter": 12, "max_evals": 160, "n_starts": 1},
    }
    return _deep_update(deepcopy(cfg), quick)


def load_config(path: Optional[str | Path] = None, quick: bool = False) -> dict:
    cfg = default_config()
    if path is not None and Path(path).exists():
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if quick:
        cfg = apply_quick(cfg)
    return cfg
