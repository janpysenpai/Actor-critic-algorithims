"""Seed-Utilities für reproduzierbare Experimente."""

from __future__ import annotations

import random

import numpy as np
import torch as th


def set_global_seeds(seed: int) -> None:
    """Setzt numpy-, torch- und Python-Random-Seeds.

    Hinweis: env.reset(seed=seed) muss zusätzlich explizit aufgerufen
    werden, da Gymnasium-Envs den globalen Seed nicht automatisch übernehmen.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
