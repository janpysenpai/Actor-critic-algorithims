"""Hyperparam-Loader: liest algo+env-spezifische Hyperparameter aus YAML-Dateien."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "configs" / "hyperparams"
_REINFORCE_CONFIG = (
    pathlib.Path(__file__).resolve().parent.parent
    / "algos"
    / "mini_batch_reinforce"
    / "hyperparams.yml"
)

# Felder, die nie direkt an den SB3-Konstruktor uebergeben werden duerfen
_STRIP_KEYS = frozenset({"n_envs", "n_timesteps", "policy", "normalize"})


def load_hyperparams(algo_name: str, env_id: str) -> dict[str, Any]:
    """Laedt Hyperparameter fuer algo_name + env_id aus der zugehoerigen YAML-Datei.

    Quellen:
    - mini_batch_reinforce: algos/mini_batch_reinforce/hyperparams.yml
    - alle anderen: configs/hyperparams/<algo_name>.yml

    Gibt ein leeres dict zurueck (mit logging.warning), wenn kein Eintrag fuer env_id
    vorhanden ist oder die YAML-Datei fehlt.
    """
    if algo_name == "mini_batch_reinforce":
        yaml_path = _REINFORCE_CONFIG
    else:
        yaml_path = _CONFIGS_DIR / f"{algo_name}.yml"

    if not yaml_path.exists():
        logger.warning(
            "Keine Hyperparam-Datei fuer '%s' gefunden: %s", algo_name, yaml_path
        )
        return {}

    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None or env_id not in data:
        logger.warning(
            "Kein Eintrag fuer env_id '%s' in %s — verwende leere Hyperparameter.",
            env_id,
            yaml_path,
        )
        return {}

    raw = dict(data[env_id])
    stripped = {k: v for k, v in raw.items() if k not in _STRIP_KEYS}
    return stripped
