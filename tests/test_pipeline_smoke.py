"""Smoke-Tests fuer die einheitliche Trainings- und Evaluations-Pipeline (Blatt 11, Aufg. 4)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from actor_critic_project.utils.algo_registry import (
    ALGO_REGISTRY,
    compatible_pairs,
    supports,
)
from actor_critic_project.utils.metrics import auc_return_per_step, final_mean_return
from actor_critic_project.utils.training import train_one

EXPECTED_ALGOS = {
    "mini_batch_reinforce",
    "a2c",
    "ppo",
    "trpo",
    "ars",
    "ddpg",
    "td3",
    "sac",
    "tqc",
}

_CLASSIC_CONTROL_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
]


def test_algo_registry_exhaustive():
    """Alle 9 erwarteten Algorithmen sind im ALGO_REGISTRY vorhanden."""
    assert EXPECTED_ALGOS == set(ALGO_REGISTRY.keys())
    assert supports("ddpg", "CartPole-v1") is False
    assert supports("a2c", "CartPole-v1") is True


def test_compatible_pairs_counts():
    """Fuer alle 5 Classic-Control-Envs muessen genau 33 kompatible Paare existieren.

    3 Discrete-Envs x 5 Algos (Discrete+Box) = 15
    2 Box-Envs x 9 Algos (alle) = 18
    Gesamt: 33
    """
    pairs = compatible_pairs(_CLASSIC_CONTROL_ENVS)
    assert len(pairs) == 33, (
        f"Erwartet 33 kompatible Paare, erhalten {len(pairs)}:\n{pairs}"
    )


def test_train_one_smoke_a2c_cartpole():
    """A2C auf CartPole-v1 laeuft fehlerfrei durch und liefert ein gueltiges TrainResult."""
    result = train_one("a2c", "CartPole-v1", total_timesteps=500, seed=0)
    assert result.wallclock_s > 0
    assert result.algo == "a2c"
    assert result.env_id == "CartPole-v1"
    assert result.seed == 0
    assert result.total_timesteps == 500
    assert result.model_path is None
    assert result.monitor_csv_path is None


def test_train_one_smoke_reinforce_cartpole():
    """Mini-batch REINFORCE auf CartPole-v1 laeuft fehlerfrei durch."""
    result = train_one(
        "mini_batch_reinforce",
        "CartPole-v1",
        total_timesteps=200,
        seed=0,
        hyperparams={"n_episodes_per_update": 2},
    )
    assert result.wallclock_s > 0
    assert result.algo == "mini_batch_reinforce"


def test_metrics_basic():
    """Basistest fuer finale Metriken auf synthetischen Daten."""
    returns = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert final_mean_return(returns, last_k=3) == pytest.approx(40.0)

    # Synthetisches monitor_df: 5 Episoden, je 10 Schritte, steigender Return
    monitor_df = pd.DataFrame({
        "r": [10.0, 20.0, 30.0, 40.0, 50.0],
        "l": [10, 10, 10, 10, 10],
        "t": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    monitor_df["step_cumsum"] = monitor_df["l"].cumsum()

    auc = auc_return_per_step(monitor_df)
    assert auc > 0, f"auc_return_per_step sollte positiv sein, erhalten: {auc}"
