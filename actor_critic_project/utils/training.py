"""Einheitliche Trainings-Pipeline fuer alle (algo, env, seed)-Kombinationen."""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .algo_registry import get_algo
from .seeds import set_global_seeds


@dataclass
class TrainResult:
    """Ergebnis eines Trainingslaufs."""

    algo: str
    env_id: str
    seed: int
    total_timesteps: int
    wallclock_s: float
    model_path: pathlib.Path | None
    monitor_csv_path: pathlib.Path | None


def train_one(
    algo_name: str,
    env_id: str,
    total_timesteps: int,
    seed: int,
    hyperparams: dict | None = None,
    device: str = "cpu",
    log_dir: pathlib.Path | None = None,
) -> TrainResult:
    """Trainiert einen Algorithmus auf einer Gymnasium-Umgebung.

    Setzt globale Seeds, wrappt das Env in Monitor + DummyVecEnv,
    und schreibt Modell + Monitor-CSV nach log_dir (wenn angegeben).

    Args:
        algo_name: Name des Algorithmus gemaess ALGO_REGISTRY.
        env_id: Gymnasium-Umgebungs-ID (z.B. 'CartPole-v1').
        total_timesteps: Trainingslaenge in Umgebungsschritten.
        seed: Zufallsinitialisierung fuer Reproduzierbarkeit.
        hyperparams: Optionale Hyperparameter, die an den Algorithmus weitergegeben werden.
        device: PyTorch-Device ('cpu', 'cuda', 'auto').
        log_dir: Verzeichnis fuer Modell und Monitor-CSV; None deaktiviert das Speichern.

    Returns:
        TrainResult mit Wanduhrzeit, Modell- und Monitor-CSV-Pfaden.
    """
    set_global_seeds(seed)

    monitor_stem: str | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        monitor_stem = str(log_dir / f"{algo_name}_{env_id}_seed{seed}")

    raw_env = gym.make(env_id)
    monitored = Monitor(raw_env, filename=monitor_stem)
    vec_env = DummyVecEnv([lambda e=monitored: e])

    spec = get_algo(algo_name)
    model = spec.cls(
        spec.policy,
        vec_env,
        seed=seed,
        device=device,
        verbose=0,
        **(hyperparams or {}),
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    wallclock_s = time.perf_counter() - t0

    model_path: pathlib.Path | None = None
    monitor_csv_path: pathlib.Path | None = None

    if log_dir is not None:
        save_stem = str(log_dir / f"{algo_name}_{env_id}_seed{seed}")
        model.save(save_stem)
        model_path = pathlib.Path(f"{save_stem}.zip")
        monitor_csv_path = pathlib.Path(f"{monitor_stem}.monitor.csv")

    vec_env.close()

    return TrainResult(
        algo=algo_name,
        env_id=env_id,
        seed=seed,
        total_timesteps=total_timesteps,
        wallclock_s=wallclock_s,
        model_path=model_path,
        monitor_csv_path=monitor_csv_path,
    )
