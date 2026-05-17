"""Sweep-Konfiguration fuer Blatt 11, Aufgabe 4 (a): Evaluation Study."""

from __future__ import annotations

from dataclasses import dataclass, field

from actor_critic_project.utils.algo_registry import compatible_pairs

_DEFAULT_ALGOS: tuple[str, ...] = (
    "mini_batch_reinforce",
    "a2c",
    "ppo",
    "trpo",
    "ars",
    "ddpg",
    "td3",
    "sac",
    "tqc",
)

_DEFAULT_ENVS: tuple[str, ...] = (
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
)

_QUICK_ALGOS: tuple[str, ...] = ("mini_batch_reinforce", "a2c", "ppo")
_QUICK_ENVS: tuple[str, ...] = ("CartPole-v1", "Pendulum-v1")


@dataclass(frozen=True)
class RunSpec:
    """Einzelner Trainingslauf: Algorithmus, Umgebung, Seed."""

    algo: str
    env_id: str
    seed: int


@dataclass(frozen=True)
class SweepConfig:
    """Konfiguration fuer den vollstaendigen Evaluation-Sweep.

    quick=True erzwingt eine minimale Konfiguration fuer schnelle Tests:
    3 Algos x 2 Envs x 1 Seed x 5_000 Timesteps.
    """

    algos: tuple[str, ...] = _DEFAULT_ALGOS
    env_ids: tuple[str, ...] = _DEFAULT_ENVS
    seeds: tuple[int, ...] = (0, 1, 2)
    total_timesteps: int = 100_000
    device: str = "cpu"
    quick: bool = False

    def __post_init__(self) -> None:
        if self.quick:
            object.__setattr__(self, "algos", _QUICK_ALGOS)
            object.__setattr__(self, "env_ids", _QUICK_ENVS)
            object.__setattr__(self, "seeds", (0,))
            object.__setattr__(self, "total_timesteps", 5_000)

    def expanded_runs(self) -> list[RunSpec]:
        """Alle gueltigen (algo, env, seed)-Tripel basierend auf algo_registry.supports().

        Filtert automatisch inkompatible Algo-Env-Kombinationen heraus.
        """
        pairs = compatible_pairs(self.env_ids)
        return [
            RunSpec(algo=algo, env_id=env_id, seed=seed)
            for algo, env_id in pairs
            if algo in self.algos
            for seed in self.seeds
        ]
