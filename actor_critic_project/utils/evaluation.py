"""Einheitliche Evaluations-Pipeline fuer SB3-kompatible Modelle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


@dataclass
class EvalResult:
    """Ergebnis der Policy-Evaluation ueber mehrere Episoden."""

    mean_return: float
    std_return: float
    n_episodes: int
    returns: list[float] = field(default_factory=list)


def evaluate_model(
    model: Any,
    env_id: str,
    n_episodes: int = 20,
    seed: int = 0,
) -> EvalResult:
    """Evaluiert ein SB3-kompatibles Modell auf einer frischen Gymnasium-Umgebung.

    Nutzt evaluate_policy mit deterministic=False. Das Eval-Env wird separat
    erstellt und nach der Evaluation geschlossen.

    Args:
        model: Trainiertes SB3-Modell (oder kompatibles Interface).
        env_id: Gymnasium-Umgebungs-ID.
        n_episodes: Anzahl der Evaluationsepisoden.
        seed: Seed fuer den initialen Reset des Eval-Envs.

    Returns:
        EvalResult mit mean_return, std_return und Einzel-Returns.
    """
    raw_env = gym.make(env_id)
    eval_env = Monitor(raw_env)
    eval_env.reset(seed=seed)

    episode_rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=False,
        return_episode_rewards=True,
    )

    eval_env.close()

    return EvalResult(
        mean_return=float(np.mean(episode_rewards)),
        std_return=float(np.std(episode_rewards)),
        n_episodes=n_episodes,
        returns=list(episode_rewards),
    )
