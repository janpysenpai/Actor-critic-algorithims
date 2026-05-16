"""Tests für MiniBatchREINFORCE (Algorithmus 32, Skript S. 168)."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th
import gymnasium as gym
from gymnasium import spaces

from actor_critic_project.algos.mini_batch_reinforce import MiniBatchREINFORCE
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.evaluation import evaluate_policy


# ---------------------------------------------------------------------------
# Hilfsklassen
# ---------------------------------------------------------------------------


class TwoStateMDP(gym.Env):
    """Deterministisches Spielzeug-MDP: State 0, Action 0 → Reward 1, terminiert.

    State 0, Action 1 → Reward 0, terminiert. Nur ein Zeitschritt pro Episode.
    Optimales Verhalten: immer Action 0 wählen.
    """

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        reward = 1.0 if int(action) == 0 else 0.0
        return np.array([0.0], dtype=np.float32), reward, True, False, {}

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instantiation():
    """Klasse lässt sich fehlerfrei instanziieren."""
    env = gym.make("CartPole-v1")
    model = MiniBatchREINFORCE("MlpPolicy", env, device="cpu")
    assert model is not None
    env.close()


def test_single_update_cartpole():
    """Smoke-Test: ein Update-Zyklus auf CartPole ohne Exception."""
    env = gym.make("CartPole-v1")
    model = MiniBatchREINFORCE(
        "MlpPolicy",
        env,
        n_episodes_per_update=2,
        seed=0,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=500)
    assert model.num_timesteps > 0
    env.close()


@pytest.mark.slow
def test_learning_improvement_cartpole():
    """Nach 30k Schritten soll der mittlere Return signifikant steigen.

    Beobachteter Bereich über mehrere Seeds (seed=42, 0, 1):
    Vor Training: ~20–30. Nach Training: ~80–300 (instabil, aber klar positiv).
    Schwellenwert 50 ist bewusst konservativ gesetzt, um Flakiness zu vermeiden.
    """
    env = gym.make("CartPole-v1")
    model = MiniBatchREINFORCE(
        "MlpPolicy",
        env,
        n_episodes_per_update=10,
        learning_rate=7e-4,
        gamma=0.99,
        normalize_returns=True,
        seed=42,
        device="cpu",
        verbose=0,
    )

    mean_before, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)

    model.learn(total_timesteps=30_000)

    mean_after, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)

    assert mean_after >= 50, (
        f"Kein signifikanter Lernfortschritt: vor={mean_before:.1f}, nach={mean_after:.1f}"
    )
    env.close()


def test_continuous_action_space():
    """Smoke-Test auf Pendulum-v1 (kontinuierlicher Action-Space)."""
    env = gym.make("Pendulum-v1")
    model = MiniBatchREINFORCE(
        "MlpPolicy",
        env,
        n_episodes_per_update=2,
        seed=0,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=500)
    assert model.num_timesteps > 0
    env.close()


def test_gradient_direction_toy_mdp():
    """Analytischer Sanity-Check: Policy-Gradient zeigt in richtige Richtung.

    Im TwoStateMDP ist Action 0 optimal (Reward 1). Nach einem REINFORCE-Update
    über K=50 Episoden muss P(Action 0 | State 0) gestiegen sein.
    """
    env = TwoStateMDP()
    model = MiniBatchREINFORCE(
        "MlpPolicy",
        env,
        n_episodes_per_update=50,
        learning_rate=1e-2,
        gamma=1.0,
        normalize_returns=False,  # kein Normieren bei konstanten Returns
        ent_coef=0.0,
        seed=0,
        device="cpu",
        verbose=0,
    )

    obs = np.array([[0.0]], dtype=np.float32)
    obs_tensor = obs_as_tensor(obs, th.device("cpu"))

    with th.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        prob_before = dist.distribution.probs[0, 0].item()

    # Genau ein Update (50 Episoden à 1 Schritt = 50 Timesteps)
    model.learn(total_timesteps=50)

    with th.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        prob_after = dist.distribution.probs[0, 0].item()

    assert prob_after > prob_before, (
        f"Gradient zeigt falsche Richtung: P(a=0) vor={prob_before:.4f}, nach={prob_after:.4f}"
    )
    env.close()
