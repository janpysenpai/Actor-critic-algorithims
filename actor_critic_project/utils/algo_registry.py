"""Zentrales Algorithmus-Register fuer alle Classic-Control-Experimente (Blatt 11, Aufg. 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class AlgoSpec:
    """Beschreibung eines Algorithmus: Klasse, Policy-Token, unterstuetzte Action-Spaces."""

    name: str
    cls: type
    policy: str
    supported_spaces: tuple[type, ...]


def _build_registry() -> dict[str, AlgoSpec]:
    from actor_critic_project.algos.mini_batch_reinforce import MiniBatchREINFORCE
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    registry: dict[str, AlgoSpec] = {
        "mini_batch_reinforce": AlgoSpec(
            name="mini_batch_reinforce",
            cls=MiniBatchREINFORCE,
            policy="MlpPolicy",
            supported_spaces=(spaces.Discrete, spaces.Box),
        ),
        "a2c": AlgoSpec(
            name="a2c",
            cls=A2C,
            policy="MlpPolicy",
            supported_spaces=(spaces.Discrete, spaces.Box),
        ),
        "ppo": AlgoSpec(
            name="ppo",
            cls=PPO,
            policy="MlpPolicy",
            supported_spaces=(spaces.Discrete, spaces.Box),
        ),
        "ddpg": AlgoSpec(
            name="ddpg",
            cls=DDPG,
            policy="MlpPolicy",
            supported_spaces=(spaces.Box,),
        ),
        "td3": AlgoSpec(
            name="td3",
            cls=TD3,
            policy="MlpPolicy",
            supported_spaces=(spaces.Box,),
        ),
        "sac": AlgoSpec(
            name="sac",
            cls=SAC,
            policy="MlpPolicy",
            supported_spaces=(spaces.Box,),
        ),
    }

    try:
        from sb3_contrib import ARS, TRPO, TQC

        registry["trpo"] = AlgoSpec(
            name="trpo",
            cls=TRPO,
            policy="MlpPolicy",
            supported_spaces=(spaces.Discrete, spaces.Box),
        )
        registry["ars"] = AlgoSpec(
            name="ars",
            cls=ARS,
            policy="LinearPolicy",
            supported_spaces=(spaces.Discrete, spaces.Box),
        )
        registry["tqc"] = AlgoSpec(
            name="tqc",
            cls=TQC,
            policy="MlpPolicy",
            supported_spaces=(spaces.Box,),
        )
    except ImportError:
        pass

    return registry


ALGO_REGISTRY: dict[str, AlgoSpec] = _build_registry()


def get_algo(name: str) -> AlgoSpec:
    """Gibt AlgoSpec fuer den gegebenen Namen zurueck.

    Raises KeyError mit klarer Fehlermeldung bei unbekanntem Namen.
    """
    if name not in ALGO_REGISTRY:
        known = sorted(ALGO_REGISTRY.keys())
        raise KeyError(f"Unbekannter Algorithmus '{name}'. Verfuegbar: {known}")
    return ALGO_REGISTRY[name]


def supports(name: str, env_id: str) -> bool:
    """Prueft ob der Algorithmus den Action-Space der Umgebung unterstuetzt.

    Instanziiert das Env einmal, liest den Action-Space-Typ, schliesst es.
    """
    spec = get_algo(name)
    env = gym.make(env_id)
    result = isinstance(env.action_space, spec.supported_spaces)
    env.close()
    return result


def compatible_pairs(env_ids: Iterable[str]) -> list[tuple[str, str]]:
    """Alle (algo_name, env_id)-Paare, bei denen supports() True ist.

    Pro env_id wird die Umgebung einmal instanziiert (effizient).
    """
    result: list[tuple[str, str]] = []
    for env_id in env_ids:
        env = gym.make(env_id)
        action_space = env.action_space
        env.close()
        for algo_name, spec in ALGO_REGISTRY.items():
            if isinstance(action_space, spec.supported_spaces):
                result.append((algo_name, env_id))
    return result
