"""Minimale Tabular-RL-Baselines fuer den Tabular-vs-Deep-Vergleich (Blatt 11, Aufg. 4 b).

Implementiert Q-Learning und SARSA in purem NumPy gegen Gymnasium-Envs mit diskretem
Zustand- und Aktionsraum. Keine Abhaengigkeit vom originalen Tabular-Projekt —
dient als reproduzierbare Referenz-Baseline im selben Repo.

Ergebnis-CSVs sind SB3-Monitor-kompatibel (r, l, t) fuer load_monitor_csv().
"""

from __future__ import annotations

import csv
import pathlib
import time
from typing import Literal

import numpy as np
import gymnasium as gym

AlgoName = Literal["q_learning", "sarsa"]

TABULAR_ENVS: tuple[str, ...] = ("FrozenLake-v1", "Taxi-v3")


def _epsilon_greedy(Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    return int(np.argmax(Q[state]))


def _run_episode(
    env,
    Q: np.ndarray,
    algo: AlgoName,
    epsilon: float,
    lr: float,
    gamma: float,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Fuehrt eine Episode durch und aktualisiert die Q-Tabelle in-place."""
    obs, _ = env.reset()
    action = _epsilon_greedy(Q, obs, epsilon, rng)
    total_return = 0.0
    length = 0

    while True:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = _epsilon_greedy(Q, next_obs, epsilon, rng)

        # Q-Learning: TD-Target = max Q(s'); SARSA: TD-Target = Q(s', a')
        td_next = (
            Q[next_obs, next_action] if algo == "sarsa" else float(np.max(Q[next_obs]))
        )
        Q[obs, action] += lr * (reward + gamma * td_next * float(not done) - Q[obs, action])

        obs, action = next_obs, next_action
        total_return += float(reward)
        length += 1
        if done:
            break

    return total_return, length


def run_tabular(
    algo: AlgoName,
    env_id: str,
    seed: int,
    n_episodes: int,
    lr: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
) -> tuple[list[float], list[int], float]:
    """Trainiert Q-Learning oder SARSA; gibt (returns, lengths, wallclock_s) zurueck."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))
    epsilon = epsilon_start
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    t0 = time.perf_counter()
    for _ in range(n_episodes):
        ret, length = _run_episode(env, Q, algo, epsilon, lr, gamma, rng)
        episode_returns.append(ret)
        episode_lengths.append(length)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    wallclock_s = time.perf_counter() - t0

    env.close()
    return episode_returns, episode_lengths, wallclock_s


def _write_monitor_csv(
    path: pathlib.Path, returns: list[float], lengths: list[int]
) -> None:
    """Schreibt Episoden-Daten im SB3-Monitor-CSV-Format (kompatibel mit load_monitor_csv)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f'#{{"t_start": {time.time():.4f}, "env_id": null}}\n')
        writer = csv.writer(f)
        writer.writerow(["r", "l", "t"])
        cumtime = 0.0
        for r, l in zip(returns, lengths):
            cumtime += l * 1e-5  # synthetische Wanduhrzeit proportional zur Episodenlaenge
            writer.writerow([round(r, 6), l, round(cumtime, 6)])


def run_and_save_all(
    output_dir: pathlib.Path,
    algos: tuple[str, ...] = ("q_learning", "sarsa"),
    env_ids: tuple[str, ...] = TABULAR_ENVS,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_episodes: int = 2000,
) -> pathlib.Path:
    """Fuehrt alle Tabular-Runs durch und speichert Ergebnisse nach output_dir.

    Erstellt:
    - runs.csv (Episode-Level-Daten + Wallclock pro Run)
    - <algo>_<env_id>_seed<s>.monitor.csv fuer jedes (algo, env_id, seed)-Tripel
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for algo in algos:
        for env_id in env_ids:
            for seed in seeds:
                returns, lengths, wallclock_s = run_tabular(
                    algo, env_id, seed, n_episodes  # type: ignore[arg-type]
                )
                total_steps = sum(lengths)
                cumstep = 0
                for ep, (r, l) in enumerate(zip(returns, lengths)):
                    cumstep += l
                    all_rows.append({
                        "algo": algo,
                        "env_id": env_id,
                        "seed": seed,
                        "episode": ep,
                        "length": l,
                        "return": r,
                        "step_cumsum": cumstep,
                        "wallclock_s": round(wallclock_s, 4),
                        "total_steps": total_steps,
                    })

                monitor_path = output_dir / f"{algo}_{env_id}_seed{seed}.monitor.csv"
                _write_monitor_csv(monitor_path, returns, lengths)

    csv_path = output_dir / "runs.csv"
    fieldnames = [
        "algo", "env_id", "seed", "episode", "length", "return",
        "step_cumsum", "wallclock_s", "total_steps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return csv_path
