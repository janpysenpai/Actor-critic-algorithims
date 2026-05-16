"""
Blatt 9 Aufgabe 3 — Gymnasium-API-Tour und SB3-Toolchain-Check auf CartPole-v1.

Dieses Skript deckt die Lernziele von Übungsblatt 9, Aufgabe 3 ab. Es demonstriert
das Gymnasium-API-Pattern, erklärt den Unterschied zwischen On- und Off-Policy-
Algorithmen in Stable Baselines3 und validiert die Toolchain (Gymnasium, SB3,
PyTorch-MPS) auf CartPole-v1.

Das Gymnasium-API-Pattern unterscheidet sich von der älteren OpenAI-Gym-API durch
die explizite Trennung von Terminierung und Truncation. ``env.reset()`` gibt ein
Tupel ``(obs, info)`` zurück; ``env.step(action)`` gibt ``(obs, reward, terminated,
truncated, info)`` zurück. ``terminated=True`` bedeutet, die Episode endet gemäß
MDP-Dynamik (z.B. Balken kippt um); ``truncated=True`` bedeutet, ein externes
Zeitlimit greift (CartPole-v1: 500 Steps). Beide Felder zusammen ersetzen das
frühere ``done``-Flag aus OpenAI Gym.

Der Unterschied zwischen On- und Off-Policy-Algorithmen ist der strukturelle Kern
von Kapitel 5 und Abschnitt 7.8 des Skripts. On-Policy-Algorithmen (A2C, PPO,
REINFORCE/Algorithmus 32) aktualisieren die Policy ausschließlich auf Basis von
Rollouts, die mit der aktuellen Policy gesammelt wurden. Nach jedem Update werden
diese Daten verworfen, da sie für die neue Policy nicht mehr gültig sind. Off-Policy-
Algorithmen (DQN, DDPG, SAC, TD3) speichern Erfahrungen in einem Replay-Buffer und
können jeden Datenpunkt mehrfach für das Update nutzen, unabhängig davon, mit welcher
Policy er gesammelt wurde. Das macht sie sample-effizienter, aber in der Regel
schwieriger zu stabilisieren.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import X_TIMESTEPS, load_results, ts2xy

ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = ROOT / "figures"
LOGS_BASE = ROOT / "logs" / "sheet09"

ENV_ID = "CartPole-v1"

LECTURE_NOTES: str = """
(a) OnPolicyAlgorithm in SB3 und Algorithmus 32 (Skript S. 168)

Stable Baselines3's OnPolicyAlgorithm und Algorithmus 32 (REINFORCE / Batch-
Stochastic Policy Gradient) teilen dieselbe Grundstruktur: beide sammeln einen
kompletten Rollout unter der aktuellen Policy, berechnen daraus Gradienten-
Schätzungen und führen dann einen Gradient-Update durch. Der Rollout-Buffer in
SB3 entspricht dem Datenpuffer D_k in Algorithmus 32. Der entscheidende Unterschied
liegt im Return-Schätzer: reines REINFORCE (Algorithmus 32) verwendet Monte-Carlo-
Returns G_t bis zum Episodenende als unverzerrte, aber hochvariante Schätzung von
Q^pi(s_t, a_t). A2C (Section 7.8.1) ergänzt ein Critic-Netzwerk, das V^pi(s)
schätzt, und ersetzt G_t durch n-step-Bootstrapping minus Baseline (Advantage
A_t = r_t + gamma*V(s_{t+1}) - V(s_t)). Das reduziert die Varianz erheblich,
erkauft aber einen Bias durch die Bootstrapping-Approximation.

(b) Warum ist DQN OffPolicy?

DQN lernt eine Q-Funktion Q(s, a; theta) über Bellman-Iterationen, analog zu
tabellarischem Q-Learning (Skript Section 3.5). Die Zielwerte y_t = r_t + gamma *
max_{a'} Q(s_{t+1}, a'; theta^-) hängen nicht von der aktuellen Policy ab, sondern
nur von der Greedy-Projektion auf Q. Damit können beliebige (s, a, r, s')-Tupel im
Replay-Buffer genutzt werden, egal unter welcher Verhaltens-Policy sie entstanden.
Der Replay-Buffer entkoppelt Datensammlung von Training, reduziert Korrelationen
zwischen aufeinanderfolgenden Samples und erhöht die Sample-Effizienz erheblich.
Nachteil: Off-Policy-Methoden konvergieren bei tiefen Netzen schwieriger, da
Zielwerte und Policy sich gegenseitig beeinflussen (Deadly Triad).

(c) SB3-Zoo-Struktur

rl-zoo3 organisiert Hyperparameter in YAML-Dateien unter
rl_zoo3/hyperparams/<algo>.yml, eine Datei pro Algorithmus. Jede YAML-Datei
enthält umgebungs-spezifische Hyperparameter-Blöcke (z.B. CartPole-v1:,
Pendulum-v1:), die von rl_zoo3.train.train() beim Aufruf via CLI eingelesen
werden. Das einheitliche CLI-Entry
    python -m rl_zoo3.train --algo a2c --env CartPole-v1
lädt die passenden Hyperparameter automatisch. Unsere eigene Mini-batch-REINFORCE-
Klasse soll dieser Konvention folgen: ein eigenes hyperparams/mini_batch_reinforce.yml
mit Einträgen pro Classic-Control-Env macht sie Zoo-kompatibel und ermöglicht den
direkten Vergleich mit A2C, PPO et al. auf demselben Infrastruktur-Fundament.
"""


def _set_seeds(seed: int) -> None:
    """Setzt numpy-, torch- und Python-Random-Seeds für Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def gymnasium_api_tour(seed: int) -> None:
    """Sektion A: Gymnasium-API-Demonstration mit zehn zufälligen Steps auf CartPole-v1."""
    env = gym.make(ENV_ID)

    print(f"\n{'=' * 60}")
    print("Sektion A — Gymnasium-API-Tour")
    print(f"{'=' * 60}")
    print(f"Umgebung:          {ENV_ID}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print()

    obs, info = env.reset(seed=seed)

    header = f"{'Step':>4}  {'Reward':>6}  {'Term':>5}  {'Trunc':>5}  Obs[0:2]"
    print(header)
    print("-" * 56)

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_str = f"[{obs[0]:+.3f}, {obs[1]:+.3f}, ...]"
        print(
            f"{step + 1:>4}  {reward:>6.2f}  {str(terminated):>5}  {str(truncated):>5}  {obs_str}"
        )
        if terminated or truncated:
            obs, info = env.reset(seed=seed + step + 1)
            print("       -> Episode beendet, Reset (neue Episode)")

    env.close()
    print()


def _train_algo(
    algo_cls: type,
    algo_name: str,
    steps: int,
    seed: int,
    device: str,
) -> tuple[str, float, float]:
    """Trainiert einen SB3-Algorithmus mit Monitor-Wrapper und evaluiert ihn.

    Logs landen in logs/sheet09/<algo_name>/monitor.monitor.csv.
    Gibt (log_dir, mean_return, std_return) zurueck.
    """
    log_dir = str(LOGS_BASE / algo_name)
    os.makedirs(log_dir, exist_ok=True)

    monitor_path = os.path.join(log_dir, "monitor")
    train_env = Monitor(gym.make(ENV_ID), monitor_path)

    _set_seeds(seed)
    model = algo_cls("MlpPolicy", train_env, device=device, seed=seed, verbose=0)
    model.learn(total_timesteps=steps)
    train_env.close()

    eval_env = gym.make(ENV_ID)
    mean_ret, std_ret = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True, warn=False
    )
    eval_env.close()

    return log_dir, float(mean_ret), float(std_ret)


def _rolling_mean(y: np.ndarray, window: int = 50) -> np.ndarray:
    """Gleitender Durchschnitt ueber window Episoden."""
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_learning_curves(
    log_dirs: dict[str, str],
    steps: int,
    seed: int,
    device: str,
) -> Path:
    """Sektion C: Zwei-Panel-Plot der Lernkurven aus Monitor-CSVs.

    Speichert unter figures/sheet09_a2c_vs_dqn_cartpole.png.
    """
    labels = {"a2c": "A2C (OnPolicy)", "dqn": "DQN (OffPolicy)"}
    colors = {"a2c": "tab:blue", "dqn": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"A2C vs. DQN auf {ENV_ID}  |  steps={steps}  seed={seed}  device={device}",
        fontsize=12,
    )

    for ax, (algo_key, log_dir) in zip(axes, log_dirs.items()):
        df = load_results(log_dir)
        x, y = ts2xy(df, X_TIMESTEPS)
        y_smooth = _rolling_mean(y, window=50)

        label = labels[algo_key]
        color = colors[algo_key]
        ax.plot(x, y, alpha=0.25, color=color, linewidth=0.8, label="Episode Return")
        ax.plot(x, y_smooth, color=color, linewidth=2.0, label="Rolling Mean (50 Ep.)")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Return")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "sheet09_a2c_vs_dqn_cartpole.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blatt 9 Aufg. 3 — Gymnasium-API-Tour + A2C vs. DQN auf CartPole-v1"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduzierter Lauf: 5_000 Steps statt --steps",
    )
    parser.add_argument(
        "--steps", type=int, default=50_000, help="Trainings-Steps (Default: 50_000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random-Seed (Default: 42)")
    parser.add_argument(
        "--device", type=str, default="auto", help='PyTorch-Device (Default: "auto")'
    )
    args = parser.parse_args()

    steps: int = 5_000 if args.quick else args.steps
    seed: int = args.seed
    device: str = args.device

    if device == "auto":
        resolved = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        resolved = device

    _set_seeds(seed)

    gymnasium_api_tour(seed)

    print(f"{'=' * 60}")
    print("Sektion B — Training A2C (OnPolicy) und DQN (OffPolicy)")
    print(f"{'=' * 60}")
    print(f"Steps: {steps}  Seed: {seed}  Device (effektiv): {resolved}")
    print()

    log_dirs: dict[str, str] = {}

    print("Trainiere A2C ...")
    log_dir_a2c, mean_a2c, std_a2c = _train_algo(A2C, "a2c", steps, seed, device)
    log_dirs["a2c"] = log_dir_a2c
    print(f"A2C  Eval (10 Ep.): {mean_a2c:.1f} +/- {std_a2c:.1f}")
    print()

    print("Trainiere DQN ...")
    log_dir_dqn, mean_dqn, std_dqn = _train_algo(DQN, "dqn", steps, seed, device)
    log_dirs["dqn"] = log_dir_dqn
    print(f"DQN  Eval (10 Ep.): {mean_dqn:.1f} +/- {std_dqn:.1f}")
    print()

    print(f"{'=' * 60}")
    print("Sektion C — Lernkurven-Plot")
    print(f"{'=' * 60}")
    out_path = plot_learning_curves(log_dirs, steps, seed, device)
    print(f"Plot gespeichert: {out_path}")
    print()
    print(LECTURE_NOTES)


if __name__ == "__main__":
    main()
