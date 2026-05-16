"""Gemeinsame Trainings- und Plot-Logik für die REINFORCE-Demo-Skripte (Blatt 10 Aufg. 7).

Beide Demo-Skripte (CartPole, Acrobot) rufen run_demo() mit env-spezifischen
Pfadkonstanten auf. Die gesamte Trainings- und Plot-Logik liegt hier.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import X_TIMESTEPS, load_results, ts2xy

from actor_critic_project.algos.mini_batch_reinforce import MiniBatchREINFORCE
from actor_critic_project.utils.plotting import plot_mean_std_band, rolling_mean, save_figure
from actor_critic_project.utils.seeds import set_global_seeds

ROOT = Path(__file__).resolve().parents[3]
HYPERPARAMS_PATH = (
    ROOT / "actor_critic_project" / "algos" / "mini_batch_reinforce" / "hyperparams.yml"
)

# YAML-Keys, die keine MiniBatchREINFORCE-Konstruktor-Kwargs sind
_YAML_META_KEYS = {"n_envs", "n_timesteps", "policy"}


def build_arg_parser(env_id: str) -> argparse.ArgumentParser:
    """Erstellt den CLI-Parser für ein Demo-Skript."""
    parser = argparse.ArgumentParser(
        description=f"Mini-batch REINFORCE Demo auf {env_id} (Blatt 10 Aufg. 7)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        metavar="N",
        help="Anzahl Seeds (Default: 5)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        metavar="INT",
        help="Trainings-Timesteps pro Seed (Default: aus YAML, Fallback 100_000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Schnellmodus: 2 Seeds, 5_000 Timesteps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='PyTorch-Device (Default: "auto")',
    )
    return parser


def _load_hyperparams(env_id: str) -> dict:
    """Lädt den Hyperparameter-Block für env_id aus hyperparams.yml."""
    with open(HYPERPARAMS_PATH) as f:
        all_params = yaml.safe_load(f)
    if env_id not in all_params:
        raise KeyError(f"Kein Hyperparameter-Block fuer '{env_id}' in {HYPERPARAMS_PATH}")
    return dict(all_params[env_id])


def _train_one_seed(
    env_id: str,
    seed: int,
    total_timesteps: int,
    algo_kwargs: dict,
    log_dir: Path,
    results_dir: Path,
    device: str,
) -> tuple[float, float, float]:
    """Trainiert MiniBatchREINFORCE fuer einen Seed.

    Gibt (eval_mean, eval_std, wallclock_sekunden) zurueck.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(seed)
    train_env = Monitor(gym.make(env_id), str(log_dir / "monitor"))

    model = MiniBatchREINFORCE(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        device=device,
        verbose=0,
        **algo_kwargs,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    wallclock = time.perf_counter() - t0

    train_env.close()

    eval_env = gym.make(env_id)
    mean_ret, std_ret = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=False, warn=False
    )
    eval_env.close()

    model.save(str(results_dir / f"seed_{seed}"))

    print(f"  Seed {seed}: Eval {mean_ret:.1f} +/- {std_ret:.1f}  ({wallclock:.1f}s)")
    return float(mean_ret), float(std_ret), float(wallclock)


def _load_seed_curve(
    log_dir: Path,
    total_timesteps: int,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Laedt Monitor-CSV und gibt (x_common, returns_interp, lengths_interp) zurueck.

    Interpoliert auf ein gemeinsames Timestep-Raster fuer seed-uebergreifende Mittelung.
    """
    df = load_results(str(log_dir))
    x_ep, y_ret = ts2xy(df, X_TIMESTEPS)
    y_len = df["l"].to_numpy()[: len(x_ep)]

    x_common = np.linspace(0, total_timesteps, n_points)

    if len(x_ep) >= 2:
        y_ret_smooth = rolling_mean(y_ret, window=20)
        y_len_smooth = rolling_mean(y_len, window=20)
        ret_interp = np.interp(x_common, x_ep, y_ret_smooth)
        len_interp = np.interp(x_common, x_ep, y_len_smooth)
    else:
        fallback_ret = float(y_ret[0]) if len(y_ret) > 0 else 0.0
        fallback_len = float(y_len[0]) if len(y_len) > 0 else 0.0
        ret_interp = np.full(n_points, fallback_ret)
        len_interp = np.full(n_points, fallback_len)

    return x_common, ret_interp, len_interp


def _make_plot(
    env_id: str,
    seeds: Sequence[int],
    logs_base: Path,
    total_timesteps: int,
    eval_means: list[float],
    device: str,
    out_path: Path,
) -> None:
    """Erstellt den 3-Panel-Plot: Episode Return, Episode Laenge, Finale Eval-Bar."""
    n_points = 200
    all_returns = []
    all_lengths = []
    x_common = None

    for seed in seeds:
        x_common, ret_interp, len_interp = _load_seed_curve(
            logs_base / f"seed_{seed}", total_timesteps, n_points
        )
        all_returns.append(ret_interp)
        all_lengths.append(len_interp)

    returns_arr = np.array(all_returns)   # (n_seeds, n_points)
    lengths_arr = np.array(all_lengths)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Mini-batch REINFORCE auf {env_id}  |  {len(seeds)} Seeds  "
        f"total_timesteps={total_timesteps}  device={device}",
        fontsize=11,
    )

    # Subplot 1: Episode Return
    ax = axes[0]
    for seed in seeds:
        df = load_results(str(logs_base / f"seed_{seed}"))
        x_ep, y_ret = ts2xy(df, X_TIMESTEPS)
        ax.plot(x_ep, y_ret, alpha=0.2, color="tab:blue", linewidth=0.8)
    plot_mean_std_band(ax, x_common, returns_arr, label="Mean +/- Std (20 Ep. Rolling)", color="tab:blue")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.set_title("Episode Return ueber Zeit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Subplot 2: Episode Laenge
    ax = axes[1]
    for seed in seeds:
        df = load_results(str(logs_base / f"seed_{seed}"))
        x_ep, _ = ts2xy(df, X_TIMESTEPS)
        y_len = df["l"].to_numpy()[: len(x_ep)]
        ax.plot(x_ep, y_len, alpha=0.2, color="tab:orange", linewidth=0.8)
    plot_mean_std_band(ax, x_common, lengths_arr, label="Mean +/- Std (20 Ep. Rolling)", color="tab:orange")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Laenge (Steps)")
    ax.set_title("Episode Laenge ueber Zeit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Subplot 3: Finale Eval-Returns als Bar-Chart
    ax = axes[2]
    x_bars = np.arange(len(seeds))
    ax.bar(x_bars, eval_means, color="tab:green", width=0.5)
    grand_mean = float(np.mean(eval_means))
    ax.axhline(
        grand_mean,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Mittel = {grand_mean:.1f}",
    )
    ax.set_xticks(x_bars)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], rotation=30, ha="right")
    ax.set_ylabel("Finale Eval-Return (20 Ep.)")
    ax.set_title("Finale Eval-Performance pro Seed")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, out_path)
    print(f"Plot gespeichert: {out_path}")


def run_demo(
    env_id: str,
    log_subdir: str,
    plot_filename: str,
    summary_subpath: str,
    args: argparse.Namespace,
) -> None:
    """Vollstaendiger Demo-Lauf: Training ueber Seeds, Eval, Plot, JSON-Summary."""
    raw_params = _load_hyperparams(env_id)

    if args.quick:
        n_seeds = 2
        total_timesteps = 5_000
    else:
        n_seeds = args.seeds
        total_timesteps = args.total_timesteps or int(raw_params.get("n_timesteps", 100_000))

    device: str = args.device
    algo_kwargs = {k: v for k, v in raw_params.items() if k not in _YAML_META_KEYS}

    logs_base = ROOT / "logs" / log_subdir
    results_dir = ROOT / "results" / log_subdir
    out_plot = ROOT / "figures" / plot_filename
    summary_path = ROOT / "results" / summary_subpath

    seeds = list(range(n_seeds))
    eval_means: list[float] = []
    eval_stds: list[float] = []
    wallclocks: list[float] = []

    print(f"\n{'=' * 60}")
    print(f"Mini-batch REINFORCE auf {env_id}")
    print(f"Seeds: {seeds}  |  Timesteps: {total_timesteps}  |  Device: {device}")
    print(f"{'=' * 60}")

    for seed in seeds:
        mean_ret, std_ret, wc = _train_one_seed(
            env_id=env_id,
            seed=seed,
            total_timesteps=total_timesteps,
            algo_kwargs=algo_kwargs,
            log_dir=logs_base / f"seed_{seed}",
            results_dir=results_dir,
            device=device,
        )
        eval_means.append(mean_ret)
        eval_stds.append(std_ret)
        wallclocks.append(wc)

    print(f"\nGesamt: Eval-Mean = {np.mean(eval_means):.1f} +/- {np.std(eval_means):.1f}")

    _make_plot(
        env_id=env_id,
        seeds=seeds,
        logs_base=logs_base,
        total_timesteps=total_timesteps,
        eval_means=eval_means,
        device=device,
        out_path=out_plot,
    )

    summary = {
        "env_id": env_id,
        "n_seeds": n_seeds,
        "total_timesteps": total_timesteps,
        "device": device,
        "hyperparams": algo_kwargs,
        "results": {
            f"seed_{s}": {
                "eval_mean": m,
                "eval_std": d,
                "wallclock_s": w,
            }
            for s, m, d, w in zip(seeds, eval_means, eval_stds, wallclocks)
        },
        "aggregate": {
            "mean": float(np.mean(eval_means)),
            "std": float(np.std(eval_means)),
            "total_wallclock_s": float(sum(wallclocks)),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary gespeichert: {summary_path}")
