"""Plot-Skript fuer Blatt 11, Aufgabe 4 (a): Lernkurven und Balkendiagramme.

Erzeugt in figures/submission/task_a/:
- learning_curves_<env_id>.png  — Rolling-Mean-Lernkurve pro Algo, gemittelt ueber Seeds
- final_return_bar_<env_id>.png — Balkendiagramm des mittleren Eval-Returns
- combined_overview.png         — 5-Subplot-Uebersicht aller Envs

Aufruf:
    python -m actor_critic_project.experiments.submission.plot_task_a \\
        [--input-dir results/submission/task_a] \\
        [--figure-dir figures/submission/task_a]
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from actor_critic_project.utils.metrics import load_monitor_csv
from actor_critic_project.utils.plotting import (
    plot_algo_comparison,
    plot_metric_bar,
    save_figure,
)


def _load_monitor_dfs(
    runs_df: pd.DataFrame,
) -> dict[str, dict[str, list[pd.DataFrame]]]:
    """Laedt Monitor-DataFrames gruppiert nach env_id → algo → [seed-DFs]."""
    result: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for _, row in runs_df.iterrows():
        env_id = str(row["env_id"])
        algo = str(row["algo"])
        csv_path_str = str(row.get("monitor_csv", ""))
        if not csv_path_str:
            continue
        csv_path = pathlib.Path(csv_path_str)
        if not csv_path.exists():
            continue
        try:
            df = load_monitor_csv(csv_path)
        except Exception:
            continue
        result.setdefault(env_id, {}).setdefault(algo, []).append(df)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plots fuer Task-(a)-Sweep-Ergebnisse"
    )
    parser.add_argument("--input-dir", default="results/submission/task_a")
    parser.add_argument("--figure-dir", default="figures/submission/task_a")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir).resolve()
    figure_dir = pathlib.Path(args.figure_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / "runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"runs.csv nicht gefunden: {csv_path}")
    runs_df = pd.read_csv(csv_path)

    env_ids = sorted(runs_df["env_id"].unique())
    per_env_monitor = _load_monitor_dfs(runs_df)

    # a) Lernkurven pro Env
    for env_id in env_ids:
        per_algo_dfs = per_env_monitor.get(env_id, {})
        if not per_algo_dfs:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_algo_comparison(per_algo_dfs, env_id=env_id, ax=ax)
        save_figure(fig, figure_dir / f"learning_curves_{env_id}.png", dpi=300)

    # b) Balkendiagramm final-Eval-Return pro Env
    for env_id in env_ids:
        env_rows = runs_df[runs_df["env_id"] == env_id]
        per_algo_metric: dict[str, list[float]] = {}
        for algo, grp in env_rows.groupby("algo"):
            per_algo_metric[str(algo)] = grp["eval_mean"].tolist()
        if not per_algo_metric:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_metric_bar(per_algo_metric, metric_name="Mean Eval Return", ax=ax)
        ax.set_title(f"Mittlerer Eval-Return: {env_id}")
        fig.tight_layout()
        save_figure(fig, figure_dir / f"final_return_bar_{env_id}.png", dpi=300)

    # c) Kombinierte Uebersicht (5 Subplots, einer pro Env)
    n_envs = len(env_ids)
    if n_envs > 0:
        fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4), sharey=False)
        if n_envs == 1:
            axes = [axes]
        for ax, env_id in zip(axes, env_ids):
            per_algo_dfs = per_env_monitor.get(env_id, {})
            if per_algo_dfs:
                plot_algo_comparison(per_algo_dfs, env_id=env_id, ax=ax)
            else:
                ax.set_title(env_id)
                ax.text(0.5, 0.5, "keine Daten", ha="center", va="center",
                        transform=ax.transAxes)
        fig.suptitle("Algorithmusvergleich — Classic Control (Blatt 11, Aufg. 4)")
        fig.tight_layout()
        save_figure(fig, figure_dir / "combined_overview.png", dpi=300)

    print(f"Plots gespeichert in: {figure_dir}")


if __name__ == "__main__":
    main()
