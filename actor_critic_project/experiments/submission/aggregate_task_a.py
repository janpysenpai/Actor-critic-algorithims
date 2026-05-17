"""Aggregation der Sweep-Ergebnisse fuer Blatt 11, Aufgabe 4 (a).

Liest runs.csv und Monitor-CSVs, berechnet Metriken und schreibt:
- summary_table.csv
- summary_table.md

Aufruf:
    python -m actor_critic_project.experiments.submission.aggregate_task_a \\
        [--input-dir results/submission/task_a] \\
        [--output-dir results/submission/task_a]
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from actor_critic_project.utils.metrics import (
    auc_return_per_step,
    final_mean_return,
    load_monitor_csv,
    sample_efficiency_to_threshold,
    wallclock_per_1k_steps,
)


def _load_runs(input_dir: pathlib.Path) -> pd.DataFrame:
    csv_path = input_dir / "runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"runs.csv nicht gefunden: {csv_path}")
    return pd.read_csv(csv_path)


def _compute_group_metrics(group_df: pd.DataFrame, total_timesteps: int) -> dict:
    """Berechnet aggregierte Metriken fuer eine (algo, env_id)-Gruppe."""
    mean_eval = group_df["eval_mean"].mean()
    std_eval = group_df["eval_mean"].std(ddof=0)
    mean_wallclock = group_df["wallclock_s"].mean()
    n_seeds = len(group_df)
    wallclock_1k = wallclock_per_1k_steps(mean_wallclock, total_timesteps)

    # Monitor-CSVs laden fuer Trainingsmetriken
    seed_dfs: list[pd.DataFrame] = []
    for _, row in group_df.iterrows():
        csv_path_str = str(row.get("monitor_csv", ""))
        if not csv_path_str:
            continue
        csv_path = pathlib.Path(csv_path_str)
        if not csv_path.exists():
            continue
        try:
            df = load_monitor_csv(csv_path)
            seed_dfs.append(df)
        except Exception:
            pass

    final_ret = float("nan")
    auc = float("nan")
    sample_eff: Optional[float] = None

    if seed_dfs:
        final_ret = float(
            np.mean([final_mean_return(df["r"].tolist(), last_k=100) for df in seed_dfs])
        )
        auc = float(np.mean([auc_return_per_step(df) for df in seed_dfs]))

        # Sample-Efficiency: Steps bis 80% des eigenen Rolling-Mean-Maximums (pro Seed)
        efficiencies: list[int] = []
        for df in seed_dfs:
            rolling = df["r"].rolling(window=100, min_periods=1).mean()
            max_rolling = rolling.max()
            if max_rolling > 0:
                threshold = 0.8 * max_rolling
                e = sample_efficiency_to_threshold(df, threshold)
                if e is not None:
                    efficiencies.append(e)
        if efficiencies:
            sample_eff = float(np.mean(efficiencies))

    return {
        "mean_eval_return": round(mean_eval, 3),
        "std_eval_return": round(std_eval, 3),
        "mean_wallclock_s": round(mean_wallclock, 2),
        "wallclock_per_1k_steps": round(wallclock_1k, 4),
        "n_seeds": n_seeds,
        "final_mean_return": round(final_ret, 3) if not np.isnan(final_ret) else None,
        "sample_efficiency": int(sample_eff) if sample_eff is not None else None,
        "auc_per_step": round(auc, 4) if not np.isnan(auc) else None,
    }


def _to_markdown(summary_df: pd.DataFrame) -> str:
    """Wandelt den Summary-DataFrame in eine Markdown-Tabelle um."""
    lines = ["| " + " | ".join(summary_df.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(summary_df.columns)) + "|")
    for _, row in summary_df.iterrows():
        lines.append("| " + " | ".join(str(v) if v is not None else "-" for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregation der Task-(a)-Sweep-Ergebnisse"
    )
    parser.add_argument("--input-dir", default="results/submission/task_a")
    parser.add_argument("--output-dir", default=None, help="Default: wie --input-dir")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = _load_runs(input_dir)
    total_timesteps = int(runs_df["total_timesteps"].iloc[0]) if len(runs_df) > 0 else 100_000

    rows = []
    for (algo, env_id), group in runs_df.groupby(["algo", "env_id"]):
        metrics = _compute_group_metrics(group, total_timesteps)
        rows.append({"algo": algo, "env_id": env_id, **metrics})

    summary_df = pd.DataFrame(rows)

    # Sortieren: env_id aufsteigend, mean_eval_return absteigend
    summary_df = summary_df.sort_values(
        ["env_id", "mean_eval_return"], ascending=[True, False]
    ).reset_index(drop=True)

    csv_path = output_dir / "summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary-CSV: {csv_path}")

    md_path = output_dir / "summary_table.md"
    md_path.write_text(_to_markdown(summary_df), encoding="utf-8")
    print(f"Summary-Markdown: {md_path}")


if __name__ == "__main__":
    main()
