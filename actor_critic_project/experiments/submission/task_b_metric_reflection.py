"""Metric-Reflexion Tabular vs. Deep RL — Blatt 11, Aufgabe 4 (b).

Laedt Ergebnisse aus Task (a) und Tabular-Baselines, berechnet normalisierte
Metriken und erzeugt Plots + einen Roh-Report (reflection.md) mit TODO-Markierungen
fuer die Prosa-Reflexion.

Aufruf (Quick-Test):
    python -m actor_critic_project.experiments.submission.task_b_metric_reflection \\
        --quick --regenerate-tabular --task-a-dir <dir>

Aufruf (voll):
    python -m actor_critic_project.experiments.submission.task_b_metric_reflection \\
        --task-a-dir results/submission/task_a
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from actor_critic_project.utils.metrics import (
    auc_return_per_step,
    final_mean_return,
    load_monitor_csv,
    sample_efficiency_to_threshold,
    wallclock_per_1k_steps,
)
from actor_critic_project.utils.normalization import (
    normalize_return,
    threshold_from_normalized,
)

from .tabular_baselines import TABULAR_ENVS, run_and_save_all

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_QUICK_TABULAR_EPISODES = 200
_FULL_TABULAR_EPISODES = 2000

# ---------------------------------------------------------------------------
# Tabular-Daten laden / generieren
# ---------------------------------------------------------------------------


def _load_or_generate_tabular(
    tabular_dir: pathlib.Path,
    regenerate: bool,
    quick: bool,
) -> pd.DataFrame:
    tabular_csv = tabular_dir / "runs.csv"
    if regenerate or not tabular_csv.exists():
        n_ep = _QUICK_TABULAR_EPISODES if quick else _FULL_TABULAR_EPISODES
        seeds = (0,) if quick else (0, 1, 2)
        print(f"Generiere Tabular-Runs: {n_ep} Episoden, Seeds={seeds} → {tabular_dir}")
        run_and_save_all(
            output_dir=tabular_dir,
            algos=("q_learning", "sarsa"),
            env_ids=TABULAR_ENVS,
            seeds=seeds,
            n_episodes=n_ep,
        )
    return pd.read_csv(tabular_csv)


# ---------------------------------------------------------------------------
# Metriken fuer einen Datensatz berechnen
# ---------------------------------------------------------------------------


def _metrics_from_monitor(
    algo: str,
    env_id: str,
    seed: int,
    monitor_csv_path: pathlib.Path,
    wallclock_s: float,
    total_steps: int,
    family: str,
) -> dict:
    norm_threshold = threshold_from_normalized(env_id, 0.7)

    try:
        df = load_monitor_csv(monitor_csv_path)
        fm_ret = final_mean_return(df["r"].tolist(), last_k=100)
        auc = auc_return_per_step(df)
        se_steps = sample_efficiency_to_threshold(df, norm_threshold)
    except Exception:
        fm_ret = float("nan")
        auc = float("nan")
        se_steps = None

    wc_1k = wallclock_per_1k_steps(wallclock_s, max(total_steps, 1))
    se_norm = (
        (1.0 - se_steps / max(total_steps, 1)) if se_steps is not None else 0.0
    )

    return {
        "family": family,
        "algo": algo,
        "env_id": env_id,
        "seed": seed,
        "final_mean_return_raw": fm_ret,
        "auc_per_step_raw": auc,
        "sample_efficiency_steps": se_steps,
        "wallclock_per_1k_raw": wc_1k,
        "final_mean_return_norm": normalize_return(env_id, fm_ret) if not np.isnan(fm_ret) else float("nan"),
        "auc_per_step_norm": normalize_return(env_id, auc) if not np.isnan(auc) else float("nan"),
        "sample_efficiency_norm": se_norm,
    }


def _compute_tabular_metrics(
    tabular_df: pd.DataFrame, tabular_dir: pathlib.Path
) -> list[dict]:
    rows: list[dict] = []
    for (algo, env_id, seed), group in tabular_df.groupby(["algo", "env_id", "seed"]):
        wallclock_s = float(group["wallclock_s"].iloc[0])
        total_steps = int(group["total_steps"].iloc[0])
        monitor_csv = tabular_dir / f"{algo}_{env_id}_seed{seed}.monitor.csv"
        rows.append(
            _metrics_from_monitor(
                str(algo), str(env_id), int(seed),
                monitor_csv, wallclock_s, total_steps, "tabular",
            )
        )
    return rows


def _compute_deep_metrics(deep_csv: pathlib.Path) -> list[dict]:
    deep_df = pd.read_csv(deep_csv)
    rows: list[dict] = []
    for _, row in deep_df.iterrows():
        monitor_path_str = str(row.get("monitor_csv", ""))
        if not monitor_path_str:
            continue
        monitor_path = pathlib.Path(monitor_path_str)
        if not monitor_path.exists():
            continue
        rows.append(
            _metrics_from_monitor(
                str(row["algo"]), str(row["env_id"]), int(row["seed"]),
                monitor_path,
                float(row["wallclock_s"]),
                int(row["total_timesteps"]),
                "deep",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    ("final_mean_return_norm", "Final Mean Return (norm.)"),
    ("auc_per_step_norm", "AUC per Step (norm.)"),
    ("sample_efficiency_norm", "Sample Efficiency (norm., hoeher = schneller)"),
    ("wallclock_per_1k_raw", "Wallclock / 1k Steps (s)"),
    ("return_std", "Return-Std ueber Env×Seed"),
]


def _std_across_env_seed(df: pd.DataFrame, col: str, family: str) -> pd.Series:
    """Std-Spalte: Standardabweichung des normalisierten Returns ueber Seeds pro Algo."""
    fam = df[df["family"] == family].copy()
    if fam.empty:
        return pd.Series(dtype=float)
    return fam.groupby("algo")[col].std(ddof=0).fillna(0.0)


def _plot_metric_grid(metrics_df: pd.DataFrame, figure_dir: pathlib.Path) -> None:
    families = metrics_df["family"].unique()
    colors = {"tabular": "steelblue", "deep": "darkorange"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    # std ueber Seeds pro Algo berechnen und in df einbauen
    std_col = "return_std"
    metrics_df = metrics_df.copy()
    std_vals = metrics_df.groupby(["family", "algo"])["final_mean_return_norm"].transform("std").fillna(0.0)
    metrics_df[std_col] = std_vals

    for idx, (col, title) in enumerate(_METRIC_COLS):
        ax = axes_flat[idx]
        if col not in metrics_df.columns:
            ax.set_visible(False)
            continue

        for fam in families:
            sub = metrics_df[metrics_df["family"] == fam].copy()
            if sub.empty:
                continue
            grouped = sub.groupby("algo")[col]
            means = grouped.mean()
            stds = grouped.std(ddof=0).fillna(0.0)
            x = np.arange(len(means))
            offset = -0.2 if fam == "tabular" else 0.2
            ax.bar(
                x + offset, means.values, 0.35, yerr=stds.values,
                label=fam, color=colors.get(fam, "gray"), capsize=3, alpha=0.85,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(means.index, rotation=30, ha="right", fontsize=7)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Algorithmus", fontsize=8)
        ax.set_ylabel("Wert", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8)

    axes_flat[-1].set_visible(False)
    fig.suptitle("Metrik-Vergleich: Tabular vs. Deep RL (normalisiert)", fontsize=12)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "metric_grid.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {path}")


def _plot_scatter(metrics_df: pd.DataFrame, figure_dir: pathlib.Path) -> None:
    colors = {"tabular": "steelblue", "deep": "darkorange"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for fam in metrics_df["family"].unique():
        sub = metrics_df[metrics_df["family"] == fam]
        x = sub["final_mean_return_norm"].to_numpy(dtype=float)
        y = sub["auc_per_step_norm"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(
            x[mask], y[mask], label=fam, color=colors.get(fam, "gray"),
            alpha=0.7, s=40,
        )
    ax.set_xlabel("Final Mean Return (normalisiert)", fontsize=10)
    ax.set_ylabel("AUC per Step (normalisiert)", fontsize=10)
    ax.set_title("Cross-World-Vergleich: Final Return vs. AUC", fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    path = figure_dir / "cross_world_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {path}")


def _plot_wallclock(metrics_df: pd.DataFrame, figure_dir: pathlib.Path) -> None:
    colors = {"tabular": "steelblue", "deep": "darkorange"}
    fig, ax = plt.subplots(figsize=(9, 5))

    all_algos = sorted(metrics_df["algo"].unique())
    x = np.arange(len(all_algos))
    families = sorted(metrics_df["family"].unique())
    n_fam = len(families)
    width = 0.35

    for fi, fam in enumerate(families):
        sub = metrics_df[metrics_df["family"] == fam]
        means_by_algo = sub.groupby("algo")["wallclock_per_1k_raw"].mean()
        vals = np.array([means_by_algo.get(a, float("nan")) for a in all_algos])
        offset = (fi - n_fam / 2 + 0.5) * width
        valid = np.isfinite(vals)
        ax.bar(
            x[valid] + offset, vals[valid], width,
            label=fam, color=colors.get(fam, "gray"), alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_algos, rotation=30, ha="right")
    ax.set_ylabel("Wallclock / 1000 Schritte (s, log-Skala)", fontsize=9)
    ax.set_title("Wanduhrzeit-Vergleich: Tabular vs. Deep RL\n(Wallclock ist zwischen den Welten nicht direkt vergleichbar)", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = figure_dir / "wallclock_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {path}")


# ---------------------------------------------------------------------------
# Reflection-Report
# ---------------------------------------------------------------------------


def _format_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = [
        "| " + " | ".join(str(v) if not (isinstance(v, float) and np.isnan(v)) else "-" for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


def _get_git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _generate_reflection(
    metrics_df: pd.DataFrame,
    output_dir: pathlib.Path,
    tabular_source: str,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    sha = _get_git_sha()

    n_tabular = metrics_df[metrics_df["family"] == "tabular"].shape[0]
    n_deep = metrics_df[metrics_df["family"] == "deep"].shape[0]
    tabular_algos = sorted(metrics_df[metrics_df["family"] == "tabular"]["algo"].unique())
    tabular_envs = sorted(metrics_df[metrics_df["family"] == "tabular"]["env_id"].unique())
    deep_algos = sorted(metrics_df[metrics_df["family"] == "deep"]["algo"].unique())
    deep_envs = sorted(metrics_df[metrics_df["family"] == "deep"]["env_id"].unique())

    # Uebersichtstabelle: mean +/- std pro (family, metric)
    norm_metrics = [
        ("final_mean_return_norm", "Final Mean Return (norm.)"),
        ("auc_per_step_norm", "AUC per Step (norm.)"),
        ("sample_efficiency_norm", "Sample Efficiency (norm.)"),
        ("wallclock_per_1k_raw", "Wallclock / 1k Steps (s)"),
    ]
    overview_rows = []
    for fam in ["tabular", "deep"]:
        sub = metrics_df[metrics_df["family"] == fam]
        for col, label in norm_metrics:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if vals.empty:
                continue
            overview_rows.append({
                "Family": fam,
                "Metric": label,
                "Mean": f"{vals.mean():.3f}",
                "Std": f"{vals.std(ddof=0):.3f}",
                "N": len(vals),
            })
    overview_df = pd.DataFrame(overview_rows)

    # Per-Metric-Bloecke
    def _metric_block(col: str, label: str) -> str:
        lines = [f"### {label}", ""]
        for fam in ["tabular", "deep"]:
            sub = metrics_df[metrics_df["family"] == fam]
            if col not in sub.columns:
                continue
            grp = sub.groupby(["algo", "env_id"])[col].mean().reset_index()
            grp.columns = ["Algorithmus", "Umgebung", "Mean"]
            grp["Mean"] = grp["Mean"].apply(lambda v: f"{v:.3f}" if not np.isnan(v) else "-")
            lines.append(f"**{fam}:**")
            lines.append("")
            lines.append(_format_table(grp))
            lines.append("")
        lines.append("**Sinnvoll in beiden Welten?** [TODO: User-Text]")
        lines.append("")
        return "\n".join(lines)

    metric_sections = "\n".join([
        f"## 3. Per-Metric Reflexion\n",
        f"### 3.1 final_mean_return\n",
        _metric_block("final_mean_return_norm", "Final Mean Return (normalisiert)"),
        f"### 3.2 auc_return_per_step\n",
        _metric_block("auc_per_step_norm", "AUC per Step (normalisiert)"),
        f"### 3.3 sample_efficiency_to_threshold\n",
        _metric_block("sample_efficiency_norm", "Sample Efficiency (normalisiert)"),
        f"### 3.4 wallclock_per_1k_steps\n",
        _metric_block("wallclock_per_1k_raw", "Wallclock / 1000 Schritte (Sekunden, log)"),
        f"### 3.5 return_std_across_seeds\n",
        "Standardabweichung des normalisierten Final-Returns ueber Seeds:\n",
    ])

    std_rows = []
    for fam in ["tabular", "deep"]:
        sub = metrics_df[metrics_df["family"] == fam]
        grp = sub.groupby(["algo", "env_id"])["final_mean_return_norm"].std(ddof=0).reset_index()
        grp.columns = ["Algorithmus", "Umgebung", "Std"]
        grp["Std"] = grp["Std"].apply(lambda v: f"{v:.3f}" if not np.isnan(v) else "-")
        std_rows.append(f"**{fam}:**\n\n{_format_table(grp)}\n")
    metric_sections += "\n".join(std_rows)
    metric_sections += "\n**Sinnvoll in beiden Welten?** [TODO: User-Text]\n"

    md = f"""# Task (b) — Metric Reflection: Tabular vs. Deep RL

Auto-generiert am {ts}, Git-SHA {sha}.

## 1. Datenbasis

- **Tabular:** {n_tabular} Runs aus {len(tabular_algos)} Algos
  ({", ".join(tabular_algos)}) x {len(tabular_envs)} Envs
  ({", ".join(tabular_envs)})
- **Deep:** {n_deep} Runs aus {len(deep_algos)} Algos
  ({", ".join(deep_algos)}) x {len(deep_envs)} Envs
  ({", ".join(deep_envs)})
- **Quelle Tabular:** {tabular_source}

## 2. Metric-Uebersicht (normalisiert)

{_format_table(overview_df)}

{metric_sections}

## 4. Theorieanker

- **Tabular:** Konvergenzgarantien unter Robbins-Monro-Bedingungen (schrittweise
  abnehmende Lernrate, ausreichende Exploration). Bellman-Optimalitaet garantiert
  eindeutige optimale Q-Werte im tabular Fall.
- **Deep RL:** Policy-Gradient (Theorem 5.1.4, 5.1.8, 5.1.11, Skript S. 166ff)
  — Konvergenz nur unter Glattheitsannahmen (L-Smoothness, Lemma 5.1.14) und mit
  abnehmender Schrittweite. In der Praxis: keine harten Garantien, Hyperparameter-
  Abhaengigkeit, Seed-Varianz.

[TODO: User-Text — verbinde die Asymmetrie der Metriken mit der Asymmetrie
der theoretischen Garantien.]

## 5. Empfehlung fuer die Submission

[TODO: User-Text — welche Metriken nimmst du in den Submission-Plot,
welche laesst du raus, und warum?]
"""
    path = output_dir / "reflection.md"
    path.write_text(md, encoding="utf-8")
    print(f"Report: {path}")


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric Reflection: Tabular vs. Deep RL — Blatt 11, Aufg. 4 (b)"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--regenerate-tabular", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-a-dir", default="results/submission/task_a")
    parser.add_argument("--output-dir", default="results/submission/task_b")
    parser.add_argument("--figure-dir", default="figures/submission/task_b")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    figure_dir = pathlib.Path(args.figure_dir).resolve()
    task_a_dir = pathlib.Path(args.task_a_dir).resolve()
    tabular_dir = output_dir / "tabular"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Tabular-Daten
    tabular_df = _load_or_generate_tabular(tabular_dir, args.regenerate_tabular, args.quick)
    tabular_source = "regenerated by tabular_baselines.py" if args.regenerate_tabular else str(tabular_dir)
    tabular_metrics = _compute_tabular_metrics(tabular_df, tabular_dir)

    # Deep-Daten
    deep_csv = task_a_dir / "runs.csv"
    if not deep_csv.exists():
        print(f"FEHLER: {deep_csv} nicht gefunden. Zuerst task_a_eval_study ausfuehren.", file=sys.stderr)
        sys.exit(1)
    deep_metrics = _compute_deep_metrics(deep_csv)

    if not deep_metrics:
        print("WARNUNG: Keine verwertbaren Deep-RL-Metriken (Monitor-CSVs fehlen?)", file=sys.stderr)

    # Kombinieren
    all_rows = tabular_metrics + deep_metrics
    metrics_df = pd.DataFrame(all_rows)

    # Speichern
    metrics_csv = output_dir / "task_b_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metriken: {metrics_csv}")

    metrics_json = output_dir / "task_b_metrics.json"
    metrics_json.write_text(
        metrics_df.to_json(orient="records", indent=2), encoding="utf-8"
    )

    # Plots
    _plot_metric_grid(metrics_df, figure_dir)
    _plot_scatter(metrics_df, figure_dir)
    _plot_wallclock(metrics_df, figure_dir)

    # Reflexions-Report
    _generate_reflection(metrics_df, output_dir, tabular_source)
    print("Fertig.")


if __name__ == "__main__":
    main()
