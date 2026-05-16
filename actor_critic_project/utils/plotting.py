"""Plotting-Hilfsklassen fuer Lernkurven und Evaluationsplots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    """Gleitender Durchschnitt über window Elemente (min_periods=1)."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_mean_std_band(
    ax,
    x: np.ndarray,
    ys: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Plottet Mittelwert über Seeds als Linie und ±1 Std als transparentes Band.

    ys hat Shape (n_seeds, n_points); x hat Shape (n_points,).
    """
    mean = ys.mean(axis=0)
    std = ys.std(axis=0)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25)


def save_figure(fig, path: Path, dpi: int = 150) -> None:
    """Speichert Figure, erstellt Parent-Verzeichnis falls noetig, schliesst Figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_algo_comparison(
    per_algo_dfs: dict[str, list[pd.DataFrame]],
    env_id: str,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Vergleichsplot: Rolling-Mean-Lernkurve pro Algorithmus, gemittelt ueber Seeds.

    Args:
        per_algo_dfs: Algo-Name → Liste von Monitor-DataFrames (ein DF pro Seed).
                      Jeder DataFrame muss die Spalten 'r', 'l', 'step_cumsum' haben
                      (wie von load_monitor_csv geliefert).
        env_id: Gymnasium-Umgebungs-ID fuer den Plot-Titel.
        ax: Bestehende Axes; wird neu erstellt, wenn None.

    Returns:
        Axes-Objekt mit dem fertigen Plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (algo_name, dfs) in enumerate(per_algo_dfs.items()):
        color = colors[i % len(colors)]

        seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for df in dfs:
            x = df["step_cumsum"].to_numpy(dtype=float)
            y = (
                df["r"]
                .rolling(window=100, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            seed_curves.append((x, y))

        if not seed_curves:
            continue

        x_max = max(x[-1] for x, _ in seed_curves)
        x_grid = np.linspace(0.0, x_max, 500)

        interpolated = np.array(
            [np.interp(x_grid, x, y) for x, y in seed_curves]
        )
        plot_mean_std_band(ax, x_grid, interpolated, label=algo_name, color=color)

    ax.set_xlabel("Kumulative Timesteps")
    ax.set_ylabel("Rolling-Mean Episode-Return (Fenster 100)")
    ax.set_title(f"Algorithmusvergleich: {env_id}")
    ax.legend()
    return ax


def plot_metric_bar(
    per_algo_metric: dict[str, list[float]],
    metric_name: str,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Balkenplot einer skalaren Metrik pro Algorithmus mit Errorbars (Mean +/- Std ueber Seeds).

    Args:
        per_algo_metric: Algo-Name → Liste von Metrikwerten (ein Wert pro Seed).
        metric_name: Bezeichnung der Metrik fuer Achsenbeschriftung und Titel.
        ax: Bestehende Axes; wird neu erstellt, wenn None.

    Returns:
        Axes-Objekt mit dem fertigen Plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    algos = list(per_algo_metric.keys())
    means = [float(np.mean(v)) for v in per_algo_metric.values()]
    stds = [float(np.std(v)) for v in per_algo_metric.values()]

    x = np.arange(len(algos))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} nach Algorithmus (Mean +/- Std ueber Seeds)")
    return ax
