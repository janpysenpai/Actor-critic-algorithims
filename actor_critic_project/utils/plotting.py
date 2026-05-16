"""Plotting-Hilfsklassen für Lernkurven und Evaluationsplots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
    """Speichert Figure, erstellt Parent-Verzeichnis falls nötig, schließt Figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
