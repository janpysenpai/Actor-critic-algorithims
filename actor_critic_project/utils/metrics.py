"""Metriken fuer die Evaluationsstudie (Blatt 11, Aufg. 4)."""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# A — Tabular-vergleichbare Metriken
# ---------------------------------------------------------------------------


def final_mean_return(returns: list[float], last_k: int = 100) -> float:
    """Mittlerer Return ueber die letzten k Episoden.

    Entspricht dem finalen Leistungsniveau des Agenten nach dem Training.
    """
    return float(np.mean(returns[-last_k:]))


def auc_return_per_step(monitor_df: pd.DataFrame) -> float:
    """Flaeche unter der Episode-Return-Kurve, normalisiert pro Timestep.

    Nutzt die Trapez-Regel ueber step_cumsum (X) und Episode-Return r (Y).
    Hohere Werte entsprechen schnellerem und stabilerem Lernen.
    """
    if monitor_df.empty or len(monitor_df) < 2:
        return 0.0
    x = monitor_df["step_cumsum"].to_numpy(dtype=float)
    y = monitor_df["r"].to_numpy(dtype=float)
    total_steps = x[-1] - x[0]
    if total_steps <= 0:
        return 0.0
    return float(np.trapezoid(y, x) / total_steps)


def sample_efficiency_to_threshold(
    monitor_df: pd.DataFrame, threshold: float
) -> Optional[int]:
    """Erster Timestep, ab dem der Rolling-Mean (Fenster 100 Episoden) >= threshold.

    Gibt None zurueck, wenn der Threshold ueber das gesamte Training nicht erreicht wird.
    Relevant fuer Blatt 11, Aufg. 4: Vergleich der Sampel-Effizienz.
    """
    if monitor_df.empty:
        return None
    rolling = monitor_df["r"].rolling(window=100, min_periods=1).mean()
    above = monitor_df["step_cumsum"][rolling >= threshold]
    if above.empty:
        return None
    return int(above.iloc[0])


# ---------------------------------------------------------------------------
# B — Deep-RL-spezifische Metriken
# ---------------------------------------------------------------------------


def wallclock_per_1k_steps(wallclock_s: float, total_timesteps: int) -> float:
    """Wanduhrzeit in Sekunden pro 1000 Trainingsschritte."""
    return wallclock_s / (total_timesteps / 1000.0)


def return_std_across_seeds(per_seed_means: list[float]) -> float:
    """Standardabweichung der finalen mittleren Returns ueber Seeds.

    Misst die Stabilitaet eines Algorithmus hinsichtlich Zufallsinitialisierung.
    """
    return float(np.std(per_seed_means))


# ---------------------------------------------------------------------------
# Hilfsfunktion: Monitor-CSV laden
# ---------------------------------------------------------------------------


def load_monitor_csv(path: pathlib.Path) -> pd.DataFrame:
    """Liest eine SB3-Monitor-CSV und fuegt kumulative Step-Spalte hinzu.

    SB3 schreibt Zeile 0 als JSON-Header (mit '#'), Zeile 1 als CSV-Header.
    Spalten: r (episode return), l (episode length), t (wallclock time seit Start).
    Die Spalte step_cumsum enthaelt die kumulierten Episodenlaengen.
    """
    df = pd.read_csv(path, skiprows=1)
    df.columns = df.columns.str.strip()
    df["step_cumsum"] = df["l"].cumsum()
    return df
