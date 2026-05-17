"""Cross-Env Return-Normalisierung fuer den Tabular-vs-Deep-Vergleich (Blatt 11, Aufg. 4 b).

Alle Werte sind Heuristiken fuer typische Return-Bereiche der jeweiligen Umgebung —
NICHT aus der Literatur, sondern aus empirischer Beobachtung abgeleitet. Ziel ist eine
grobe [0, 1]-Projektion fuer den Cross-Env-Vergleich von Metriken, nicht eine
mathematisch exakte Normalisierung.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# (offset, scale): normalize(r) = (r - offset) / scale
# offset ≈ Return einer Zufallspolicy; offset + scale ≈ Return einer guten Policy
_NORMALIZERS: dict[str, tuple[float, float]] = {
    "FrozenLake-v1": (0.0, 1.0),
    "Taxi-v3": (-200.0, 220.0),       # optimal ≈ +7; random ≈ -200
    "CartPole-v1": (0.0, 500.0),
    "Acrobot-v1": (-500.0, 400.0),    # gut ≈ -100; zufaellig ≈ -500
    "MountainCar-v0": (-200.0, 90.0), # optimal ≈ -110; zufaellig ≈ -200
    "MountainCarContinuous-v0": (-100.0, 200.0),
    "Pendulum-v1": (-1600.0, 1500.0), # gut ≈ -200; zufaellig ≈ -1200
}


def get_normalizer(env_id: str) -> tuple[float, float]:
    """Gibt (offset, scale) fuer env_id zurueck.

    Unbekannte Envs erhalten (0.0, 1.0) — kein Fehler, aber kein sinnvoller Vergleich.
    """
    return _NORMALIZERS.get(env_id, (0.0, 1.0))


def normalize_return(env_id: str, raw_return: float) -> float:
    """Normalisiert einen Return auf den [0, 1]-Bereich des Envs.

    Werte ausserhalb [0, 1] sind moeglich (Clips werden bewusst nicht gesetzt,
    um Ausreisser sichtbar zu halten).
    """
    offset, scale = get_normalizer(env_id)
    return (raw_return - offset) / scale


def threshold_from_normalized(env_id: str, normalized_threshold: float = 0.7) -> float:
    """Rechnet einen normalisierten Schwellenwert in den Original-Return-Raum zurueck."""
    offset, scale = get_normalizer(env_id)
    return offset + normalized_threshold * scale


def normalize_series(env_id: str, raw_values: "np.ndarray") -> "np.ndarray":
    """Vektorisierte Normalisierung eines Arrays von Returns."""
    offset, scale = get_normalizer(env_id)
    return (np.asarray(raw_values, dtype=float) - offset) / scale
