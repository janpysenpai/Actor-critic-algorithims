"""Tests fuer Task (b) — Metric Reflection (Blatt 11, Aufg. 4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from actor_critic_project.utils.normalization import (
    normalize_return,
    threshold_from_normalized,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Unit-Tests Normalisierung (nicht slow)
# ---------------------------------------------------------------------------


def test_normalization_round_trip() -> None:
    """normalize_return und threshold_from_normalized sind konsistent."""
    # CartPole: offset=0, scale=500
    assert normalize_return("CartPole-v1", 500.0) == pytest.approx(1.0)
    assert normalize_return("CartPole-v1", 0.0) == pytest.approx(0.0)

    # Acrobot: offset=-500, scale=400 → guter Return -100 → 1.0
    assert normalize_return("Acrobot-v1", -100.0) == pytest.approx(1.0)
    assert normalize_return("Acrobot-v1", -500.0) == pytest.approx(0.0)

    # FrozenLake: offset=0, scale=1
    assert normalize_return("FrozenLake-v1", 1.0) == pytest.approx(1.0)
    assert normalize_return("FrozenLake-v1", 0.0) == pytest.approx(0.0)

    # threshold_from_normalized ist die Umkehrfunktion
    thr = threshold_from_normalized("CartPole-v1", 0.7)
    assert thr == pytest.approx(350.0)
    assert normalize_return("CartPole-v1", thr) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# End-to-End-Smoke (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_task_b_quick_pipeline(tmp_path: Path) -> None:
    """End-to-End-Smoke fuer Task (b) im Quick-Modus."""
    task_a_dir = tmp_path / "task_a"
    task_b_dir = tmp_path / "task_b"
    fig_dir = tmp_path / "figures"

    # Schritt 1: Task (a) Quick-Sweep, damit die Deep-Metriken vorliegen
    result_a = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.submission.task_a_eval_study",
            "--quick",
            "--output-dir",
            str(task_a_dir),
            "--no-confirm",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result_a.returncode == 0, (
        f"Task (a) fehlgeschlagen:\nSTDOUT:\n{result_a.stdout}\nSTDERR:\n{result_a.stderr}"
    )

    # Schritt 2: Task (b) Quick-Sweep mit --regenerate-tabular
    result_b = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.submission.task_b_metric_reflection",
            "--quick",
            "--regenerate-tabular",
            "--task-a-dir",
            str(task_a_dir),
            "--output-dir",
            str(task_b_dir),
            "--figure-dir",
            str(fig_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result_b.returncode == 0, (
        f"Task (b) fehlgeschlagen:\nSTDOUT:\n{result_b.stdout}\nSTDERR:\n{result_b.stderr}"
    )

    # task_b_metrics.csv hat tabular- und deep-Zeilen
    metrics_csv = task_b_dir / "task_b_metrics.csv"
    assert metrics_csv.exists(), "task_b_metrics.csv fehlt"

    import csv
    with metrics_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "task_b_metrics.csv ist leer"
    families = {r["family"] for r in rows}
    assert "tabular" in families, "Keine tabular-Zeilen in task_b_metrics.csv"
    assert "deep" in families, "Keine deep-Zeilen in task_b_metrics.csv"

    # reflection.md enthaelt alle TODO-Marker
    reflection_md = task_b_dir / "reflection.md"
    assert reflection_md.exists(), "reflection.md fehlt"
    content = reflection_md.read_text(encoding="utf-8")
    assert "[TODO: User-Text]" in content, "TODO-Marker fehlen in reflection.md"

    # Alle drei PNGs existieren und sind nicht trivial klein
    for png_name in ("metric_grid.png", "cross_world_scatter.png", "wallclock_comparison.png"):
        png_path = fig_dir / png_name
        assert png_path.exists(), f"{png_name} fehlt"
        assert png_path.stat().st_size > 10_000, f"{png_name} ist kleiner als 10 kB"
