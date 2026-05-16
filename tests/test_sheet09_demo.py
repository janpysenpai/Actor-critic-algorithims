"""Smoke-Test für run_sb3_explore im Quick-Modus (Blatt 9 Aufgabe 3)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLOT_PATH = ROOT / "figures" / "sheet09_a2c_vs_dqn_cartpole.png"


@pytest.mark.slow
def test_run_sb3_explore_quick() -> None:
    """Startet das Demo-Skript im Quick-Modus und prüft Returncode + Plot-Datei."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.demos.run_sb3_explore",
            "--quick",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Skript beendet mit Returncode {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert PLOT_PATH.exists(), (
        f"Plot-Datei nicht gefunden: {PLOT_PATH}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
