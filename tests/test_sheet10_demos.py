"""Smoke-Tests fuer die REINFORCE-Demo-Skripte (Blatt 10 Aufgabe 7).

Startet die Skripte als Subprozess im --quick-Modus und prueft:
- Returncode 0
- Plot-Datei existiert
- Summary-JSON existiert und ist valides JSON
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_cartpole_demo_quick():
    """CartPole-Demo im Quick-Modus: 2 Seeds, 5_000 Timesteps."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.demos.run_reinforce_cartpole",
            "--quick",
            "--device",
            "cpu",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Skript fehlgeschlagen:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    plot = ROOT / "figures" / "sheet10_reinforce_cartpole.png"
    assert plot.exists(), f"Plot nicht gefunden: {plot}"

    summary = ROOT / "results" / "sheet10_cartpole" / "summary.json"
    assert summary.exists(), f"Summary nicht gefunden: {summary}"
    with open(summary) as f:
        data = json.load(f)
    assert "env_id" in data
    assert data["env_id"] == "CartPole-v1"
    assert "aggregate" in data


@pytest.mark.slow
def test_acrobot_demo_quick():
    """Acrobot-Demo im Quick-Modus: 2 Seeds, 5_000 Timesteps."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.demos.run_reinforce_acrobot",
            "--quick",
            "--device",
            "cpu",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Skript fehlgeschlagen:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    plot = ROOT / "figures" / "sheet10_reinforce_acrobot.png"
    assert plot.exists(), f"Plot nicht gefunden: {plot}"

    summary = ROOT / "results" / "sheet10_acrobot" / "summary.json"
    assert summary.exists(), f"Summary nicht gefunden: {summary}"
    with open(summary) as f:
        data = json.load(f)
    assert "env_id" in data
    assert data["env_id"] == "Acrobot-v1"
    assert "aggregate" in data
