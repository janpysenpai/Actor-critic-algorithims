"""Tests fuer den Task-(a)-Evaluation-Sweep (Blatt 11, Aufg. 4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from actor_critic_project.experiments.submission.sweep_config import SweepConfig

ROOT = Path(__file__).resolve().parents[1]


def test_sweep_config_expanded_runs() -> None:
    """SweepConfig.expanded_runs() liefert die richtige Anzahl Runs."""
    default_cfg = SweepConfig()
    runs = default_cfg.expanded_runs()
    # 33 kompatible (algo, env)-Paare x 3 Seeds = 99 Runs
    assert len(runs) == 99, f"Erwartet 99, erhalten {len(runs)}"

    quick_cfg = SweepConfig(quick=True)
    quick_runs = quick_cfg.expanded_runs()
    # mini_batch_reinforce, a2c, ppo x CartPole-v1, Pendulum-v1 x 1 Seed = 6 Runs
    assert len(quick_runs) == 6, f"Erwartet 6 Quick-Runs, erhalten {len(quick_runs)}"


@pytest.mark.slow
def test_task_a_quick_sweep(tmp_path: Path) -> None:
    """End-to-end-Test des Task-(a)-Sweeps im Quick-Modus (< 5 Minuten)."""
    run_dir = tmp_path / "runs"
    fig_dir = tmp_path / "figures"

    # Schritt 1: Sweep
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.submission.task_a_eval_study",
            "--quick",
            "--output-dir",
            str(run_dir),
            "--no-confirm",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Sweep fehlgeschlagen:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runs_csv = run_dir / "runs.csv"
    assert runs_csv.exists(), "runs.csv wurde nicht erstellt"

    import csv
    with runs_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1, "runs.csv ist leer"

    config_json = run_dir / "config.json"
    assert config_json.exists(), "config.json wurde nicht erstellt"
    cfg = json.loads(config_json.read_text(encoding="utf-8"))
    assert "git_sha" in cfg, "config.json enthaelt kein 'git_sha'"

    # Schritt 2: Aggregation
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.submission.aggregate_task_a",
            "--input-dir",
            str(run_dir),
            "--output-dir",
            str(run_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result2.returncode == 0, (
        f"Aggregation fehlgeschlagen:\nSTDOUT:\n{result2.stdout}\nSTDERR:\n{result2.stderr}"
    )

    summary_md = run_dir / "summary_table.md"
    assert summary_md.exists(), "summary_table.md wurde nicht erstellt"
    md_content = summary_md.read_text(encoding="utf-8")
    for algo in ("mini_batch_reinforce", "a2c", "ppo"):
        assert algo in md_content, f"Algorithmus '{algo}' fehlt in summary_table.md"

    # Schritt 3: Plots
    result3 = subprocess.run(
        [
            sys.executable,
            "-m",
            "actor_critic_project.experiments.submission.plot_task_a",
            "--input-dir",
            str(run_dir),
            "--figure-dir",
            str(fig_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result3.returncode == 0, (
        f"Plot-Skript fehlgeschlagen:\nSTDOUT:\n{result3.stdout}\nSTDERR:\n{result3.stderr}"
    )

    png_files = list(fig_dir.glob("*.png"))
    assert len(png_files) >= 1, f"Keine PNG-Dateien in {fig_dir}"
