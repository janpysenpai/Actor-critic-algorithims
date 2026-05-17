"""Evaluation-Sweep fuer Blatt 11, Aufgabe 4 (a).

Trainiert alle kompatiblen (Algorithmus, Env, Seed)-Kombinationen, evaluiert
die Policies und schreibt Ergebnisse in runs.csv / runs.json.

Aufruf (Quick-Test, < 5 Minuten):
    python -m actor_critic_project.experiments.submission.task_a_eval_study --quick

Aufruf (voller Sweep):
    python -m actor_critic_project.experiments.submission.task_a_eval_study \\
        --device cpu --seeds 0 1 2 --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import stable_baselines3
import tqdm

from actor_critic_project.utils.algo_registry import get_algo
from actor_critic_project.utils.evaluation import evaluate_model
from actor_critic_project.utils.hyperparam_loader import load_hyperparams
from actor_critic_project.utils.training import train_one

from .sweep_config import SweepConfig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_CSV_FIELDS = [
    "algo", "env_id", "seed", "eval_mean", "eval_std",
    "wallclock_s", "total_timesteps", "monitor_csv",
]


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_config(output_dir: pathlib.Path, config: SweepConfig, n_runs: int) -> None:
    cfg = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "python_version": platform.python_version(),
        "sb3_version": stable_baselines3.__version__,
        "algos": list(config.algos),
        "env_ids": list(config.env_ids),
        "seeds": list(config.seeds),
        "total_timesteps": config.total_timesteps,
        "device": config.device,
        "quick": config.quick,
        "n_runs": n_runs,
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _run_one(
    run_algo: str,
    run_env: str,
    run_seed: int,
    total_timesteps: int,
    device: str,
    output_dir: pathlib.Path,
) -> dict:
    """Fuehrt einen einzelnen Trainingslauf durch und gibt das Ergebnis-Dict zurueck."""
    hyperparams = load_hyperparams(run_algo, run_env)
    log_dir = output_dir / run_algo / run_env

    train_result = train_one(
        algo_name=run_algo,
        env_id=run_env,
        total_timesteps=total_timesteps,
        seed=run_seed,
        hyperparams=hyperparams,
        device=device,
        log_dir=log_dir,
    )

    # Modell von Disk laden (train_one gibt kein Modell-Objekt zurueck)
    spec = get_algo(run_algo)
    if train_result.model_path and train_result.model_path.exists():
        model = spec.cls.load(str(train_result.model_path))
    else:
        raise FileNotFoundError(
            f"Modell-Datei nicht gefunden: {train_result.model_path}"
        )

    eval_result = evaluate_model(
        model,
        run_env,
        n_episodes=20,
        seed=run_seed + 1000,
    )

    return {
        "algo": run_algo,
        "env_id": run_env,
        "seed": run_seed,
        "eval_mean": eval_result.mean_return,
        "eval_std": eval_result.std_return,
        "wallclock_s": train_result.wallclock_s,
        "total_timesteps": total_timesteps,
        "monitor_csv": str(train_result.monitor_csv_path or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation Study Sweep — Blatt 11, Aufg. 4 (a)"
    )
    parser.add_argument("--quick", action="store_true", help="Minimaler Test-Sweep (3x2x1, 5k Steps)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--algos", nargs="+", default=list(SweepConfig.algos))
    parser.add_argument("--envs", nargs="+", default=list(SweepConfig.env_ids))
    parser.add_argument(
        "--output-dir",
        default="results/submission/task_a",
        help="Ausgabeverzeichnis fuer Runs-CSV, JSON und Modelle",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Bestaetigung bei grossen Sweeps ueberspringen",
    )
    args = parser.parse_args()

    config = SweepConfig(
        algos=tuple(args.algos),
        env_ids=tuple(args.envs),
        seeds=tuple(args.seeds),
        total_timesteps=args.total_timesteps,
        device=args.device,
        quick=args.quick,
    )
    runs = config.expanded_runs()
    n_runs = len(runs)

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(output_dir, config, n_runs)
    print(f"Sweep: {n_runs} Runs → {output_dir}")

    if (
        not config.quick
        and config.total_timesteps >= 50_000
        and not args.no_confirm
    ):
        total_steps = n_runs * config.total_timesteps
        est_hours = total_steps * 0.0001 / 3600
        response = input(
            f"Sweep ueber {n_runs} Runs mit {total_steps:,} Gesamt-Timesteps.\n"
            f"Geschaetzte Wallclock: {est_hours:.1f} Stunden. Fortfahren? [y/N]: "
        )
        if response.strip().lower() not in ("y", "yes"):
            print("Abgebrochen.")
            sys.exit(0)

    results: list[dict] = []
    errors: list[dict] = []

    for run in tqdm.tqdm(runs, desc="Sweep", unit="run"):
        try:
            row = _run_one(
                run_algo=run.algo,
                run_env=run.env_id,
                run_seed=run.seed,
                total_timesteps=config.total_timesteps,
                device=config.device,
                output_dir=output_dir,
            )
            results.append(row)
        except Exception as exc:
            logger.error(
                "Fehler bei %s/%s/seed%d: %s", run.algo, run.env_id, run.seed, exc
            )
            errors.append({
                "algo": run.algo,
                "env_id": run.env_id,
                "seed": run.seed,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    if results:
        csv_path = output_dir / "runs.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(results)
        print(f"Runs gespeichert: {csv_path} ({len(results)} Eintraege)")

        runs_json: dict = {}
        for row in results:
            algo, env_id, seed = row["algo"], row["env_id"], str(row["seed"])
            runs_json.setdefault(algo, {}).setdefault(env_id, {})[seed] = {
                k: v for k, v in row.items() if k not in ("algo", "env_id", "seed")
            }
        (output_dir / "runs.json").write_text(
            json.dumps(runs_json, indent=2), encoding="utf-8"
        )

    if errors:
        (output_dir / "errors.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
        print(f"Fehler: {output_dir / 'errors.json'} ({len(errors)} Eintraege)")

    print(f"Sweep abgeschlossen: {len(results)} erfolgreich, {len(errors)} fehlgeschlagen.")


if __name__ == "__main__":
    main()
