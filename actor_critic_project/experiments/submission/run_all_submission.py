"""Top-Level-Driver fuer den vollstaendigen Submission-Workflow (Blatt 11, Aufg. 4).

Fuehrt Task (a) und Task (b) sequenziell aus, schreibt ein MANIFEST.json mit
Laufzeiten und Returncodes, und gibt eine abschliessende Uebersicht aus.

Aufruf (Quick-Modus, < 10 Minuten):
    python -m actor_critic_project.experiments.submission.run_all_submission --quick

Voller Sweep:
    python -m actor_critic_project.experiments.submission.run_all_submission \
        --device cpu --seeds 0 1 2 --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import stable_baselines3
    _SB3_VERSION = stable_baselines3.__version__
except ImportError:
    _SB3_VERSION = "unknown"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODULE_BASE = "actor_critic_project.experiments.submission"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    try:
        r = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=str(_REPO_ROOT), capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return True


def _tee_stream(stream, log_file, target_stream) -> None:
    """Liest stream zeilenweise, schreibt in log_file und target_stream."""
    for line in iter(stream.readline, b""):
        decoded = line.decode("utf-8", errors="replace")
        target_stream.write(decoded)
        target_stream.flush()
        log_file.write(decoded)
        log_file.flush()
    stream.close()


def _run_subtask(
    label: str,
    cmd: list[str],
    log_path: pathlib.Path,
    strict: bool,
) -> tuple[int, float]:
    """Startet einen Subprozess, teet stdout/stderr live und in eine Logdatei."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[{label}] Start: {' '.join(cmd)}")
    print(f"[{label}] Log:   {log_path}")
    print(f"{'='*60}\n")
    t0 = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        t_out = threading.Thread(target=_tee_stream, args=(proc.stdout, lf, sys.stdout), daemon=True)
        t_err = threading.Thread(target=_tee_stream, args=(proc.stderr, lf, sys.stderr), daemon=True)
        t_out.start()
        t_err.start()
        proc.wait()
        t_out.join()
        t_err.join()

    duration = time.perf_counter() - t0
    rc = proc.returncode
    status = "OK" if rc == 0 else f"FEHLER (returncode={rc})"
    print(f"\n[{label}] {status}, Dauer: {duration:.1f}s")

    if rc != 0 and strict:
        print(f"[{label}] --strict gesetzt: Abbruch.", file=sys.stderr)
        sys.exit(rc)

    return rc, duration


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submission-Driver: Task (a) + Task (b), Blatt 11, Aufg. 4"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--skip-task-a", action="store_true")
    parser.add_argument("--skip-task-b", action="store_true")
    parser.add_argument("--output-root", default="results/submission")
    parser.add_argument("--strict", action="store_true",
                        help="Abbruch beim ersten fehlgeschlagenen Subtask.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Dirty Git-Zustand erlauben (nicht empfohlen).")
    args = parser.parse_args()

    output_root = (_REPO_ROOT / args.output_root).resolve()
    task_a_dir = output_root / "task_a"
    task_b_dir = output_root / "task_b"
    log_dir = output_root / "logs"
    figure_root = _REPO_ROOT / "figures" / "submission"
    start_time = datetime.now(timezone.utc)

    # Pre-Flight-Check
    if not args.allow_dirty and not _git_is_clean():
        print(
            "FEHLER: Git-Zustand ist nicht clean. Committen oder --allow-dirty nutzen.",
            file=sys.stderr,
        )
        sys.exit(1)

    # MANIFEST schreiben
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "start_time": start_time.isoformat(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "sb3_version": _SB3_VERSION,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "config": {
            "quick": args.quick,
            "device": args.device,
            "seeds": args.seeds,
            "total_timesteps": args.total_timesteps,
            "skip_task_a": args.skip_task_a,
            "skip_task_b": args.skip_task_b,
            "strict": args.strict,
        },
        "subtasks": {},
        "artifacts": [],
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    seed_args = [str(s) for s in args.seeds]
    results: dict[str, dict] = {}

    # Task (a): Sweep
    if not args.skip_task_a:
        task_a_cmd = [
            sys.executable, "-m", f"{_MODULE_BASE}.task_a_eval_study",
            "--output-dir", str(task_a_dir),
            "--device", args.device,
            "--seeds", *seed_args,
            "--total-timesteps", str(args.total_timesteps),
            "--no-confirm",
        ]
        if args.quick:
            task_a_cmd.append("--quick")
        rc_a, dur_a = _run_subtask("task_a_sweep", task_a_cmd, log_dir / "task_a_sweep.log", args.strict)
        results["task_a_sweep"] = {"returncode": rc_a, "duration_s": round(dur_a, 2)}

        # Aggregation
        agg_cmd = [
            sys.executable, "-m", f"{_MODULE_BASE}.aggregate_task_a",
            "--input-dir", str(task_a_dir),
            "--output-dir", str(task_a_dir),
        ]
        rc_agg, dur_agg = _run_subtask("aggregate_task_a", agg_cmd, log_dir / "aggregate_task_a.log", args.strict)
        results["aggregate_task_a"] = {"returncode": rc_agg, "duration_s": round(dur_agg, 2)}

        # Plots Task (a)
        plot_a_cmd = [
            sys.executable, "-m", f"{_MODULE_BASE}.plot_task_a",
            "--input-dir", str(task_a_dir),
            "--figure-dir", str(figure_root / "task_a"),
        ]
        rc_pa, dur_pa = _run_subtask("plot_task_a", plot_a_cmd, log_dir / "plot_task_a.log", args.strict)
        results["plot_task_a"] = {"returncode": rc_pa, "duration_s": round(dur_pa, 2)}

    # Task (b): Metric Reflection
    if not args.skip_task_b:
        task_b_cmd = [
            sys.executable, "-m", f"{_MODULE_BASE}.task_b_metric_reflection",
            "--task-a-dir", str(task_a_dir),
            "--output-dir", str(task_b_dir),
            "--figure-dir", str(figure_root / "task_b"),
            "--regenerate-tabular",
        ]
        if args.quick:
            task_b_cmd.append("--quick")
        rc_b, dur_b = _run_subtask("task_b_reflection", task_b_cmd, log_dir / "task_b_reflection.log", args.strict)
        results["task_b_reflection"] = {"returncode": rc_b, "duration_s": round(dur_b, 2)}

    # Artefakte sammeln
    artifacts: list[str] = []
    for pattern in [
        output_root / "task_a" / "*.csv",
        output_root / "task_a" / "*.md",
        output_root / "task_a" / "*.json",
        output_root / "task_b" / "*.csv",
        output_root / "task_b" / "*.md",
        output_root / "task_b" / "*.json",
        figure_root / "task_a" / "*.png",
        figure_root / "task_b" / "*.png",
    ]:
        for p in pathlib.Path(pattern.parent).glob(pattern.name) if pathlib.Path(pattern.parent).exists() else []:
            artifacts.append(str(p.relative_to(_REPO_ROOT)))

    end_time = datetime.now(timezone.utc)
    total_s = (end_time - start_time).total_seconds()

    manifest.update({
        "end_time": end_time.isoformat(),
        "total_duration_s": round(total_s, 2),
        "subtasks": results,
        "artifacts": artifacts,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Abschluss-Uebersicht
    print(f"\n{'='*60}")
    print("Submission-Driver abgeschlossen")
    print(f"Gesamtdauer: {total_s:.1f}s")
    print(f"Manifest:    {manifest_path}")
    print(f"\nSubtask-Ergebnisse:")
    any_error = False
    for name, info in results.items():
        status = "OK" if info["returncode"] == 0 else f"FEHLER ({info['returncode']})"
        print(f"  {name:<30} {status:<20} {info['duration_s']:.1f}s")
        if info["returncode"] != 0:
            any_error = True
    print(f"\nErzeugte Artefakte ({len(artifacts)}):")
    for a in artifacts[:20]:
        print(f"  {a}")
    if len(artifacts) > 20:
        print(f"  ... und {len(artifacts) - 20} weitere (siehe MANIFEST.json)")
    print(f"{'='*60}")

    if any_error:
        print("\nWARNUNG: Mindestens ein Subtask ist fehlgeschlagen.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
