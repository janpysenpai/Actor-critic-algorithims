"""Baut das Submission-ZIP fuer Uebungsblatt 11, Aufgabe 4.

Inhalt des Archivs:
- MANIFEST.json (aus dem Driver-Lauf)
- task_a/  CSV, JSON, Markdown (keine Modell-.zip)
- task_b/  CSV, JSON, reflection.md
- figures/ alle PNGs aus task_a/ und task_b/
- code_snapshot.zip  git-archive HEAD des gesamten Repos
- README.md  abgespecktes Submission-README (via Marker-Kommentare)

Aufruf:
    python -m actor_critic_project.experiments.submission.build_submission_archive
"""

from __future__ import annotations

import argparse
import io
import pathlib
import subprocess
import zipfile
from datetime import datetime, timezone


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT = _REPO_ROOT / "results" / "submission" / "jan_salama_uebungsblatt11.zip"
_SUBMISSION_ROOT = _REPO_ROOT / "results" / "submission"
_FIGURES_ROOT = _REPO_ROOT / "figures" / "submission"

_MARKER_START = "<!-- submission-readme-start -->"
_MARKER_END = "<!-- submission-readme-end -->"


def _extract_submission_readme(readme_path: pathlib.Path) -> str:
    """Extrahiert den Abschnitt zwischen den Submission-Markern."""
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(_MARKER_START)
    end = text.find(_MARKER_END)
    if start == -1 or end == -1:
        return text
    return text[start + len(_MARKER_START):end].strip()


def _git_archive_bytes() -> bytes:
    """Gibt den Inhalt von 'git archive HEAD' als Bytes-ZIP zurueck."""
    result = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git archive fehlgeschlagen: {result.stderr.decode()}")
    return result.stdout


def _add_dir_glob(
    zf: zipfile.ZipFile,
    source_dir: pathlib.Path,
    pattern: str,
    arcname_prefix: str,
    exclude_suffix: tuple[str, ...] = (),
) -> list[str]:
    """Fuegt alle Dateien, die pattern in source_dir matchen, ins ZIP ein."""
    added: list[str] = []
    if not source_dir.exists():
        return added
    for path in source_dir.glob(pattern):
        if exclude_suffix and path.suffix in exclude_suffix:
            continue
        arcname = arcname_prefix + "/" + path.name
        zf.write(path, arcname)
        added.append(arcname)
    return added


def build_archive(output_path: pathlib.Path, include_models: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    added_files: list[str] = []

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # MANIFEST.json
        manifest_path = _SUBMISSION_ROOT / "MANIFEST.json"
        if manifest_path.exists():
            zf.write(manifest_path, "MANIFEST.json")
            added_files.append("MANIFEST.json")

        # task_a: CSV, JSON, MD — keine .zip-Modelldateien
        task_a_dir = _SUBMISSION_ROOT / "task_a"
        exclude = () if include_models else (".zip",)
        for suffix in ("*.csv", "*.json", "*.md"):
            added_files.extend(
                _add_dir_glob(zf, task_a_dir, suffix, "task_a", exclude_suffix=exclude)
            )

        # task_b: CSV, JSON, MD
        task_b_dir = _SUBMISSION_ROOT / "task_b"
        for suffix in ("*.csv", "*.json", "*.md"):
            added_files.extend(_add_dir_glob(zf, task_b_dir, suffix, "task_b"))

        # figures/task_a und task_b
        for sub in ("task_a", "task_b"):
            fig_dir = _FIGURES_ROOT / sub
            added_files.extend(_add_dir_glob(zf, fig_dir, "*.png", f"figures/{sub}"))

        # Code-Snapshot (git archive HEAD als verschachteltes ZIP)
        try:
            archive_bytes = _git_archive_bytes()
            zf.writestr("code_snapshot.zip", archive_bytes)
            added_files.append("code_snapshot.zip")
        except RuntimeError as e:
            print(f"WARNUNG: code_snapshot konnte nicht erstellt werden: {e}")

        # Abgespecktes README
        readme_src = _REPO_ROOT / "README.md"
        if readme_src.exists():
            readme_text = _extract_submission_readme(readme_src)
            zf.writestr("README.md", readme_text.encode("utf-8"))
            added_files.append("README.md")

        # Erstellungszeitstempel als Metadatei
        meta = (
            f"Erstellt: {datetime.now(timezone.utc).isoformat()}\n"
            f"Dateien: {len(added_files)}\n"
        )
        zf.writestr("BUILD_INFO.txt", meta.encode("utf-8"))
        added_files.append("BUILD_INFO.txt")

    # Integritaetspruefung
    with zipfile.ZipFile(output_path, "r") as zf_check:
        bad = zf_check.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP-Korruption erkannt: {bad}")

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"Archiv erstellt: {output_path}")
    print(f"Dateien: {len(added_files)}, Groesse: {size_mb:.2f} MB")
    print("Integritaetspruefung: OK")
    for f in added_files:
        print(f"  {f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baut das Submission-ZIP fuer Uebungsblatt 11"
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Pfad zum Ausgabe-ZIP",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Schliesst Modell-.zip-Dateien ein (gross, default aus)",
    )
    args = parser.parse_args()

    output_path = pathlib.Path(args.output).resolve()
    build_archive(output_path, include_models=args.include_models)


if __name__ == "__main__":
    main()
