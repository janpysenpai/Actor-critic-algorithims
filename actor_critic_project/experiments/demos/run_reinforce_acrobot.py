"""Blatt 10 Aufgabe 7 — Mini-batch REINFORCE auf Acrobot-v1.

Trainiert den eigenen MiniBatchREINFORCE-Algorithmus (Algorithmus 32, Skript S. 168)
auf Acrobot-v1 über mehrere Seeds. Die Hyperparameter werden aus dem YAML-Block
`Acrobot-v1` in
`actor_critic_project/algos/mini_batch_reinforce/hyperparams.yml` geladen.

Acrobot konvergiert langsamer als CartPole. Mit dem Default-Budget (200k Steps)
ist eine Eval-Mean von -200 bis -100 realistisch. Wenn die Policy noch nicht
konvergiert ist, zeigt das der Plot — diese Beobachtung ist Teil der Analyse
für Blatt 11 Aufgabe 4(b).

Ausgaben:
- Lernkurven-Plot: `figures/sheet10_reinforce_acrobot.png`
- Trainierte Modelle: `results/sheet10_acrobot/seed_*.zip`
- Zusammenfassung: `results/sheet10_acrobot/summary.json`

Verwendung:
    python -m actor_critic_project.experiments.demos.run_reinforce_acrobot
    python -m actor_critic_project.experiments.demos.run_reinforce_acrobot --quick
    python -m actor_critic_project.experiments.demos.run_reinforce_acrobot --seeds 3 --total-timesteps 100000
"""

from __future__ import annotations

from actor_critic_project.experiments.demos._reinforce_demo_helpers import (
    build_arg_parser,
    run_demo,
)

ENV_ID = "Acrobot-v1"
LOG_SUBDIR = "sheet10_acrobot"
PLOT_FILENAME = "sheet10_reinforce_acrobot.png"
SUMMARY_SUBPATH = "sheet10_acrobot/summary.json"


def main() -> None:
    parser = build_arg_parser(ENV_ID)
    args = parser.parse_args()
    run_demo(ENV_ID, LOG_SUBDIR, PLOT_FILENAME, SUMMARY_SUBPATH, args)


if __name__ == "__main__":
    main()
