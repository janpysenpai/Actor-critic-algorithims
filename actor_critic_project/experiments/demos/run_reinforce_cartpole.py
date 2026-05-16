"""Blatt 10 Aufgabe 7 — Mini-batch REINFORCE auf CartPole-v1.

Trainiert den eigenen MiniBatchREINFORCE-Algorithmus (Algorithmus 32, Skript S. 168)
auf CartPole-v1 über mehrere Seeds. Die Hyperparameter werden aus dem YAML-Block
`CartPole-v1` in
`actor_critic_project/algos/mini_batch_reinforce/hyperparams.yml` geladen.

Ausgaben:
- Lernkurven-Plot: `figures/sheet10_reinforce_cartpole.png`
- Trainierte Modelle: `results/sheet10_cartpole/seed_*.zip`
- Zusammenfassung: `results/sheet10_cartpole/summary.json`

Verwendung:
    python -m actor_critic_project.experiments.demos.run_reinforce_cartpole
    python -m actor_critic_project.experiments.demos.run_reinforce_cartpole --quick
    python -m actor_critic_project.experiments.demos.run_reinforce_cartpole --seeds 3 --total-timesteps 50000
"""

from __future__ import annotations

from actor_critic_project.experiments.demos._reinforce_demo_helpers import (
    build_arg_parser,
    run_demo,
)

ENV_ID = "CartPole-v1"
LOG_SUBDIR = "sheet10_cartpole"
PLOT_FILENAME = "sheet10_reinforce_cartpole.png"
SUMMARY_SUBPATH = "sheet10_cartpole/summary.json"


def main() -> None:
    parser = build_arg_parser(ENV_ID)
    args = parser.parse_args()
    run_demo(ENV_ID, LOG_SUBDIR, PLOT_FILENAME, SUMMARY_SUBPATH, args)


if __name__ == "__main__":
    main()
