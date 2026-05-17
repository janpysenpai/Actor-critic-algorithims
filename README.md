# Actor-Critic Reinforcement Learning

Implementierung und Evaluation von Policy-Gradient- und Actor-Critic-Algorithmen
in Python — von eigenem Mini-batch REINFORCE (Algorithmus 32 aus dem Skript) bis
zu SB3/sb3-contrib-Algorithmen (A2C, PPO, SAC, DDPG, TD3, TQC, ARS, TRPO).
Entwickelt im Rahmen der Vorlesung *Reinforcement Learning*
(Prof. Dr. Leif Döring, Universität Mannheim, FSS 2026),
Übungsblatt 11 Aufgabe 4.

---

## Submission — Übungsblatt 11

Abgabe-Reproduktion (Quick-Modus, < 10 Minuten):

```bash
python -m actor_critic_project.experiments.submission.run_all_submission --quick
```

Voller Sweep (alle 9 Algos × 5 Envs × 3 Seeds × 100k Steps, Stunden):

```bash
python -m actor_critic_project.experiments.submission.run_all_submission \
    --device cpu --seeds 0 1 2 --total-timesteps 100000
```

Submission-Archiv bauen (nach dem Sweep):

```bash
python -m actor_critic_project.experiments.submission.build_submission_archive
```

Abgabestand: v1.0-uebungsblatt11 (siehe Git-Tag).

---

<!-- submission-readme-start -->

## Algorithmen

| Algorithmus | Quelle | Typ |
|---|---|---|
| Mini-batch REINFORCE | Alg. 32 (Skript, S. 168) — eigene Impl. | On-policy, Policy Gradient |
| A2C | Stable Baselines3 | On-policy, Actor-Critic |
| PPO | Stable Baselines3 | On-policy, Actor-Critic |
| SAC | Stable Baselines3 | Off-policy, Actor-Critic |
| DDPG | Stable Baselines3 | Off-policy, Actor-Critic |
| TD3 | Stable Baselines3 | Off-policy, Actor-Critic |
| TQC | sb3-contrib | Off-policy, Actor-Critic |
| ARS | sb3-contrib | Evolution Strategy |
| TRPO | sb3-contrib | On-policy, Second-Order |

---

## Umgebungen

| Umgebung | Aktionsraum | Episodentyp |
|---|---|---|
| CartPole-v1 | Diskret | Episodisch |
| Acrobot-v1 | Diskret | Episodisch |
| MountainCar-v0 | Diskret | Episodisch |
| MountainCarContinuous-v0 | Kontinuierlich | Episodisch |
| Pendulum-v1 | Kontinuierlich | Episodisch |

---

## Projektstruktur

```
actor_critic_project/
├── algos/
│   └── mini_batch_reinforce/     Eigene REINFORCE-Implementation (Alg. 32)
├── configs/
│   └── hyperparams/              Pro-Algo YAML-Hyperparameter (rl-zoo3-Defaults)
├── envs/                         Gymnasium-Wrapper und Hilfsklassen
├── utils/                        algo_registry, training, evaluation, metrics,
│                                 plotting, normalization, seeds
└── experiments/
    ├── demos/                    Sanity-Checks: ein Demo-Skript pro Algo/Env
    └── submission/               Vollstaendige Submission-Skripte (Blatt 11 Aufg. 4)
        ├── run_all_submission.py       Top-Level-Driver
        ├── task_a_eval_study.py        Task (a): vollstaendiger Sweep
        ├── aggregate_task_a.py         Task (a): Aggregation
        ├── plot_task_a.py              Task (a): Plots
        ├── task_b_metric_reflection.py Task (b): Metric Reflection
        ├── tabular_baselines.py        Q-Learning/SARSA Baselines
        └── build_submission_archive.py Archiv-Builder

figures/submission/               Abgabe-Plots (task_a/, task_b/)
results/submission/               Numerische Ergebnisse, MANIFEST.json
tests/                            Pytest-Tests
```

---

## Installation

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 oder neuer empfohlen.

---

## Tests ausfuehren

```bash
pytest -m "not slow" tests/   # schnelle Tests (< 5 Sekunden)
pytest tests/                 # alle Tests inkl. slow (< 10 Minuten)
```

---

## Kompatibilitaet Algorithmus x Umgebung

x = unterstuetzt, - = nicht kompatibel (Aktionsraum-Inkompatibilitaet):

| Algorithmus | CartPole-v1 | Acrobot-v1 | MountainCar-v0 | MountainCarContinuous-v0 | Pendulum-v1 |
|---|---|---|---|---|---|
| mini_batch_reinforce | x | x | x | x | x |
| a2c | x | x | x | x | x |
| ppo | x | x | x | x | x |
| trpo | x | x | x | x | x |
| ars | x | x | x | x | x |
| ddpg | - | - | - | x | x |
| td3 | - | - | - | x | x |
| sac | - | - | - | x | x |
| tqc | - | - | - | x | x |

Gesamt: 33 kompatible Paare (5 × 3 Discrete-Algos + 5 × 2 Box-Only-Algos + 5 × 2 Box-Envs × 4 = 15 + 18 = 33).

---

## Submission Task (a) — Evaluation Study

Trainiert alle 9 Algorithmen auf allen 5 Classic-Control-Envs ueber mehrere
Seeds und erzeugt Lernkurven, Balkendiagramme und eine Metrik-Tabelle.

**Quick-Run** (< 1 Minute, 3 Algos x 2 Envs x 1 Seed x 5k Steps):

```bash
python -m actor_critic_project.experiments.submission.task_a_eval_study --quick
```

**Voller Sweep** (alle 9 Algos x 5 Envs x 3 Seeds x 100k Steps):

```bash
python -m actor_critic_project.experiments.submission.task_a_eval_study \
    --device cpu --seeds 0 1 2 --total-timesteps 100000
```

**Aggregation und Plots** nach dem Sweep:

```bash
python -m actor_critic_project.experiments.submission.aggregate_task_a
python -m actor_critic_project.experiments.submission.plot_task_a
```

Output-Pfade:

| Inhalt | Pfad |
|---|---|
| Sweep-Ergebnisse (CSV/JSON) | `results/submission/task_a/runs.csv` |
| Metrik-Tabelle (Markdown) | `results/submission/task_a/summary_table.md` |
| Lernkurven-Plots | `figures/submission/task_a/learning_curves_<env>.png` |
| Balkendiagramme | `figures/submission/task_a/final_return_bar_<env>.png` |
| Kombinierte Uebersicht | `figures/submission/task_a/combined_overview.png` |

---

## Submission Task (b) — Metric Reflection

Vergleicht normalisierte Metriken der Tabular-Baselines (Q-Learning, SARSA auf
FrozenLake-v1 und Taxi-v3) mit den Deep-RL-Ergebnissen aus Task (a).

**Quick-Run** (< 1 Minute):

```bash
python -m actor_critic_project.experiments.submission.task_b_metric_reflection \
    --quick --regenerate-tabular --task-a-dir results/submission/task_a
```

**Voller Lauf** (nach Task-(a)-Sweep):

```bash
python -m actor_critic_project.experiments.submission.task_b_metric_reflection \
    --regenerate-tabular --task-a-dir results/submission/task_a
```

Die generierte `reflection.md` enthaelt `[TODO: User-Text]`-Marker fuer die
Prosa-Reflexion, die vom Autor ausgefuellt werden.

Output-Pfade:

| Inhalt | Pfad |
|---|---|
| Metriken (CSV/JSON) | `results/submission/task_b/task_b_metrics.csv` |
| Reflexions-Report | `results/submission/task_b/reflection.md` |
| Metrik-Raster-Plot | `figures/submission/task_b/metric_grid.png` |
| Cross-World-Scatter | `figures/submission/task_b/cross_world_scatter.png` |
| Wallclock-Vergleich | `figures/submission/task_b/wallclock_comparison.png` |

---

## Known Limitations

Mini-batch REINFORCE konvergiert auf MountainCar-v0 und MountainCarContinuous-v0
typischerweise schlecht. Diese Umgebungen haben seltene Belohnungssignale, die reine
Monte-Carlo-Policy-Gradienten ohne Curiosity oder Reward-Shaping kaum auflosen
koennen. Im vollen Sweep ist REINFORCE dort mit grosser Wahrscheinlichkeit das
schwaechste Verfahren.

Die rl-zoo3-Hyperparameter sind nicht fuer jedes Algo-Env-Paar optimiert. Einige
Kombinationen (z.B. ARS auf Pendulum-v1 mit LinearPolicy) lernen mit den geerbten
Defaults langsam und wuerden von einer env-spezifischen Suche profitieren.

Die Cross-World-Metric-Normalisierung in Task (b) beruht auf empirisch ermittelten
Heuristiken (offset, scale) und nicht auf einer etablierten Vergleichsskala. Die
normalisierten Werte ermoglichen eine grobe Orientierung, aber keinen statistisch
belastbaren Vergleich zwischen Tabular- und Deep-RL-Familien.

`results/submission/` und `figures/submission/` sind bewusst nicht in `.gitignore`
eingetragen — Submission-Outputs werden versioniert. Modell-Zips (`.zip`) sind
ausgenommen.

<!-- submission-readme-end -->

---

## Demos ausfuehren (Entwicklungsnotizen)

**Blatt 9 Setup-Check** — Gymnasium-API und SB3-Toolchain-Verifikation:

```bash
python -m actor_critic_project.experiments.demos.run_sb3_explore --quick
```

**Blatt 10 Aufgabe 7** — Mini-batch REINFORCE auf CartPole-v1 und Acrobot-v1:

```bash
python -m actor_critic_project.experiments.demos.run_reinforce_cartpole --quick
python -m actor_critic_project.experiments.demos.run_reinforce_acrobot --quick
```

---

## Hinweis zur KI-Unterstuetzung

Die technische Implementierung wurde mit Unterstuetzung von Claude (Anthropic)
durchgefuehrt. Konzeption der Experimente, Auswahl der Algorithmen und
Hyperparameter, inhaltliche Validierung der Ergebnisse sowie alle
Submission-Texte stammen vom Autor.

---

## Autor

Jan Salama, Universitaet Mannheim, FSS 2026
