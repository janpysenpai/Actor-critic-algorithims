# Actor-Critic Reinforcement Learning

Implementierung und Evaluation von Policy-Gradient- und Actor-Critic-Algorithmen
in Python — von eigenem Mini-batch REINFORCE (Algorithmus 32 aus dem Skript) bis
zu vortrainierten SB3/sb3-contrib-Algorithmen (A2C, PPO, SAC, DDPG, TD3, TQC,
ARS, TRPO). Entwickelt im Rahmen der Vorlesung *Reinforcement Learning*
(Prof. Dr. Leif Döring, Universität Mannheim, FSS 2026),
Übungsblatt 11 Aufgabe 4. Theoretische Grundlage: Vorlesungsskript Kapitel 5
und Abschnitt 7.8.

Der eigene Mini-batch REINFORCE-Algorithmus erbt von
`stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm` und ist
SB3-Zoo-kompatibel.

---

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
│   └── mini_batch_reinforce/   Eigene REINFORCE-Implementation (Alg. 32)
├── envs/                       Gymnasium-Wrapper und Hilfsklassen
├── utils/                      Plotting, Logging, Seed-Utilities
└── experiments/
    ├── demos/                  Sanity-Checks: ein Demo-Skript pro Algo/Env
    └── submission_tasks/       Vollständige Experiment-Skripte (Blatt 11 Aufg. 4)

figures/submission/             Abgabe-Plots
results/submission/             Numerische Ergebnisse als JSON
tests/                          Pytest-Tests
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

## Tests ausführen

```bash
pytest tests/
```

---

## Demos ausführen

**Blatt 9 Setup-Check** — Gymnasium-API-Tour und SB3-Toolchain-Verifikation
(A2C vs. DQN auf CartPole-v1, On- vs. Off-Policy-Vergleich):

```bash
python -m actor_critic_project.experiments.demos.run_sb3_explore
python -m actor_critic_project.experiments.demos.run_sb3_explore --quick  # 5 000 Steps
```

```bash
python -m actor_critic_project.experiments.demos.run_reinforce_cartpole
```

Plots landen in `figures/`.

---

## Submission reproduzieren

```bash
python -m actor_critic_project.experiments.run_all_submission
```

**Output-Pfade:**

| Inhalt | Pfad |
|---|---|
| Plots (PNG) | `figures/submission/task_X_*.png` |
| Numerische Ergebnisse | `results/submission/task_X.json` |

---

## Hinweis zur KI-Unterstützung

Die technische Implementierung wurde mit Unterstützung von Claude (Anthropic)
durchgeführt. Konzeption der Experimente, Auswahl der Algorithmen und
Hyperparameter, inhaltliche Validierung der Ergebnisse sowie alle
Submission-Texte stammen vom Autor.

---

## Autor

Jan Salama, Universität Mannheim, FSS 2026
