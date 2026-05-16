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

**Blatt 10 Aufgabe 7** — Mini-batch REINFORCE auf CartPole-v1 und Acrobot-v1
(mehrere Seeds, Lernkurven-Plot, JSON-Summary). Diese Demos decken Blatt 10
Aufgabe 7 ab.

```bash
python -m actor_critic_project.experiments.demos.run_reinforce_cartpole
python -m actor_critic_project.experiments.demos.run_reinforce_cartpole --quick  # 2 Seeds, 5 000 Steps
python -m actor_critic_project.experiments.demos.run_reinforce_acrobot
python -m actor_critic_project.experiments.demos.run_reinforce_acrobot --quick
```

Plots landen in `figures/`, Modelle unter `results/sheet10_*/seed_*.zip`.

---

## Trainierte Agenten laden

```python
from actor_critic_project.algos.mini_batch_reinforce import MiniBatchREINFORCE
import gymnasium as gym

model = MiniBatchREINFORCE.load("results/sheet10_cartpole/seed_0")
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()
```

---

## Unified Pipeline

Einheitliche API fuer Training und Evaluation aller Algo-Env-Kombinationen:

```python
from actor_critic_project.utils.training import train_one
from actor_critic_project.utils.evaluation import evaluate_model
from stable_baselines3 import A2C

result = train_one("a2c", "CartPole-v1", total_timesteps=10_000, seed=0)
# result.model_path enthaelt den Pfad zur gespeicherten .zip-Datei (wenn log_dir gesetzt)

model = A2C.load(result.model_path)
eval_result = evaluate_model(model, "CartPole-v1", n_episodes=20, seed=0)
print(f"Mean Return: {eval_result.mean_return:.1f} +/- {eval_result.std_return:.1f}")
```

Kompatibilitaet Algorithmus x Umgebung (x = unterstuetzt, - = nicht kompatibel):

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

Gesamt: 33 kompatible Paare (3 x 5 Discrete-Envs + 2 x 9 Box-Envs).

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
