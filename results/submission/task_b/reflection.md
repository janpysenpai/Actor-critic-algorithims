# Task (b) — Metric Reflection: Tabular vs. Deep RL

Auto-generiert am 2026-05-17T13:34:05.387510+00:00, Git-SHA ef19cc8e128abcf0164a6b698c4aa761ba1065f9.

## 1. Datenbasis

- **Tabular:** 4 Runs aus 2 Algos
  (q_learning, sarsa) x 2 Envs
  (FrozenLake-v1, Taxi-v3)
- **Deep:** 6 Runs aus 3 Algos
  (a2c, mini_batch_reinforce, ppo) x 2 Envs
  (CartPole-v1, Pendulum-v1)
- **Quelle Tabular:** /Users/jansalama/Documents/Uni Mannheim/Reinforcement Learning/Programmieraufgaben & Code/RL-Actor-Critic-Algorithms/results/submission/task_b/tabular

## 2. Metric-Uebersicht (normalisiert)

| Family | Metric | Mean | Std | N |
|---|---|---|---|---|
| tabular | Final Mean Return (norm.) | -0.470 | 0.485 | 4 |
| tabular | AUC per Step (norm.) | -0.768 | 0.799 | 4 |
| tabular | Sample Efficiency (norm.) | 0.000 | 0.000 | 4 |
| tabular | Wallclock / 1k Steps (s) | 0.005 | 0.001 | 4 |
| deep | Final Mean Return (norm.) | 0.198 | 0.076 | 6 |
| deep | AUC per Step (norm.) | 0.239 | 0.086 | 6 |
| deep | Sample Efficiency (norm.) | 0.000 | 0.000 | 6 |
| deep | Wallclock / 1k Steps (s) | 0.198 | 0.112 | 6 |

## 3. Per-Metric Reflexion

### 3.1 final_mean_return

### Final Mean Return (normalisiert)

**tabular:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| q_learning | FrozenLake-v1 | 0.020 |
| q_learning | Taxi-v3 | -0.931 |
| sarsa | FrozenLake-v1 | 0.010 |
| sarsa | Taxi-v3 | -0.977 |

**deep:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| a2c | CartPole-v1 | 0.228 |
| a2c | Pendulum-v1 | 0.246 |
| mini_batch_reinforce | CartPole-v1 | 0.054 |
| mini_batch_reinforce | Pendulum-v1 | 0.275 |
| ppo | CartPole-v1 | 0.142 |
| ppo | Pendulum-v1 | 0.242 |

**Sinnvoll in beiden Welten?** Ja — final_mean_return ist die strukturell
robusteste Metrik im Vergleich. In beiden Welten misst sie dasselbe: die
Performanz der gelernten Politik gegen einen festen Referenzwert. Die
normalisierte Skala fängt die Heterogenität der Returns (FrozenLake in
[0, 1] vs. CartPole in [0, 500] vs. Pendulum mit negativen Returns) sauber
auf. Wichtig: der absolute Wert ist nicht zwischen Welten vergleichbar,
nur die relative Ordnung innerhalb einer Welt und das qualitative Niveau
("nahe 1.0" = gut, "nahe 0.0" = random policy, negativ = unter random).
Tabular liegt im Schnitt schlechter, was hier nicht an der Methode liegt,
sondern an der dürftigen Episodenanzahl im Quick-Sweep (2000 Episoden für
Taxi-v3 reichen nicht aus, um den initialen –200er Penalty-Tail zu
überwinden) — das illustriert die Schwäche der Metrik: sie urteilt erst
bei genügender Trainingsdauer fair.

### 3.2 auc_return_per_step

### AUC per Step (normalisiert)

**tabular:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| q_learning | FrozenLake-v1 | 0.038 |
| q_learning | Taxi-v3 | -1.562 |
| sarsa | FrozenLake-v1 | 0.024 |
| sarsa | Taxi-v3 | -1.572 |

**deep:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| a2c | CartPole-v1 | 0.329 |
| a2c | Pendulum-v1 | 0.246 |
| mini_batch_reinforce | CartPole-v1 | 0.058 |
| mini_batch_reinforce | Pendulum-v1 | 0.278 |
| ppo | CartPole-v1 | 0.284 |
| ppo | Pendulum-v1 | 0.238 |

**Sinnvoll in beiden Welten?** Bedingt sinnvoll. AUC integriert die
gesamte Lernkurve und ist damit eine Mischung aus "wie schnell" und "wie
gut" — eine Metrik, die zwei Dimensionen vermengt. Im Tabular-Setting,
wo Konvergenz unter Robbins-Monro garantiert ist, sagt AUC vor allem
etwas über die Lerngeschwindigkeit aus (final wird immer optimal). In
der Deep-Welt ohne Konvergenzgarantien wird AUC durch zwei sehr
verschiedene Szenarien erzeugt: schnelles Lernen bis Plateau (gut) oder
früh hoher Return mit anschließender Instabilität (verzerrt). Bei
unseren Quick-Daten sieht man das für PPO auf CartPole, wo AUC (0.28)
über dem final_mean_return (0.14) liegt — AUC misst dort die transient
gute Performance, nicht die End-Performanz. Empfehlung: AUC nur in
Kombination mit final_mean_return berichten, nie alleinstehend.

### 3.3 sample_efficiency_to_threshold

### Sample Efficiency (normalisiert)

**tabular:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| q_learning | FrozenLake-v1 | 0.000 |
| q_learning | Taxi-v3 | 0.000 |
| sarsa | FrozenLake-v1 | 0.000 |
| sarsa | Taxi-v3 | 0.000 |

**deep:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| a2c | CartPole-v1 | 0.000 |
| a2c | Pendulum-v1 | 0.000 |
| mini_batch_reinforce | CartPole-v1 | 0.000 |
| mini_batch_reinforce | Pendulum-v1 | 0.000 |
| ppo | CartPole-v1 | 0.000 |
| ppo | Pendulum-v1 | 0.000 |

**Sinnvoll in beiden Welten?** Konzeptuell ja, im vorliegenden Quick-Run
nein. Sample-Efficiency-bis-Schwelle ist im Prinzip die natürlichste
Cross-World-Metrik: sie fragt, wie viele Schritte ein Algorithmus
braucht, um normalisierte 70%-Performance zu erreichen. Sie ist
budgetunabhängig in ihrer Interpretation und macht für beide Welten
dasselbe — Tabular und Deep haben den gleichen "Steps-zu-Threshold"-
Begriff, weil beide pro Step ein Environment-Sample verbrauchen.
Pragmatisches Problem: keiner unserer Quick-Runs erreicht die Schwelle.
Im normalisierten Wert wird `None` als 0.0 dargestellt, was im Bar-Chart
fälschlicherweise nach "alle Algorithmen sind gleich schnell" aussieht,
tatsächlich aber bedeutet "keiner erreicht die Schwelle innerhalb des
gegebenen Step-Budgets" — also genau das Gegenteil. Diese
Darstellungsfalle illustriert ein zentrales Risiko bei
Sample-Efficiency-Metriken: sie sind nur dann aussagekräftig, wenn das
Step-Budget hinreichend groß gegenüber der erwarteten Konvergenzzeit
ist. Für die finale Submission würden wir die Schwelle auf 0.5
absenken und den vollen 100k-Sweep fahren — siehe Abschnitt 5.

### 3.4 wallclock_per_1k_steps

### Wallclock / 1000 Schritte (Sekunden, log)

**tabular:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| q_learning | FrozenLake-v1 | 0.005 |
| q_learning | Taxi-v3 | 0.006 |
| sarsa | FrozenLake-v1 | 0.004 |
| sarsa | Taxi-v3 | 0.005 |

**deep:**

| Algorithmus | Umgebung | Mean |
|---|---|---|
| a2c | CartPole-v1 | 0.211 |
| a2c | Pendulum-v1 | 0.163 |
| mini_batch_reinforce | CartPole-v1 | 0.094 |
| mini_batch_reinforce | Pendulum-v1 | 0.101 |
| ppo | CartPole-v1 | 0.429 |
| ppo | Pendulum-v1 | 0.192 |

**Sinnvoll in beiden Welten?** Nur innerhalb einer Welt sinnvoll, nicht
zwischen ihnen. Wallclock pro 1000 Steps liegt für Tabular bei ~0.005 s,
für Deep bei ~0.1–0.4 s — ein Faktor 20–80. Diese Differenz ist nicht
inhaltlich (welcher Algorithmus rechnet effizienter?) sondern strukturell
(Tabular-Update ist ein einzelner Dictionary-Lookup, Deep-Update ist ein
Forward-Backward-Pass durch ein neuronales Netz). Ein Vergleich
Tabular ↔ Deep auf Wallclock-Basis ist deshalb keine Algorithmus-Aussage,
sondern eine Hardware-/Architektur-Aussage. Innerhalb der Deep-Welt
hingegen ist Wallclock nützlich, um den Trade-off zwischen
Sample-Effizienz und Rechenkosten sichtbar zu machen — PPO etwa braucht
hier ~4× so lange pro Step wie REINFORCE, was bei gleichem Step-Budget
zu einem realen Zeitnachteil führt.

### 3.5 return_std_across_seeds

Standardabweichung des normalisierten Final-Returns ueber Seeds:
**tabular:**

| Algorithmus | Umgebung | Std |
|---|---|---|
| q_learning | FrozenLake-v1 | 0.000 |
| q_learning | Taxi-v3 | 0.000 |
| sarsa | FrozenLake-v1 | 0.000 |
| sarsa | Taxi-v3 | 0.000 |

**deep:**

| Algorithmus | Umgebung | Std |
|---|---|---|
| a2c | CartPole-v1 | 0.000 |
| a2c | Pendulum-v1 | 0.000 |
| mini_batch_reinforce | CartPole-v1 | 0.000 |
| mini_batch_reinforce | Pendulum-v1 | 0.000 |
| ppo | CartPole-v1 | 0.000 |
| ppo | Pendulum-v1 | 0.000 |

**Sinnvoll in beiden Welten?** Asymmetrisch sinnvoll — und im Quick-Run
strukturell nicht messbar, weil nur ein Seed gefahren wurde. Im
Tabular-Setting ist die Seed-Varianz konzeptuell klein, weil
Konvergenzgarantien greifen und nach genügender Episodenzahl alle Seeds
auf denselben Q-Wert zulaufen (Q-Learning unter Robbins-Monro). In der
Deep-Welt ist die Seed-Varianz strukturell relevant: ohne harte
Konvergenzgarantien, mit nicht-konvexem Loss und stochastischem
Gradient kann derselbe Algorithmus mit verschiedenen Seeds in komplett
unterschiedlichen lokalen Optima landen — die Seed-Std ist ein direktes
Maß für Reproduzierbarkeit und sollte deshalb in der Deep-Welt
*immer* berichtet werden. Empfehlung: Min. 3 Seeds, besser 5; im
vorliegenden Quick-Sweep nicht erfüllt.


## 4. Theorieanker

- **Tabular:** Konvergenzgarantien unter Robbins-Monro-Bedingungen (schrittweise
  abnehmende Lernrate, ausreichende Exploration). Bellman-Optimalitaet garantiert
  eindeutige optimale Q-Werte im tabular Fall.
- **Deep RL:** Policy-Gradient (Theorem 5.1.4, 5.1.8, 5.1.11, Skript S. 166ff)
  — Konvergenz nur unter Glattheitsannahmen (L-Smoothness, Lemma 5.1.14) und mit
  abnehmender Schrittweite. In der Praxis: keine harten Garantien, Hyperparameter-
  Abhaengigkeit, Seed-Varianz.

Die Asymmetrie der theoretischen Garantien überträgt sich direkt auf die
Sinnhaftigkeit der Metriken. Im Tabular-Setting hat jede Methode ein
eindeutiges Konvergenzziel (Q* bzw. V*), und alle Standardmetriken
messen letztlich nur, wie schnell man dorthin gelangt — final_mean_return
ist asymptotisch trivial (es konvergiert), sample_efficiency ist die
relevante differenzierende Größe, und Seed-Varianz wird im Limit klein.
In der Deep-Welt fehlen alle drei Eigenschaften: Konvergenz ist nicht
garantiert (nur unter L-Smoothness und passender Schrittweite, Lemma
5.1.14), das Optimum ist nicht eindeutig (mehrere lokale Maxima im
Policy-Parameterraum), und Seed-Varianz ist eine fundamentale Eigenschaft
des Trainings, kein Artefakt endlicher Daten. Konsequenz: in der
Deep-Welt muss man final_mean_return zusammen mit Seed-Varianz berichten,
sample_efficiency relativ zu einem festen Step-Budget interpretieren und
Wallclock als reines Operations-Maß behandeln, nicht als
Algorithmus-Eigenschaft.

## 5. Empfehlung fuer die Submission

Für die Submission empfehlen wir folgende Metrik-Auswahl: (i)
final_mean_return über die letzten 100 Episoden, normalisiert pro Env,
gemittelt über min. 3 Seeds, als Hauptmetrik mit explizit ausgewiesener
Seed-Std. (ii) Lernkurven (rolling mean ±1 Std-Band) pro Algorithmus
und Env als visuelles Substitut für die strukturell schwer
quantifizierbare Trainingsstabilität. (iii) sample_efficiency mit auf
0.5 (statt 0.7) abgesenkter Schwelle und ein hinreichend großes
Step-Budget (≥ 100 000 Schritte für die On-Policy-Algorithmen, ≥ 50 000
für die Off-Policy-Algorithmen), damit ein Großteil der Algorithmen
die Schwelle tatsächlich erreicht. Wir lassen weg: (a) AUC per Step
als Stand-Alone-Plot — die Information ist in den Lernkurven besser
sichtbar; (b) Wallclock als Cross-World-Vergleich — wir berichten nur
Wallclock innerhalb der Deep-Welt, und auch dort nur als
Zusatzinformation, nicht als Performance-Kriterium. Diese Auswahl macht
die strukturellen Eigenschaften der Algorithmen sichtbar, ohne die
Quick-Run-Limitierungen als inhaltliche Aussage zu verkleiden.
