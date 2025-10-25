## Program Overview:
This Python program simulates dating outcomes using a control-theoretic model inspired by PID systems. It models emotional dynamics, communication frequency, care events, and reply delays to estimate the likelihood of different relationship outcomes. The simulation is stochastic and uses Monte Carlo methods to generate statistically meaningful results across thousands of trials. The values used in this simulation — especially those ranging from 0 to 1 — are entirely personal and should be based on your own experiences, patterns, and reflections. You’re not modeling a generic dating scenario; you’re modeling your dating dynamics.
When setting values, consider:
• 	How often you and your dates typically communicate
• 	How responsive or expressive you tend to be
• 	How often thoughtful gestures occur
• 	How emotionally open you or your partners are
• 	How reply delays have historically affected outcomes
These values can be informed by past conversations, dates, and your own intuition. There’s no “correct” number — just what feels representative of your patterns. The simulation is designed to be flexible and introspective, not prescriptive.

---

##  Output Overview

After running the simulation, the program prints a breakdown of dating outcomes as percentages. These outcomes represent the **simulated likelihood** of different relationship trajectories based on your input parameters and behavioral assumptions.

### Example Output:
```
Parameters: {...}
Initial x: 0.70
Results (%):
     defined_relationship:  14.82%
           stable_casual:  48.37%
            gradual_fade:  30.12%
     abrupt_disengagement:  6.69%
```

---

## 🧠 What Each Outcome Means

| Outcome | Description |
|--------|-------------|
| `defined_relationship` | High emotional connection and at least two scheduled plans — indicates a committed relationship |
| `stable_casual` | Moderate connection with at least one plan — suggests a consistent but undefined dynamic |
| `gradual_fade` | Connection drops below threshold over time — reflects slow disengagement or loss of interest |
| `abrupt_disengagement` | Connection collapses early (within first 2 steps) — simulates ghosting or sudden cutoff |

These categories are **mutually exclusive** — each simulation run ends in exactly one outcome.

---

##  Sweep Mode Output

When using the `sweep` command, the program:
- Varies the `reply_delay_days` parameter across a range
- Runs simulations for each delay value
- Saves results to a CSV file (e.g. `delay_sweep.csv`)
- Optionally generates a plot showing how delay affects outcome probabilities

### Example Plot:
- X-axis: Reply delay (in days)
- Y-axis: Probability of each outcome
- Lines: One for each outcome category

This helps you visualize how communication timing influences dating trajectories.

---

## Interpreting the Results

The percentages are based on `N` simulation runs (e.g. 10,000). They reflect **statistical tendencies**, not deterministic predictions. Use them to explore patterns, not to forecast specific outcomes.

You can rerun the simulation with different values for:
- Initial connection level (`init_x`)
- Communication frequency
- Care baseline
- Reply delay

This lets you test hypotheses about your dating behavior and see how small changes might shift the outcome distribution.
