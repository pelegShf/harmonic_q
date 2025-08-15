# Harmonic Q-Learning (Tabular) — with Inertial & Windy Gridworlds

A tiny, modular repo to compare **standard** tabular Q-learning vs a **harmonic-mean** update on discrete Gymnasium environments — including custom **inertial** and **windy** grids, macro-action (**duration**) variants, and a tabular **MountainCar**.

```
harmonic_q/
├─ run.py
├─ run_series.py                  # optional: YAML-driven batches
├─ series.yaml                    # optional: main series file
├─ experiments/                   # optional: per-env YAMLs
│   ├─ mountaincar.yaml
│   └─ windy.yaml
├─ hq/
│  ├─ __init__.py
│  ├─ core.py          # Q-learning (standard + harmonic), training, eval
│  ├─ plotting.py      # headless plots (no temporal smoothing by default)
│  └─ video.py         # single rollout + grid-of-trajectories
└─ envs/
   ├─ __init__.py
   ├─ velocity_grid.py     # VelocityGrid-v0 (inertial gridworld, cliffs)
   ├─ multistep_grid.py    # MultiStepGrid-v0 (4 dirs × {1,2,3} cells; time metric)
   ├─ windy_gridworld.py   # WindyGrid-{v0,Kings,Stoch}
   ├─ mountaincar_tab.py   # MountainCarTab-v0 (tabular/discretized)
   └─ duration_actions.py  # *Hold* variants (macro-actions / durations)
```

---

## Table of Contents

- [Theory: Standard vs Harmonic Q](#theory-standard-vs-harmonic-q)
  - [Updates (math)](#updates-math)
  - [Bottom lines: when harmonic helps](#bottom-lines-when-harmonic-helps)
- [Environments](#environments)
- [How to Use (short)](#how-to-use-short)

---

## Theory: Standard vs Harmonic Q

### Updates (math)

Let the one-step target be
$$
y_t=
\begin{cases}
r_t, & \text{if terminal at } t, \\
r_t+\gamma\max\limits_{a'} Q(s_{t+1},a'), & \text{otherwise.}
\end{cases}
$$

**Standard tabular Q-learning**
$$
Q_{t+1}(s_t,a_t)=Q_t(s_t,a_t)+\alpha\,\big(y_t-Q_t(s_t,a_t)\big)
=(1-\alpha)Q_t(s_t,a_t)+\alpha\,y_t.
$$

**Harmonic-mean update (robust EMA)**  
Harmonic mean needs positive inputs, so we apply a tiny **shift** to both operands, average harmonically, then shift back:
$$
s = \min\{Q_t(s_t,a_t),\,y_t,\,0\}-\varepsilon_{\rm shift},\quad
Q^{\sim}=Q_t-s,\quad y^{\sim}=y_t-s.
$$
$$
Q^{\sim}_{t+1}(s_t,a_t)
=\left(\frac{1-\alpha}{\max(Q^{\sim},\varepsilon_{\rm div})}
+\frac{\alpha}{\max(y^{\sim},\varepsilon_{\rm div})}\right)^{-1},
\qquad
Q_{t+1}=Q^{\sim}_{t+1}+s.
$$

> For $x,y>0$, $H(x,y)\le A(x,y)$, so the harmonic update is **more conservative** than the arithmetic EMA; it damps large upward target spikes.

### Bottom lines: when harmonic helps

- **Spiky TD targets:** sparse/terminal bonuses or mixed large $\pm$ terminals (e.g., cliffs) create big jumps in $y_t$. Harmonic reduces overshoot and value blowups.
- **Velocity / inertia / macro-actions:** with momentum or **durations** ($\tau>1$), a single update backs up $r_{t:t+\tau-1}+\gamma^{\tau}\max Q$, which is jumpier; harmonic’s conservatism stabilizes learning.
- **No free lunch:** on well-shaped, low-variance tasks (e.g., classic Windy with $-1$/step, $0$ goal), harmonic can learn **slower** unless you tune $\alpha$ slightly higher for it.

---

## Environments

**Legend.** $\mathcal{S}$: state space, $\mathcal{A}$: actions. Rewards shown both as math and plain English. All custom envs support `render_mode="rgb_array"` and `"ansi"`. Macro-action *Hold* variants expose `info['tau']` (duration) and `info['dt']` (time).

| Env ID | $\mathcal{S}$ (state) | $\mathcal{A}$ (actions) | Rewards (math) | Rewards (words) | Notes |
|---|---|---|---|---|---|
| **VelocityGrid-v0** | $(x,y,v_x,v_y)$, discrete (encoded) | Accelerations: noop, up, down, left, right (5) | $$r_t=\begin{cases}+100 & \text{goal}\\ -100 & \text{cliff}\\ -1 & \text{else}\end{cases}$$ | −1 per step; +100 at goal; −100 on cliff | Inertial dynamics with speed cap; set `max_speed=2` for stronger inertia. |
| **MultiStepGrid-v0** | $(x,y)$, discrete | Moves: 4 dirs × magnitudes $m\in\{1,2,3\}$ (12) | same as above; time per step: $\Delta t=1/m$ (in `info['dt']`) | Larger moves “faster”; −1 per decision; cliffs/goal as above | Logs **Time** = $\sum \Delta t$ per episode. |
| **WindyGrid-v0** | $(x,y)$, discrete | 4-neighbors (L,R,U,D) | $$r_t=-1\ \forall t;\ \text{on goal: } r_t{+}=0$$ | −1 per step; 0 at goal | Classic Sutton & Barto; wind pushes up by column. |
| **WindyGridKings-v0** | $(x,y)$, discrete | 8-neighbors (King’s moves) | same as Windy | same | Faster optimal routes; same reward. |
| **WindyGridStoch-v0** | $(x,y)$, discrete | 4-neighbors | same as Windy | same | Wind adds $\pm1$ stochastically with small prob. |
| **VelocityGridHold-v0** | same as VelocityGrid | Macro-actions: (base action, duration $\tau\in\{1,2,3\}$) | $$r_t=\sum_{i=0}^{\tau-1} r^{\rm base}_{t+i}$$; SMDP discount $\gamma^{\tau}$ | Repeats base action $\tau$ steps; returns cumulative reward; sets `tau`,`dt` | Great to test durations; harmonic often steadier. |
| **WindyGridHold-v0** | same as Windy | Macro-actions: $\tau\in\{1,2,3\}$ | cumulative as above | cumulative as above | Macro-steps in windy grid. |
| **MountainCarTab-v0** | $(\text{pos},\text{vel})$ discretized to bins | 3: push-left, no-push, push-right | $$r_t=-1;\ \text{terminate on goal (no bonus)}$$ | −1 per step; no terminal bonus | Tabular wrapper; optional shaped/bonus variants are easy to add. |
| **CliffWalking-v0** *(built-in)* | $(x,y)$ | 4-neighbors | $$r_t=\begin{cases}0 & \text{goal}\\ -100 & \text{cliff}\\ -1 & \text{else}\end{cases}$$ | −1 step; −100 cliff; 0 goal | Standard benchmark (spiky negatives). |
| **FrozenLake-v1** *(built-in)* | $(x,y)$ | 4-neighbors | $$r_t=\begin{cases}+1 & \text{goal}\\ 0 & \text{else}\end{cases}$$ | +1 goal; 0 otherwise (holes end with 0) | With slip (stochastic). |

> SMDP note (Hold variants): we use $\gamma^{\tau}$ inside the target, i.e. $$y_t = R_{t:t+\tau-1} + \gamma^{\tau}\max_{a'}Q(s_{t+\tau},a').$$

---

## How to Use (short)

**Install**
```bash
pip install "gymnasium[toy-text]" pygame numpy<2 matplotlib imageio imageio-ffmpeg pyyaml
```

**Single run (CLI)**
```bash
# Train both variants, plot & compare
python run.py --env VelocityGrid-v0 --variants both --seeds 0-4   --episodes 8000 --eps_decay_episodes 7600 --task all
```

**Batch runs from YAML (optional)**
```bash
# Run the series defined in series.yaml (uses experiments/*.yaml)
python run_series.py --series series.yaml --task all
```

**Outputs**
- Logs: `logs/<variant>/s<seed>/{returns,steps,time}.csv` (or under `logs/<exp_id>/<model_id>/...` when using YAML runs)
- Plots: `plots/{standard|harmonic|compare}/<ENV_KEY>/*.png` (mean in bold; seeds faint)
- Videos (optional): `videos/<variant>/s<seed>/*.mp4` (single) and `.../grids/*grid*.mp4` (R×C trajectories)

**Tip:** For velocity/duration-heavy tasks, use a slightly larger $\alpha$ for harmonic (e.g., `--alpha 0.30` vs `0.20`) and consider slower $\epsilon$ decay to avoid late flat plateaus.
