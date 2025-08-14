# Harmonic Q-Learning (Tabular) — with Inertial Gridworld

A tiny, modular repo to compare **standard** tabular Q-learning vs a **harmonic-mean** update on discrete Gymnasium environments — including a custom **VelocityGrid-v0** with inertia (actions are accelerations).

```
harmonic_q/
├─ run.py
├─ hq/
│  ├─ __init__.py
│  ├─ core.py          # Q-learning (standard + harmonic), training, eval
│  ├─ plotting.py      # headless plots (smoothed only)
│  └─ video.py         # simple RecordVideo helper (optional)
└─ envs/
   ├─ __init__.py
   └─ velocity_grid.py # VelocityGrid-v0 (inertial gridworld, cliff band)
```

---

## 1) Math & Algorithms

### 1.1 Notation
- Finite MDP with states $s \in \mathcal{S}$, actions $a \in \mathcal{A}$.
- Reward $r_t \in \mathbb{R}$, discount $\gamma \in [0,1)$, learning rate $\alpha \in (0,1]$.
- Q-table $Q(s,a) \in \mathbb{R}$.
- One-step **bootstrapped target**:
\[
y_t \;=\;
\begin{cases}
r_t, & \text{if episode terminates at } t \\
r_t + \gamma \,\max_{a'} Q(s_{t+1}, a'), & \text{otherwise.}
\end{cases}
\]

### 1.2 Standard tabular Q-learning
**Update:**
\[
Q_{t+1}(s_t,a_t) \;\leftarrow\; Q_t(s_t,a_t) + \alpha\big(y_t - Q_t(s_t,a_t)\big).
\]

**EMA view (arithmetic mean):**
\[
Q_{t+1} \;=\; (1-\alpha)\,Q_t \;+\; \alpha\,y_t.
\]

### 1.3 Harmonic-mean update (robust EMA)
We replace the arithmetic blend with a **weighted harmonic mean**; because the harmonic mean requires **positive** inputs, we apply a small **shift**:

- Shift:
\[
s \;=\; \min\{Q_t(s_t,a_t),\; y_t,\; 0\}\;-\;\varepsilon_{\text{shift}}, \quad \varepsilon_{\text{shift}}>0.
\]
- Shifted positives $Q^\sim = Q_t - s$, $y^\sim = y_t - s$.
- Harmonic blend in shifted space:
\[
Q^{\sim}_{t+1} \;=\;
\left(\frac{1-\alpha}{Q^\sim} \;+\; \frac{\alpha}{y^\sim}\right)^{-1}.
\]
- Shift back:
\[
Q_{t+1} \;=\; Q^{\sim}_{t+1} + s.
\]

**Guards used in code:** small $\varepsilon_{\text{div}}>0$ in denominators; small $\varepsilon_{\text{shift}}>0$ for the shift.

> For $x,y>0$, $H(x,y)\le A(x,y)$ (harmonic ≤ arithmetic), so the harmonic update is **more conservative** against upward spikes in $y_t$ (common with bootstrapping), often stabilizing learning in inertial/cliffy dynamics.

### 1.4 Behavior policy: $\epsilon$-greedy (linear decay)
At state $s$:
\[
a \sim
\begin{cases}
\text{Uniform}(\mathcal{A}) & \text{with prob. } \epsilon\\
\arg\max_a Q(s,a) & \text{with prob. } 1-\epsilon
\end{cases}
\]
$\epsilon$ decays linearly from `eps_start` to `eps_end` over `eps_decay_episodes`.

---

## 2) Environments

### 2.1 VelocityGrid-v0 (custom inertial gridworld)
- **State:** $(x,y,v_x,v_y)$ (discrete), encoded to a single integer.
- **Actions (accelerations):**
  - `0` = noop $(a_x,a_y)=(0,0)$ → **coast** with current velocity
  - `1` = up $(0,-1)$
  - `2` = down $(0,+1)$
  - `3` = left $(-1,0)$
  - `4` = right $(+1,0)$
- **Dynamics:**
\[
\begin{aligned}
v_x' &= \mathrm{clip}(v_x + a_x,\; -v_{\max},\, +v_{\max}),\\
v_y' &= \mathrm{clip}(v_y + a_y,\; -v_{\max},\, +v_{\max}),\\
x'   &= \mathrm{clip}(x + v_x',\; 0,\, W-1),\\
y'   &= \mathrm{clip}(y + v_y',\; 0,\, H-1).
\end{aligned}
\]
- **Noise:** small probability of random accel on any axis you didn’t press (drift).
- **Rewards:** step $-1$, goal $+100$, cliff $-100$.
- **Termination:** goal or cliff; **truncation** at `max_steps`.
- **Make inertia stronger:** set `max_speed=2` in `envs/velocity_grid.py` (or wire a flag).

### 2.2 Built-in toy-text (Gymnasium)
- **CliffWalking-v0, FrozenLake-v1**: actions are **moves**, not accelerations.
  - `0=left, 1=down, 2=right, 3=up`

---

## 3) Metrics & Plots

### 3.1 Per-episode **Return**
\[
G^{(ep)} \;=\; \sum_{t=0}^{T-1} r_t.
\]
Logged to:
```
logs/<variant>/s<seed>/<variant>_returns.csv
```

### 3.2 Per-episode **Steps to finish**
Number of env steps until **termination** (goal/cliff) or **truncation** (max-steps).
Logged to:
```
logs/<variant>/s<seed>/<variant>_steps.csv
```

### 3.3 Plots (smoothed only)
- Per-seed **returns** and **steps** (moving average window = `--smooth`).
- Cross-seed comparison (**mean ± standard error**) for returns and steps.

Outputs:
```
plots/<variant>/learning_curve_s<seed>.png        # smoothed returns
plots/<variant>/steps_curve_s<seed>.png          # smoothed steps
plots/compare/<TAG>_mean_se_returns.png          # cross-seed returns
plots/compare/<TAG>_mean_se_steps.png            # cross-seed steps
```

---

## 4) Install

```bash
pip install "gymnasium[toy-text]" pygame numpy<2 matplotlib imageio imageio-ffmpeg
# If needed for encoding: conda install -c conda-forge ffmpeg
```

> If you see `ImportError: numpy.core.multiarray failed to import`, reinstall NumPy via conda-forge inside your env:
> `conda install -c conda-forge "numpy<2"`

---

## 5) Run

Train both variants on the inertial env (recommended), record videos, plot, compare:

```bash
python run.py --env VelocityGrid-v0 --variants both --seeds 0-4   --episodes 4000 --record_video --task all
```

Use a built-in env:

```bash
python run.py --env CliffWalking-v0 --variants both --seeds 0-4 --episodes 3000 --task all
```

**Common flags**
- Variants: `--variants {standard|harmonic|both}`
- Seeds: `--seeds "0-4"` or `"0,1,2"`
- Training: `--episodes 4000` `--alpha 0.2` `--gamma 0.99`
- Exploration: `--eps_start 1.0` `--eps_end 0.05` `--eps_decay_episodes 3500`
- Plots: `--smooth 151` (MA window)
- Tasks: `--task {train|plot|compare|all}`
- Video (optional): `--record_video --video_max_steps 300`
- Roots: `--log_root logs` `--plots_root plots` `--videos_root videos`

---

## 6) What’s Where (code pointers)

- **Updates & training:** `hq/core.py`
  - `train_one(...)` implements standard & harmonic updates; choose via `--variants`.
- **VelocityGrid-v0:** `envs/velocity_grid.py`
  - Inertial dynamics + cliffs; registered on import.
- **Plots:** `hq/plotting.py`
  - Only **smoothed** curves saved; also cross-seed mean ± SE comparison.
- **Video:** `hq/video.py`
  - `RecordVideo` with `render_mode="rgb_array"` (skips if unsupported).

---

## 7) Why harmonic can help here

Bootstrapped targets $y_t = r + \gamma \max_a Q(s',a)$ can **spike** via optimistic chains. The arithmetic EMA update moves proportionally to spike size; the **harmonic mean** is bounded by the smaller operand and **dampens** large spikes, often reducing oscillations and cliff crashes in inertial tasks while still converging.

---

## 8) Pseudocode (both updates)

```
Initialize Q(s,a) = 0
for episode = 1..E:
  reset env → s
  for t = 0..max_steps-1:
    with prob ε: a ← random
    else:        a ← argmax_a Q(s,a)
    take a → (s', r, done)
    y ← r if done else r + γ * max_{a'} Q(s', a')
    if variant == "standard":
        Q(s,a) ← (1-α) Q(s,a) + α y
    else:  # harmonic
        sft ← min(Q(s,a), y, 0) − ε_shift
        q ← Q(s,a) − sft;  y' ← y − sft
        Q(s,a) ← sft + 1 / ( (1-α)/max(q,ε_div) + α/max(y',ε_div) )
    s ← s'
    if done: break
```

---

## 9) Tweaks

- **More inertia:** set `max_speed=2` in `envs/velocity_grid.py`.
- **Harder task:** increase `slip_prob`; lower `goal_reward`; raise `cliff_penalty`.
- **Success-only steps:** filter steps by episodes that reached the goal (easy to add).
