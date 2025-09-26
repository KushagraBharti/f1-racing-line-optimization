# F1 Racing Line Optimization (Modernization Roadmap)

This repository contains the legacy implementation of an F1 racing-line optimization project that combines a custom OpenAI Gym
environment, Pygame rendering, and Ray RLlib training utilities. The current code base targets Python 3.8/3.9-era libraries and
requires a ground-up modernization effort to benefit from the latest reinforcement learning, simulation, and tooling ecosystems.

The documentation below captures the **future-facing plan** for rebuilding the project with a contemporary stack while preserving
the project’s core goals. It also summarizes how to run the existing scripts until the modernization is complete.

![Race environment](media/rlo.gif)

---

## Modernization Goals

1. **Adopt maintained, feature-rich libraries** that align with modern Gymnasium APIs, GPU-enabled learning, and high-fidelity 2D
   physics.
2. **Modularize the code base** into clear simulation, training, visualization, and tooling packages with typed configurations and
testable components.
3. **Improve developer experience** with reproducible environments, automated quality gates, structured experiment tracking, and
   reproducible dataset exports.
4. **Deliver richer analytics and UX** for comparing agents, visualizing telemetry, and authoring new tracks or curricula.

---

## Target Technology Stack

| Capability | Selected Library | Rationale |
| --- | --- | --- |
| RL Environment API | `gymnasium>=0.29` | Successor to classic Gym with active maintenance and vectorized API support. |
| RL Algorithms | `stable-baselines3>=2.3`, `ray[rllib]>=2.9` | Provides state-of-the-art on-policy/off-policy agents and large-scale experimentation. |
| Deep Learning Backend | `torch>=2.1` (CUDA enabled) | PyTorch 2.x compiler support, strong RL ecosystem, GPU acceleration. |
| Physics Engine | `pymunk>=6.6` | 2D rigid-body dynamics with good documentation and Python-first API. |
| Geometry Utilities | `shapely>=2.0`, `scipy>=1.11` | Robust collision checks, spline interpolation, and computational geometry. |
| Rendering | `pygame-ce>=2.5`, `moderngl>=5.8`, `imageio[ffmpeg]>=2.33` | Maintained SDL bindings, optional GPU pipeline, frame/video export. |
| Configuration | `hydra-core>=1.3`, `omegaconf>=2.3` | Hierarchical configuration management and experiment sweeping. |
| Data Modeling | `pydantic>=2.5`, `numpy>=1.26`, `polars>=0.20`, `xarray>=2023.9` | Typed configs, efficient analytics, labeled trajectory storage. |
| Experiment Tracking | `wandb>=0.16` | Cloud/on-prem experiment logging, artifact storage, metric comparisons. |
| Visualization & Dashboards | `plotly>=5.18`, `streamlit>=1.29` | Interactive telemetry dashboards and replay tools. |
| CLI & Tooling | `typer>=0.9`, `pytest>=7.4`, `pytest-benchmark>=4.0`, `pytest-xdist>=3.4`, `black>=23.11`, `ruff>=0.1.7`, `mypy>=1.7` | Productive developer workflows, QA automation, parallel testing. |

The corresponding dependency pins are recorded in [`requirements.txt`](requirements.txt).

---

## Modernization Blueprint

A high-level summary is provided here for quick reference. Refer to [`MODERNIZATION_PLAN.md`](MODERNIZATION_PLAN.md) for the
detailed, phase-by-phase execution plan.

1. **Repository Bootstrap** – Create a modern Python package layout, adopt Poetry or `uv` for dependency management, configure
   pre-commit hooks, and scaffold CI with lint, type-check, and test jobs.
2. **Simulation Core Rewrite** – Replace bespoke kinematics with a `pymunk`-powered vehicle model, redesign sensors using physics
   raycasts, and migrate to Gymnasium-compatible observation/reward interfaces.
3. **Rendering Overhaul** – Rebuild the renderer with `pygame-ce` + `moderngl`, separating simulation step rate from FPS and adding
   headless/off-screen rendering.
4. **Data & Analytics Layer** – Standardize trajectory exports via `xarray` or `polars`, introduce schema validation with
   `pydantic`, and provide notebook/Dash apps for replay and telemetry analysis.
5. **Training & Experimentation** – Ship Stable-Baselines3 pipelines with Hydra-configured hyperparameters, update Ray RLlib
   training scripts, and integrate Weights & Biases logging.
6. **Tooling & Documentation** – Expand unit/integration tests, document APIs and configs, and publish tutorials for creating new
   tracks, running curriculum learning, and benchmarking agents.

---

## Setting Up a Development Environment (Windows + Conda)

The commands below allow you to experiment with the **current** implementation while preparing the environment for the planned
upgrade. Adjust the CUDA tooling steps to match your GPU driver stack.

```powershell
# 1. Clone the repository
git clone https://github.com/KGolemo/f1-racing-line-optimization.git
cd f1-racing-line-optimization

# 2. Create an isolated environment targeting Python 3.11+
conda create --name f1-rlo python=3.11 -y

# 3. Activate the environment
conda activate f1-rlo

# 4. (Optional) Install CUDA-enabled PyTorch that matches your driver stack
# Example for CUDA 12.1 users; consult https://pytorch.org/get-started/locally/ for alternatives
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 5. Install project dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. (Optional) Enable pre-commit hooks once modernization lands
# pre-commit install
```

> **Tip:** If you prefer Poetry or `uv`, replicate the dependency list from `requirements.txt` in your chosen tool and ensure the
> lock file is committed for reproducibility.

---

## Running Legacy Scripts

Until the modernization refactor is delivered, the legacy Python scripts remain functional and can be invoked as follows:

| Script | Purpose | Command |
| --- | --- | --- |
| Manual control demo | Drive the car with keyboard input | `python manual_mode.py` |
| Ray RLlib training | Launch legacy SAC training | `python ray_training.py` |
| Ray rollout utility | Evaluate a saved checkpoint | `python ray_rollout.py` |
| SAC inference demo | Restore a checkpoint for interactive play | `python use_agent.py` |
| Trajectory replay | Visualize exported `.npy` trajectories | `python render_agents.py` |

> These scripts depend on legacy APIs (classic Gym, original Pygame). Refactoring them to the new stack is covered in the
> modernization plan.

---

## Learning Path for New Contributors

1. **Review the Legacy Code** – Start with `race.py`, `car.py`, and `track.py` to understand the environment flow and identify pain
   points the modernization targets.
2. **Study the Modernization Plan** – Read [`MODERNIZATION_PLAN.md`](MODERNIZATION_PLAN.md) to see how each subsystem will evolve
   and where you can contribute.
3. **Prototype New Components** – Begin by porting isolated features (e.g., sensor raycasts with `pymunk`, Hydra config schemas)
   before integrating them into the full environment.
4. **Establish Quality Gates** – Help configure `pytest`, `black`, `ruff`, and `mypy` in CI to safeguard the reimplementation.
5. **Experiment with RL Baselines** – Use Stable-Baselines3 to train quick PPO/SAC agents once the Gymnasium environment is ready,
   logging results to Weights & Biases for comparison.

Happy racing—and happy refactoring!
=======
# F1 Racing Line Optimization

This project applies reinforcement learning to optimize the racing line of a Formula 1 car. It combines a custom Gym environment, a physics-based vehicle model, and Ray RLlib training utilities to learn control policies that complete the Monza circuit efficiently.

![Race environment](media/rlo.gif)

## Project Layout

| File | Description |
| --- | --- |
| `race.py` | Gym-compatible environment that connects the car physics, track geometry, rendering, and reward logic. |
| `car.py` | Vehicle dynamics, sensor model, reward bookkeeping, and collision detection. |
| `track.py` | Generates track boundaries and checkpoint lines from the Monza contour assets. |
| `utils.py` | Helper utilities for geometry math and rendering. |
| `manual_mode.py` | Keyboard-controlled driving for debugging and collecting human demonstrations. |
| `render_agents.py` | Replays previously exported trajectories as “ghost” cars. |
| `ray_training.py` | Launches Ray Tune to train a Soft Actor-Critic (SAC) agent. |
| `ray_rollout.py` | Evaluates saved checkpoints using RLlib’s rollout tooling. |
| `use_agent.py` | Restores a trained SAC agent and runs it inside the environment. |
| `exported_states/` | Sample trajectory `.npy` files produced during rollouts. |
| `imgs/` & `media/` | Track background, car sprites, and demo GIFs for rendering. |

## Prerequisites

* Windows 10/11 with [Anaconda](https://www.anaconda.com/products/distribution) installed.
* (Optional but recommended) A GPU and the appropriate CUDA/cuDNN drivers for Ray RLlib training workloads.
* Git for cloning the repository.

## Environment Setup (Windows + Conda)

```powershell
# 1. Clone the repository
git clone https://github.com/KGolemo/f1-racing-line-optimization.git
cd f1-racing-line-optimization

# 2. Create a dedicated conda environment (adjust Python version if needed)
conda create --name f1-rlo python=3.9

# 3. Activate the environment
conda activate f1-rlo

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. (Optional) Verify pygame can open a window
python -m pygame.examples.aliens
```

> **Note:** RLlib installs both TensorFlow and PyTorch support by default. On systems without GPUs you can still run the scripts, but training will be significantly slower.

## Running the Applications

### Manual Driving Demo

```powershell
python manual_mode.py
```

*Steer with the arrow keys (Up/Down for acceleration, Left/Right for steering). Press `R` to reset the car.*

### Replay Saved Trajectories

```powershell
python render_agents.py
```

Loads `.npy` files from `exported_states/` and renders them as spectator cars. You can place your own exports in this folder.

### Evaluate a Trained Agent

1. Update the `checkpoint_path` in `use_agent.py` to point to your RLlib checkpoint.
2. Run:
   ```powershell
   python use_agent.py
   ```

### Train an Agent with Ray Tune

```powershell
python ray_training.py
```

*Adjust GPU/CPU counts in the script to match your hardware. Training produces checkpoints under `ray_results/`.*

### Batch Rollout of Checkpoints

```powershell
python ray_rollout.py
```

`ray_rollout.py` uses RLlib’s evaluation CLI to run multiple episodes from a saved checkpoint. Edit the `checkpoint_path` string and `env_config` options to suit your run.

## Exporting Data

Set `"export_states": True` (and optionally `"export_frames": True`) in `env_config` dictionaries to dump numpy arrays or rendered frames for later analysis. Files are written to `exported_states/` with the prefix defined by `"export_string"`.

## Next Steps

* Experiment with different `env_config` settings in `race.py` to tune reward shaping and sensor layouts.
* Swap in custom track images in `imgs/` and regenerate checkpoints through `track.py`.
* Extend `ray_training.py` with alternative RLlib algorithms or custom callbacks for logging.

Happy racing!
