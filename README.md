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
