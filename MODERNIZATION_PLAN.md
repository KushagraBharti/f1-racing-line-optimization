# Modernization Plan

This document outlines the end-to-end roadmap for rebuilding the F1 Racing Line Optimization project with a contemporary,
maintainable technology stack. It assumes access to a CUDA-capable GPU but maintains CPU fallbacks.

---

## Phase 0 – Assessment & Foundations

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Baseline audit | Profile legacy scripts, capture FPS, episode rewards, sensor noise, and failure cases. | Benchmark report, representative rollout traces. |
| Asset inventory | Catalog sprites, track contours, checkpoint definitions, and reward constants. | Asset manifest with licensing notes. |
| Tooling bootstrap | Introduce `pyproject.toml`, configure Poetry or `uv`, set up pre-commit hooks (`black`, `ruff`, `mypy`), and initialize GitHub Actions CI. | Modern project scaffold with automated formatting/type checks. |

**Key Decisions**
- Target Python **3.11** for improved performance and typing features.
- Adopt `poetry` (or `uv`) for dependency management while maintaining a `requirements.txt` export for compatibility.

---

## Phase 1 – Environment Architecture Redesign

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Modular package layout | Create `src/` packages: `f1_env`, `f1_assets`, `f1_training`, `f1_tools`. | Namespaced modules with typed interfaces. |
| Configuration schema | Implement Hydra/OMEGACONF config hierarchy (`env`, `vehicle`, `track`, `reward`, `training`). | Versioned configs, default overrides, CLI entry points. |
| Data contracts | Define observation/reward/telemetry models with Pydantic v2 for validation and serialization. | Contract tests ensuring schema stability. |

**Libraries**: `hydra-core`, `omegaconf`, `pydantic`, `typer` for CLI wrappers.

---

## Phase 2 – Simulation Core (Physics, Sensors, Rewards)

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Physics engine integration | Model car chassis, tires, and contact with `pymunk`; encapsulate parameters (mass, drag, grip) in configs. | Deterministic physics step with unit tests and property-based checks. |
| Sensor suite redesign | Implement raycast-based distance sensors via `pymunk.SegmentQuery`, with configurable counts/angles and GPU-accelerated batching via `torch` if needed. | Sensor module exposing normalized observations with latency benchmarks. |
| Reward & termination refactor | Provide pluggable reward components (progress, smoothness, collision penalties) and termination criteria (lap completion, off-track). | Reward registry with test coverage and regression baselines vs. legacy metrics. |

**Libraries**: `pymunk`, `numpy`, `torch`, `shapely`, `scipy`.

---

## Phase 3 – Rendering & User Experience

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Renderer overhaul | Migrate to `pygame-ce` for windowing, integrate `moderngl` for GPU-accelerated layers, and decouple simulation from render loop. | Renderer package supporting windowed, headless, and off-screen modes. |
| Asset pipeline | Convert track assets to vector formats (SVG/SDF) or procedural splines; support dynamic theming and resolution scaling. | Asset loader with caching, CLI to import new tracks. |
| Telemetry overlays | Render HUD with speed, lap times, sensor readings; integrate screenshot/video capture via `imageio[ffmpeg]`. | Debug overlay module with tests for HUD placement and value ranges. |

**Libraries**: `pygame-ce`, `moderngl`, `imageio[ffmpeg]`.

---

## Phase 4 – Data, Analytics, and Tooling

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Trajectory storage | Store rollouts in `xarray` datasets or `polars` DataFrames with metadata (seed, config hash). | Serialization utilities exporting Parquet/Arrow + schema docs. |
| Replay & analytics apps | Build Streamlit dashboards for lap comparison, sensor traces, and reward breakdowns; optional Plotly animations. | `streamlit/` app with deployment instructions. |
| Experiment tracking | Standardize Weights & Biases logging for hyperparameters, metrics, videos, and artifacts. | Reusable logging callbacks, project templates. |

**Libraries**: `xarray`, `polars`, `streamlit`, `plotly`, `wandb`.

---

## Phase 5 – RL Pipelines & Curriculum

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Gymnasium compliance | Implement `Env` and `VectorEnv` APIs (`reset` returning `(obs, info)`, `step` returning `(obs, reward, terminated, truncated, info)`). | Environment package published (optionally) to PyPI for reuse. |
| Baseline algorithms | Provide Hydra-configured Stable-Baselines3 training scripts (PPO, SAC, TD3) with evaluation harnesses and `SyncVectorEnv` support. | `scripts/train.py`, `scripts/eval.py`, configuration presets, CI smoke tests. |
| Ray RLlib integration | Update RLlib workflows using `AlgorithmConfig`, multi-GPU rollout workers, and curriculum/randomization wrappers. | Modern Ray training module with launch instructions. |
| Curriculum learning | Implement domain randomization (weather, grip, sensor noise) and multi-track training schedules. | Curriculum configs, experiment reports comparing generalization. |

**Libraries**: `gymnasium`, `stable-baselines3`, `ray[rllib]`, `torch`.

---

## Phase 6 – Quality Assurance & DevOps

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Testing strategy | Combine `pytest`, `pytest-benchmark`, and `pytest-xdist` for unit, integration, and performance regression tests. | Test suite with coverage targets and baseline benchmarks. |
| Static analysis | Enforce `ruff`, `black`, and `mypy` via pre-commit and CI; add docstring linting and import sorting. | Passing CI status, contributor guidelines. |
| Documentation | Generate MkDocs or Sphinx docs covering architecture, API reference, and tutorials. | Published documentation site linked from README. |

**Libraries**: `pytest`, `pytest-benchmark`, `pytest-xdist`, `black`, `ruff`, `mypy`.

---

## Phase 7 – Release & Expansion

| Goal | Tasks | Deliverables |
| --- | --- | --- |
| Packaging & versioning | Adopt semantic versioning, publish release notes, and (optionally) distribute wheels to PyPI. | Automated release pipeline with changelog generation. |
| Community extensions | Explore multi-agent racing via PettingZoo, imitation learning from telemetry datasets, and esports-style leaderboards. | Roadmap addendum, issue templates for contributions. |
| Maintenance | Schedule dependency upgrades, CI infrastructure reviews, and benchmark recalibration every quarter. | Maintenance checklist and ownership assignments. |

---

## Risk Mitigation & Research Items

- **Physics fidelity** – Validate `pymunk` tire models; investigate hybrid approaches (e.g., Pacejka curves) if higher realism is
  required.
- **Performance** – Benchmark rendering and simulation throughput on CPU vs. GPU; leverage `torch.compile` for policy inference.
- **Data volume** – Ensure trajectory logging remains efficient; consider chunked Parquet and compression strategies.
- **Licensing** – Replace legacy assets with CC0 or self-generated resources to avoid redistribution issues.

---

## Contribution On-Ramp

1. Set up the environment per the README instructions, installing CUDA-enabled PyTorch if applicable.
2. Run the legacy scripts to understand baseline behavior and collect comparison data.
3. Pick a phase from this plan, fork the repository, and coordinate implementation steps via issues/PRs.
4. Maintain alignment with the chosen library versions and update the roadmap when significant architecture decisions change.

This roadmap should serve as the single source of truth for the modernization effort. Revisit and revise it at the conclusion of
each phase to incorporate lessons learned and newly available tooling.
