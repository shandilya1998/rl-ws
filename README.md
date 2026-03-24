# TRON1A Bipedal Locomotion RL Framework

Reinforcement learning training and evaluation framework for the [Limx Dynamics](https://www.limxdynamics.com/en) TRON1A bipedal robot family. Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl), with a fully Dockerized environment managed through the `djinn` CLI.

For a detailed description of the software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Prerequisites

The following must be installed on the **host machine** before proceeding:

- NVIDIA GPU with CUDA support
- NVIDIA drivers (recommended: 535+)
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Git (for submodule management)

No Python, Isaac Sim, or Isaac Lab installation is required on the host. All simulation and training runs inside the Docker container.

---

## Installation & Workspace Setup

### 1. Clone the repository

```bash
git clone <repo-url> tron1-rl-isaaclab-cozum
cd tron1-rl-isaaclab-cozum
```

### 2. Pull submodules

`IsaacLab` and `rsl_rl` are included as git submodules pinned to tested commits. Pull them with:

```bash
git submodule update --init --recursive
```

> **Note:** The submodules provide code references used inside the container. Do not modify them directly.

### 3. Set `PROJECT_WS`

`PROJECT_WS` must point to the **workspace root** — the directory that contains `djinn`, `tron1-rl-isaaclab-cozum/`, `docker/`, and the submodules.

```bash
export PROJECT_WS=/path/to/workspace
```

Add to your shell profile to persist across sessions:

```bash
echo 'export PROJECT_WS=/path/to/workspace' >> ~/.bashrc
```

### 4. Add `djinn` to `PATH`

`djinn` is the CLI entry point for all workspace operations. Adding `PROJECT_WS` to `PATH` makes it available system-wide:

```bash
export PATH=$PROJECT_WS:$PATH
```

Persist this alongside `PROJECT_WS`:

```bash
echo 'export PATH=$PROJECT_WS:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify with:

```bash
which djinn   # should print: /path/to/workspace/djinn
```

### 5. Build or pull the Docker image

```bash
djinn build docker isaaclab
```

### 6. Start the container

```bash
djinn up lab
```

### 7. Install `bipedal_locomotion` inside the container

```bash
djinn exec lab "pip install -e biped/exts/bipedal_locomotion biped/himloco"
djinn exec lab "pip install plotly dash openpyxl"
```

### 8. Verify

```bash
djinn ps    # should show isaac-lab-base as running
```

---

## Quick Start

End-to-end example after installation:

```bash
# Start the container
djinn up lab

# Train (base mode, GPU 0) — runs ~30k iterations, saves checkpoints every 500
djinn start train base 0

# Evaluate a checkpoint
djinn start play base logs/rsl_rl/sf_tron_1a_flat/2024-01-01_12-00-00/model_15000.pt 42 0

# Batch evaluate all experiments in logs/rsl_rl/test/
djinn start evaluation 42 0

# View interactive results dashboard
djinn start visualise-evaluation
```

See [ARCHITECTURE.md §7](ARCHITECTURE.md#djinn-automation-interface) for more information regarding the djinn interface.

---

## Task ID Reference

| Task ID | Use Case | `djinn` Mode |
|---|---|---|
| `Isaac-Limx-SF-Identified-Blind-Rough-v0` | Primary training — blind, rough terrain | `train base` |
| `Isaac-Limx-SF-Identified-Berkeley-v0` | Berkeley-style training | `train berkeley` |
| `Isaac-Limx-SF-Berkeley-HIM-v0` | HIM policy training | `train him` |
| `Isaac-Limx-SF-Identified-Blind-Rough-Play-v0` | Evaluate base checkpoint | `play base` |
| `Isaac-Limx-SF-Identified-Berkeley-Play-v0` | Evaluate Berkeley checkpoint | `play berkeley` |
| `Isaac-Limx-SF-Blind-Flat-v0` | Batch evaluation baseline | `evaluation` |

For the full task registry including PF and WF variants, see [ARCHITECTURE.md §8](ARCHITECTURE.md#8-complete-task-registry).

---

## Troubleshooting

**`djinn: command not found`**
Ensure `PROJECT_WS` is set and `$PROJECT_WS` is on your `PATH`. Run `source ~/.bashrc` after editing your shell profile.

**`djinn up lab` fails — container name conflict**
A container named `isaac-lab-base` already exists (stopped). Run `djinn down lab` first, then `djinn up lab`.

**GPU not detected inside container**
Verify the NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`. If this fails, reinstall the toolkit.

**Checkpoint not found in `djinn start play`**
The checkpoint path is relative to `/workspace/isaaclab/logs/rsl_rl/` inside the container. Use `djinn exec lab "ls logs/rsl_rl"` to browse available experiments.
