# TRON1A Bipedal Locomotion RL Framework

Reinforcement learning training and evaluation framework for the [Limx Dynamics](https://www.limxdynamics.com/en) TRON1A bipedal robot family. Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl), with a fully Dockerized environment managed through the `djinn` CLI.

For a detailed description of the software architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Documentation

The workspace keeps its two architecture documents at the root and organises everything else into two indexed directories.

| Location | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Software architecture, task registry, training and evaluation pipelines, `djinn` reference |
| [CO_OPTIMISATION.md](CO_OPTIMISATION.md) | The design and controller co-optimisation architecture in full |
| [context/](context/README.md) | Established facts, codebase investigations, experiment records, literature survey |
| [context/artefacts/](context/artefacts/README.md) | Debugging artefacts, the exported plots and dashboards cited as evidence |
| [plans/](plans/README.md) | Designs and implementation plans, each carrying its implementation status |
| [tron1-rl-isaaclab-cozum/context/](tron1-rl-isaaclab-cozum/context/README.md) | Facts specific to the simulation repository, robot parameterisation and algorithm mathematics |
| [tron1-rl-isaaclab-cozum/plans/](tron1-rl-isaaclab-cozum/plans/README.md) | Plans confined to the simulation repository, presently empty |

Every one of those directories carries a `README.md` registering its contents with a summary, a revision date, and a currency or implementation status, so that a reader may determine which documents are relevant without explicitly reading any of them. Start from [context/README.md](context/README.md) for facts about how the system behaves, and [plans/README.md](plans/README.md) for work proposed or completed. The conventions governing changes to code and to documents, including where a debugging artefact must be placed, are in [CLAUDE.md](CLAUDE.md).

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
djinn exec lab bash 
pip install -e biped/exts/bipedal_locomotion biped/himloco biped/co_optimisation
pip install plotly dash openpyxl
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

# Start Tensorboard
djinn exec lab bash
tensorboard --log_dir <path>
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

---

## Changelog Generation

`changelog_creator.py` parses the last N commits from a specified branch and produces a `changelog_<timestamp>.log` markdown file grouped into **Problems**, **Solutions**, and **Notes** sections. To ensure commits are consistently formatted for parsing, set up a `.gitmessage` template in each repository:

```bash
git config commit.template .gitmessage
```

### Commit Message Format

```
Problem
=======
1. Description of the problem

Solution
========
1. Description of the solution

Note
====
1. Any additional notes
```

Bullet points can span multiple lines — indent continuation lines with three spaces.

### Prerequisites

```bash
pip install gitpython
```

### Usage

```bash
python changelog_creator.py [--count N] [--branch BRANCH] [--repo-url URL] [--local-path PATH]
```

| Argument | Default | Description |
|---|---|---|
| `--count` | `5` | Number of recent commits to parse |
| `--branch` | `shreyas/design_coptimisation_updates` | Branch to parse commits from |
| `--repo-url` | SSH URL (see script) | SSH/HTTPS URL of the repository |
| `--local-path` | `/ws/tron1-rl-isaaclab-cozum/` | Local path to clone/open the repository |

### Example

```bash
# Parse the last 10 commits from the default branch
python changelog_creator.py --count 10

# Parse from a specific branch
python changelog_creator.py --count 5 --branch main
```
