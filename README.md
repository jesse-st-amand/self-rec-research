# self-rec-research

Research experiments for **self-generated text recognition (SGTR)** in large language models. Can an LLM tell whether a piece of text was written by itself or by another model?

This repo is the experiment workspace. It contains evaluation configs, training configs, analysis scripts, and data pipelines. The core logic lives in two companion packages:

| Package | Purpose |
|---------|---------|
| [self-rec-framework](https://github.com/MARS-3-0-self-recognition/self-rec-framework) | Evaluation framework — generate data, run recognition evals, analyze results |
| [SGTR-RL](https://github.com/jesse-st-amand/SGTR-RL) | Training framework — SFT/RL fine-tuning for SGTR with LoRA |

## Setup

### Option A: Standard install (run experiments only)

Clone the repo and let `uv` install the framework packages from GitHub:

```bash
git clone https://github.com/jesse-st-amand/self-rec-research.git
cd self-rec-research
uv sync
cp .env.template .env  # fill in API keys
```

This pulls `self-rec-framework` and `sgtr-rl` as regular dependencies.

### Option B: Editable install (develop all three repos)

For contributors who want to modify the framework or training code alongside experiments:

```bash
git clone https://github.com/jesse-st-amand/self-rec-research.git
cd self-rec-research

# Clone companion repos as editable packages
mkdir -p _external
git clone https://github.com/MARS-3-0-self-recognition/self-rec-framework.git _external/self-rec-framework
git clone https://github.com/jesse-st-amand/SGTR-RL.git _external/SGTR-RL

# Install everything with editable links
uv sync
cp .env.template .env  # fill in API keys
```

With this setup, changes to `_external/self-rec-framework/` or `_external/SGTR-RL/` take effect immediately without reinstalling. The `pyproject.toml` uses `[tool.uv.sources]` to point at the local paths:

```toml
[tool.uv.sources]
self-rec-framework = { path = "_external/self-rec-framework", editable = true }
sgtr-rl = { path = "_external/SGTR-RL", editable = true }
```

**How the three repos coordinate:**

The `_external/` directory is gitignored by self-rec-research. Each repo under `_external/` is an independent git repository with its own remotes, branches, and history. There are no git submodules — just three separate repos that happen to share a Python environment.

This means you commit and push to each repo independently:

```bash
# Commit changes to the research repo
cd ~/projects/python/self-rec-research
git add experiments_eval/...
git commit -m "Add COLM_05 experiment configs"
git push origin training

# Commit changes to the framework
cd _external/self-rec-framework
git add self_rec_framework/...
git commit -m "Add score-distance analysis"
git push origin public

# Commit changes to the training package
cd _external/SGTR-RL
git add sgtr_rl/...
git commit -m "Fix LoRA checkpoint saving"
git push origin js/local_editable
```

When a feature spans multiple repos (e.g., adding a new analysis script to the framework and a new bash wrapper in the research repo), commit to both repos. The research repo doesn't track which version of each package it depends on beyond the pip package name — the assumption is that developers on the editable setup keep all three repos on compatible branches.

For Option A users who install from GitHub, the framework and SGTR-RL versions are pinned by `uv.lock`.

## Environment Variables

Copy `.env.template` to `.env` and fill in the keys you need:

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI/GPT models |
| `ANTHROPIC_API_KEY` | Claude models |
| `GOOGLE_API_KEY` | Gemini models |
| `TOGETHER_API_KEY` | Together AI models (DeepSeek, Qwen, Kimi, GLM, etc.) |
| `XAI_API_KEY` | Grok models |
| `FIREWORKS_API_KEY` | Fireworks AI models |
| `HF_TOKEN` | Gated model downloads (Llama, etc.) |
| `WANDB_API_KEY` | Training metrics logging |
| `RUNPOD_API_KEY` | RunPod GPU training |
| `RUNPOD_NETWORK_VOLUME_ID` | RunPod persistent storage |

You only need the keys for providers you plan to use.

## Project Structure

```
self-rec-research/
├── _external/                     # Companion packages (editable installs)
│   ├── self-rec-framework/        # Evaluation framework (srf-* commands)
│   └── SGTR-RL/                   # Training framework (sgtr-* commands)
│
├── experiments_eval/              # Evaluation experiment configs
│   ├── ICML/                      # ICML paper experiments (01-08)
│   ├── COLM/                      # COLM paper experiments (01-04)
│   ├── pre-ICML/                  # Earlier iterations
│   ├── _tutorials/                # Tutorial evaluation examples
│   └── 00_data_gen/               # Data generation configs
│
├── experiments_training/          # Training experiment configs
│   ├── _tutorial/                 # Minimal SFT examples (local + RunPod)
│   ├── _data_processing/          # Training data extraction configs
│   ├── COLM_Jesse/                # COLM training experiments
│   └── COLM_Callum/               # COLM training experiments
│
├── data/                          # All data (gitignored)
│   ├── input/                     # Source datasets
│   ├── results/                   # Evaluation outputs (.eval files)
│   ├── training_data/             # Extracted train/val JSONL
│   └── analysis/                  # Analysis outputs and aggregations
│
├── scripts/
│   ├── training/                  # Data preparation scripts
│   └── utils/                     # Figure generation, data upload
│
├── pyproject.toml
└── .env
```

## Running Evaluations

Each evaluation experiment has bash scripts for sweeps and analysis under its `bash/` directory.

### 1. Generate model outputs

```bash
bash experiments_eval/00_data_gen/bash/01_config_sweep/data_gen_sweep_sharegpt_english_26.sh
```

### 2. Run evaluation sweeps

```bash
bash experiments_eval/COLM/COLM_03_UT_PW-Q_Rec_NPr_CoT_Rsn/bash/sweep/06_sharegpt_english_26.sh
```

### 3. Run analysis

```bash
# Per-dataset analysis (accuracy, evaluator performance, disagreement)
bash experiments_eval/COLM/COLM_03_UT_PW-Q_Rec_NPr_CoT_Rsn/bash/analysis/run_all_analyses.sh

# Inter-dataset analysis (aggregation, rank distance, etc.)
bash experiments_eval/COLM/COLM_03_UT_PW-Q_Rec_NPr_CoT_Rsn/bash/analysis/_inter-dataset/03-rank-distance.sh
```

### 4. Compare experiments

```bash
bash experiments_eval/COLM/comparisons/COLM_04-vs-COLM_03/00-performance_contrast.sh
```

## Running Training

### Local GPU

```bash
# Extract training data from eval results
bash experiments_training/_data_processing/prepare.sh \
    experiments_training/_tutorial/prepare_data.yaml

# Train (uses local GPU)
bash experiments_training/_tutorial/run.sh
```

### RunPod (remote GPU)

```bash
# Dry run — prints the pod config without launching
bash experiments_training/_tutorial/run_runpod.sh

# Launch on RunPod
bash experiments_training/_tutorial/run_runpod.sh --launch
```

The RunPod launcher clones the repo onto a pod, installs dependencies, runs training, and tears down the pod when finished. Results are saved to the network volume.

## Experiment Naming

Experiments follow a structured naming convention:

```
COLM_03_UT_PW-Q_Rec_NPr_CoT_Rsn
│     │  │  │    │   │   │   └─ Reasoning: Rsn (CoT reasoning), Inst (instruct), DR (direct response)
│     │  │  │    │   │   └───── Output: CoT (chain-of-thought), FA (final answer)
│     │  │  │    │   └───────── Priming: NPr (not primed), Pr (primed)
│     │  │  │    └───────────── Task: Rec (recognition), Pref (preference)
│     │  │  └────────────────── Format-Interaction: PW-Q (pairwise query), IND-C (individual conversation)
│     │  └───────────────────── Tags: UT (user tags), AT (assistant tags)
│     └──────────────────────── Experiment number
└────────────────────────────── Paper/group
```

## CLI Commands

All commands are available via `uv run` after setup:

### Evaluation (srf-*)

| Command | Description |
|---------|-------------|
| `srf-generate` | Generate model outputs for a dataset |
| `srf-generate-sweep` | Generate outputs across multiple models |
| `srf-eval` | Run a single evaluation |
| `srf-eval-sweep` | Run evaluations across model pairs |
| `srf-recognition-accuracy` | Compute accuracy pivot tables and heatmaps |
| `srf-evaluator-performance` | Rank evaluator models by performance |
| `srf-rank-distance` | Analyze accuracy vs LM Arena rank distance |
| `srf-experiment-contrast` | Compare two experiments side-by-side |
| `srf-list-models` | List available models by provider |

### Training (sgtr-*)

| Command | Description |
|---------|-------------|
| `sgtr-train` | Run SFT/GRPO training |
| `sgtr-runpod-launch` | Launch training on RunPod |
| `sgtr-prepare-data` | Extract training data from eval results |
| `sgtr-plot-summary` | Generate per-run training summary plots |
| `sgtr-plot-cross-evals` | Generate cross-evaluation analysis |

Use `--help` with any command for full usage.
