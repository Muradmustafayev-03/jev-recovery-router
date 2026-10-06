# Jev Recovery Router

**[Model package and documentation on Hugging Face](https://huggingface.co/Quicksort-fr/jev-recovery-router)** · **[GitHub source](https://github.com/Muradmustafayev-03/jev-recovery-router)**

A research implementation of cost-sensitive intent routing: **commit, retrieve more candidates, or hand off**. A Qwen3-1.7B belief model scores retrieved intent candidates plus `NONE`; a separate small policy decides whether another retrieval is worth its cost.

This repository contains source, tests, configurations and protocol documentation. Selected weights and model documentation are published separately at [Quicksort-fr/jev-recovery-router](https://huggingface.co/Quicksort-fr/jev-recovery-router). Generated datasets, benchmark outputs, checkpoints and caches are not part of the code repository. This is an independent Jev-inspired experiment, not an official TypeSafe project or a reproduction of its proprietary RLCD training.

## Install

Requires **Python 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,reports]"
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate `.venv\Scripts\Activate.ps1`. GPU experiments were run with PyTorch 2.7.1+cu118 and BF16 on an A100 80 GB. For that CUDA wheel, install it before the editable package:

```bash
python -m pip install "torch==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118
```

## Load the selected model

Download the separate [Hugging Face model package](https://huggingface.co/Quicksort-fr/jev-recovery-router) into an ignored local directory:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Quicksort-fr/jev-recovery-router', local_dir='artifacts/model')"
```

Run inference using the source installation:

```bash
python -m recovery_router.inference \
  --model artifacts/model/model \
  --policy artifacts/model/policy \
  --calibration artifacts/model/calibration.json \
  --data-dir artifacts/model/data/processed \
  --catalog banking --device cuda \
  --text "My card was charged twice"
```

Use `--device cpu` for CPU execution. Other catalog modes are `clinc_base` and `clinc_updated`. This is a **custom LoRA, decision-head and actor package**, not an `AutoModel.from_pretrained` endpoint. Full Qwen and MiniLM backbone weights are referenced by pinned revisions and may download on first use; the package does not duplicate them. Local inference requires no Jev API key.

The trained schedule is **1 initial candidate → 5 after expansion**. The second retrieval/model pass runs only when selected by the actor. `NONE` retains probability mass rather than being removed and renormalized. Change policy costs with `--wrong-cost`, `--defer-cost` and `--expand-cost`; these do not change the belief model's probabilities. A handoff remains unresolved; candidate confidence is not a guarantee of correct routing. Capacity for twenty candidate positions does not imply trained support beyond 1/5.

## What was measured

V2 executed twelve belief-training epochs / 5,532 updates and selected **epoch seven / step 3,227** using tuning loss. Belief training uses supervised cross-entropy; the separate recovery actor uses REINFORCE. The following natural-test results use each provider's own calibrated forecasts and prespecified seed-42 actor.

| Population | Queries | Selected V2 utility | Jev 1.13.0 utility |
|---|---:|---:|---:|
| CLINC, 120-intent catalog | 4,600 | 0.553891 | 0.394065 |
| CLINC, enlarged 150-intent catalog | 5,500 | 0.497082 | 0.394382 |
| BANKING77 | 3,080 | -0.001672 | 0.346055 |

Utility uses **correct +1, wrong −5, handoff −0.2, expansion −0.05**. These are simulated units, not money or measured human-review outcomes. The two CLINC populations overlap.

V2 exceeded its own CLINC-120 confidence gate by **+0.058946**, with a descriptive paired 95% interval **[+0.040585, +0.078435]**. That gain was concentrated in out-of-scope rejection; the gate performed better on naturally recoverable cases. BANKING77 remained poor: V2 never-expand utility was **0.308052**, well above its RL policy. Exact policy optimization remained competitive, and the fixed-cost actor performed better on the primary CLINC endpoint. Neither small online timing sample showed an adaptive speed advantage.

V2 was designed after inspecting the original two-epoch results, and changed schedule, augmentation and checkpoint selection together. Its improvements are **exploratory**, not evidence that more epochs alone caused them. V1's primary gate comparison remained inconclusive. These complete-system comparisons do not establish general superiority over Jev, sampled-RL necessity or industrial ROI. Jev's bounded validation retries, including a disclosed post-seal exception, remain documented with the model evidence. See the [model card](MODEL_CARD.md) and the separate model package for measured results and limitations.

## Reproduce and test

Tests use small generated fixtures; downloaded weights and benchmark data are not required for the unit suite:

```bash
python -m pytest -q --basetemp .pytest-tmp-public
```

To generate the pinned public benchmark inputs and run the A100 experiments on a suitable Linux CUDA host:

```bash
python scripts/prepare_data.py --backend minilm
python scripts/run_experiment.py --config configs/a100.json --output runs/a100-001 --stress --dry-run
bash scripts/run_a100.sh --stress
python scripts/run_extended.py --data-dir data/processed --baseline-run runs/a100-001 --output runs/a100-extended-001
```

Data preparation downloads public source data and the pinned retriever. Training downloads the pinned backbone and writes local `runs/` outputs. The extended run requires the completed baseline, including its comparator evidence. Configured time/cost limits are assumptions for the runner; they do not stop cloud-provider billing.

Read the [data setup](docs/DATA.md), [research protocol](docs/RESEARCH_PROTOCOL.md), [design decisions](docs/DESIGN_DECISIONS.md) and [extended protocol](docs/EXTENDED_PROTOCOL.md). The extended protocol's exact bytes are checked against its configured hash. Optional paid Jev reproduction is separate and requires explicit activation and a key; see [Jev comparison setup](docs/JEV_COMPARISON.md).

## Licensing

Code and owned artifacts: [Apache-2.0](LICENSE). CLINC150 text: CC BY 3.0; BANKING77 text: CC BY 4.0. Backbone, retriever and dataset attributions remain separate; see [NOTICE.md](NOTICE.md). BANKING77 transfer uses labeled training exemplars in its catalog, and foundation-model pretraining overlap is unknown.
