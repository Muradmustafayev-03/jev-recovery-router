# Model documentation

The selected trained weights and canonical model card are published at [Quicksort-fr/jev-recovery-router](https://huggingface.co/Quicksort-fr/jev-recovery-router). The [README](README.md) documents installing this source package and downloading those artifacts.

The package uses a custom Qwen3-1.7B LoRA adapter, typed decision head, actor and calibration. Belief training is supervised; only the recovery actor uses sampled RL. It is not an AutoModel text-generation endpoint. Supported candidate counts are 1 and 5.

The 12-epoch exploratory run selected epoch seven by validation. Its CLINC gains primarily concern out-of-scope rejection; BANKING77 transfer and candidate-recovery results remain weak. No distinct sampled-RL advantage or production readiness is claimed. See the canonical card for measured results, provenance, limitations and citation.
