# Attribution and artifact licenses

The project name expresses inspiration. This project is not affiliated with or endorsed by TypeSafe AI. Jev is an external optional comparator; no Jev weights are redistributed.

Original project code is licensed under Apache License 2.0, in `LICENSE`. That license does not replace third-party model or dataset licenses.

- CLINC150 / out-of-scope benchmark: Stefan Larson and collaborators, *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019. [Paper](https://aclanthology.org/D19-1131/), [original repository](https://github.com/clinc/oos-eval). Source license: CC BY 3.0. This project deterministically splits, decontaminates and augments records with retrieved candidate slates. Original texts are retained; generated metadata are clearly distinguished.
- BANKING77: Iñigo Casanueva and collaborators, *Efficient Intent Detection with Dual Sentence Encoders*, 2020. [Paper](https://aclanthology.org/2020.nlp4convai-1.5/), [original repository](https://github.com/PolyAI-LDN/task-specific-datasets). Source license: CC BY 4.0. This project selects training exemplars, filters duplicate overlaps and constructs a retrieval index.
- Qwen3-1.7B and Qwen3-0.6B: Qwen team. [Main model card](https://huggingface.co/Qwen/Qwen3-1.7B). Pinned source revision appears in the experiment configuration. Apache-2.0 model license; retain the source model's attribution when publishing a derived adapter/model.
- all-MiniLM-L6-v2: sentence-transformers contributors. [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Apache-2.0 model license. The encoder is frozen; its weights are downloaded from the pinned upstream revision.

Dataset source URLs and revisions are pinned in `src/recovery_router/data.py`. Running preparation writes exact file hashes to `data/provenance/manifest.json`; generated data are not included in this source repository. Original license texts are retained in `data/licenses/`. Do not publish private customer data in place of these public examples without a separate data-handling review.
