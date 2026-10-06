# Data and retrieval for jev-recovery-router

This is public benchmark data, not customer deployment evidence. Gold labels and
diagnostic groups live in dataset records for training/evaluation. They must never
be included in model inputs. Candidate descriptions are explicitly exposed catalog
metadata. Input rendering belongs to the model module, not this data loader.

## Sources and redistribution

The downloader uses immutable Git revisions. `data/downloads/manifest.json` records
the exact source URLs, SHA-256 digests and byte counts; original licenses are saved
alongside the source data. Existing cached files are checked against their saved
hashes before reuse.

* CLINC150 OOS+: Larson et al., EMNLP-IJCNLP 2019,
  [paper](https://aclanthology.org/D19-1131/),
  [original repository](https://github.com/clinc/oos-eval).
  Revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`, CC BY 3.0.
  In-scope train/validation/test: 15,000/3,000/4,500. OOS: 250/100/1,000.
* BANKING77: Casanueva et al., ACL NLP for Conversational AI workshop 2020,
  [paper](https://aclanthology.org/2020.nlp4convai-1.5/),
  [original repository](https://github.com/PolyAI-LDN/task-specific-datasets).
  Revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, CC BY 4.0.
  Original train/test: 10,003/3,080; 77 intents.

Derived JSONL files change splits, add candidate lists and diagnostic fields. Cite
the original authors, preserve original licenses, and state these modifications
when redistributing the data. Public benchmark pretraining contamination is unknown.

## Deterministic roles

The counts below are **nominal allocations before duplicate removal**. The manifest
contains the actual counts and every removed record ID and reason is in the audit.
Seed 42 selects 120 existing CLINC intents; 30 novel labels are withheld from
supervised fitting, policy fitting, calibration and threshold selection. Per-class
training rows are shuffled with stable SHA-256-derived seeds, then allocated:

| Role | Contents | Catalog |
|---|---|---|
| `index.jsonl` | 20 examples per intent, all 150 intents: 3,000 rows | 150 |
| `train.jsonl` | 60 examples × 120 existing intents + 200 OOS: 7,400 | 120 |
| `policy_train.jsonl` | 20 × 120 + 50 OOS: 2,450 | 120 |
| `calibration.jsonl` | Half existing-intent validation + half OOS: 1,250 | 120 |
| `tuning.jsonl` | Other half: 1,250 | 120 |
| `test_base.jsonl` | Official existing-intent and OOS test: 4,600 | 120 |
| `test.jsonl` | Entire official test: 5,500 | Updated 150 |
| `novel_validation.jsonl` | 600 novel-intent validation rows, audit only | 150 |
| `banking_index.jsonl` | 20 original train examples × 77: 1,540 | BANKING77 |
| `transfer.jsonl` | Entire untouched BANKING77 test: 3,080 | BANKING77 |
| `direct_classifier_train.jsonl` | All original CLINC training: 15,250 | 150 |

`test_base` measures the original task; `test` tests catalog addition. Novel intent
descriptions and 20 exemplars per added intent are available during the catalog
update. This is **label/exemplar-exposed catalog transfer**, not zero-shot discovery.
`novel_validation` is never for selection. BANKING77 receives no adaptation and has
no OOS examples; do not claim OOS performance or apply artificial OOS weighting to it.

The direct classifier using original training rows after deduplication is a **stronger-data practical
comparator**, including novel training labels. Also use the 7,400-row supervised
subset when comparing methods under equal labeled training data.

## Retrieval and interventions

```
python -m recovery_router.data prepare --data-dir data --seed 42 --backend minilm --first-k 1 --expanded-k 5
python -m recovery_router.data prepare --data-dir data --seed 42 --backend minilm --first-k 5 --expanded-k 20 --processed-subdir control-5to20
python -m recovery_router.data prepare --data-dir data --seed 42 --backend tfidf --processed-subdir diagnostic-tfidf-top1
```

The script entry point is `python scripts/prepare_data.py --backend minilm`.
The default profile is top-one to top-five. `--processed-subdir` separates
candidate profiles without changing original query IDs. Prediction artifacts must
match the entire manifest hash and candidate positions; matching IDs alone does
not make a prediction reusable across profiles. TF-IDF is the inexpensive CPU
diagnostic, not the intended main semantic-retrieval result.

MiniLM uses frozen `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Only original index examples are
searchable. Score each intent using the maximum similarity of its examples,
then return one unique intent and five unique intents in the primary profile.
The saturation control uses five and twenty. The latter state is a real
superset of the former; no gold injection occurs. The MiniLM cache is keyed by
model revision and exact input texts. Both backends break ties deterministically.

Candidate positions are shuffled deterministically and independently in each
state. The explicit arrays specify the output-position mapping. Candidate labels
are short descriptions obtained from original intent names; no paid teacher or
test-derived exemplars are used.

`*_drop_gold.jsonl` and `*_shuffle.jsonl` are separate controlled stress artifacts
(at most 1,000 rows each). Drop-gold applies only where the real shortlist contains
gold; it substitutes a real expanded-list distractor without changing expanded
membership. It therefore evaluates deliberately manufactured recoverability, not
natural prevalence or production accuracy. Order shuffling preserves all candidates.
`parent_id` links a variant to its original; use it for paired confidence intervals.

## Leakage and calibration cautions

The preparation audit checks normalized-text collisions across disjoint fitting,
index and evaluation roles, including CLINC-to-BANKING77 collisions. It excludes
intentional alternate evaluation views and the explicitly overlapping stronger
direct-classifier training set. All official test rows are preserved. Fitting and
index rows matching evaluation text are excluded; other fitting-role collisions
are resolved deterministically, prioritizing calibration, tuning, index, then
adaptation and policy rows. Removed rows are not replaced, so individual index
classes may have fewer than 20 exemplars. Preparation asserts every catalog intent
still has index examples and that no unresolved fitting collision remains.
The direct classifier is also deduplicated against evaluation/selection text.
Any evaluation-to-evaluation overlap is retained and reported: these datasets
cannot then be treated as completely independent confirmations. Every removal is
recorded with ID, hash, reason and matching record. This catches identical text,
not paraphrase leakage or pretraining contamination.

The nominal fitting OOS priors are not the official test prior: training 200/7,400,
policy 50/2,450, calibration 50/1,250, base test 1,000/4,600, updated test
1,000/5,500. Report subgroup metrics and explicit prior assumptions. A tiny OOS
calibration subset does not establish a tiny false-accept rate. Freeze all weighting,
calibration and policy-selection choices before using either test set.

`manifest.json` records role counts, stable/novel labels, natural recovery counts,
catalog sizes, source checksums and processed-file hashes. Natural retrieval situations
are `present_initially`, `recoverable_after_expansion`, `missing_after_expansion`,
and `true_oos`. Older archived manifests retain the original numeric case names.

## Primary profile choice from training data only

Before training the decision backbone, frozen MiniLM retrieval diagnostics on
the 2,429 **policy-training** queries showed:

| Initial → expanded unique intents | Naturally recoverable training queries |
|---|---:|
| 1 → 5 | 243 |
| 3 → 10 | 36 |
| 5 → 20 | 9 |

The primary experiment therefore uses top-one to top-five. The first step is a
cheap accept/retrieve/handoff decision about one proposed intent; the wider step
can distinguish five candidates. The initial model still returns a probability
for the candidate and for NONE, so this is not a forced one-option classification.
This choice used policy-training retrieval labels, not novel/test model performance.
It does not establish that RL will improve outcomes. The historical preparation
workspace retained the original frozen MiniLM profile as a saturation control in
`data/control-5to20` and the original TF-IDF diagnostic in `data/diagnostic-tfidf`.
These generated directories are excluded from this code-only repository.
No test outcome was used to select this
profile, and no trained-model evaluation is implied by these retrieval counts.

## Observed preparation result (29 September 2026, seed 42, TF-IDF)

These are data-preparation measurements, **not trained-model performance**. The
source downloads contained 73 normalized-text collision groups across proposed
roles. Cleaning removed 18 CLINC index rows, 5 BANKING77 index rows, 38 adaptation
rows, 21 policy rows, 2 calibration rows and 1 tuning row. The stronger direct
classifier's independently cleaned training set lost 96 rows. No fitting collision
remained. Three evaluation-only collision groups remained explicitly flagged.

Actual clean sizes: 2,982 CLINC index; 7,362 adaptation; 2,429 policy;
1,248 calibration; 1,249 tuning; 15,154 direct-classifier training. Every catalog
intent still has exemplars: minimum 16 per CLINC intent and 18 per BANKING77 intent.
Original tests remain 4,600 base-catalog, 5,500 updated-catalog and 3,080 BANKING77.

| Natural TF-IDF retrieval | Gold in 5 | Missing in 5, present in 20 | Missing in 20 | OOS |
|---|---:|---:|---:|---:|
| Policy fitting | 2,285 | 70 | 24 | 50 |
| CLINC base-catalog test | 3,416 | 128 | 56 | 1,000 |
| CLINC updated-catalog test | 4,234 | 184 | 82 | 1,000 |
| BANKING77 transfer | 2,856 | 186 | 38 | 0 |

Only 70 policy-fitting examples are naturally recoverable under this lexical
retriever. This limits the evidence available to the RL policy; controlled omissions
must remain separately labeled if used for additional training or stress tests.
Regenerating with MiniLM will change retrieval counts, but not the data-role split.

## Final primary data provenance (MiniLM, top 1 → top 5)

The primary prepared dataset uses seed 42 and frozen MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Preparation completed offline from the
cached model and embeddings. Manifest SHA-256:
`f590fe035c785d7fd388b67f6a225b345cd0f4573c4566807d2b30265fe542ec`.
The manifest's profile is `minilm-top1-to5`. All 25,468 natural rows across fitting,
calibration/tuning and the three main tests passed explicit 1/5 candidate-count
and nested-membership checks. Dataset roles and duplicate exclusions are unchanged.

| Primary MiniLM retrieval | Gold initially | Recoverable after expansion | Missing after expansion | OOS |
|---|---:|---:|---:|---:|
| Supervised fitting | 6,366 | 732 | 64 | 200 |
| Policy fitting | 2,122 | 243 | 14 | 50 |
| Calibration | 1,055 | 128 | 15 | 50 |
| Threshold tuning | 1,043 | 137 | 19 | 50 |
| CLINC base-catalog test | 3,101 | 444 | 55 | 1,000 |
| CLINC updated-catalog test | 3,827 | 598 | 75 | 1,000 |
| BANKING77 transfer | 2,676 | 339 | 65 | 0 |

These are **retrieval coverage counts**, not decision-model accuracy. The primary
profile was selected from policy-training diagnostics before this test-profile
summary was produced. No test-model outcome was used for that selection.

The classical benchmark command is:

```
python scripts/run_classical.py --data-dir data/processed --output runs/classical-minilm --backend minilm
```

This fits a fixed logistic-regression classifier on frozen MiniLM vectors, using
15,154 clean CLINC training labels across all 150 intents plus OOS. A separate
BANKING77 classifier fits the 1,535 catalog exemplars. It does **not** adapt the
encoder or train the proposed decision backbone or RL policy. Stable-class CLINC
calibration/tuning labels choose temperature and handoff thresholds under the
predeclared OOS-prior assumption 2/11. BANKING77 test labels never select a gate:
the report uses always-route and fixed raw-probability 0.8 variants, plus the
unchanged CLINC-calibrated nearest-neighbor gate. Reports explicitly mark these
calibration-transfer limitations and the direct classifier's larger label budget.

The primary CPU baseline run completed on 29 September 2026. Fixed classifier
fitting took 38.0 seconds and the complete run took 46.8 seconds on this local
machine; these are not H100 timing estimates. A final integration rerun reused the
embedding cache and took 3.7 seconds for fitting and 11.5 seconds overall, with
the same reported routing rates. The historical generated report records that rerun,
including explicit test-data fingerprints. No optimization-convergence warning
was emitted. Optional BLAS thread limiting failed because the Windows library
enumeration did not preserve the non-ASCII path, so computation used default
thread pools; that warning is retained in the results.

| Measured primary classical baseline | Coverage | Error among accepted | Correct-route rate |
|---|---:|---:|---:|
| CLINC base, calibrated full-label direct classifier | 69.98% | 3.14% | 67.78% |
| CLINC updated, calibrated full-label direct classifier | 72.98% | 2.72% | 71.00% |
| BANKING77 catalog-example classifier, raw fixed 0.8 gate | 13.90% | 0.93% | 13.77% |

Without abstention, the BANKING77 catalog-example classifier achieved 88.28%
accuracy and its nearest-exemplar comparator achieved 86.88%. The fixed 0.8 gate
had low coverage and negative assumed utility; it was not retuned after observing
that test outcome. The CLINC-calibrated nearest-exemplar gate transferred with
96.23% coverage and 11.98% accepted error on BANKING77, illustrating that the
calibration's behavior does not transfer uniformly across datasets.

The direct classifier's temperature was 0.6272 and its handoff threshold 0.84,
both selected using only the allocated stable-class calibration/tuning data.
The nearest-exemplar gate threshold was 0.78. In this top-one profile its
runner-up-gap input is identically zero: it does not access expanded candidates
before expansion. Historical output paths were `reports/classical-minilm/RESULTS.md`,
`results.json`, and `fitting.json`, with model files and per-query records under
`runs/classical-minilm-final`. These generated outputs are excluded from this
code-only repository. The historical report carries the same primary manifest
hash above; the source scripts can generate new local baseline outputs.
