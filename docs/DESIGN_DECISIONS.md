# Design decisions before decision-model training

Status: dated design history, 2026-09-29. This is not an independently timestamped public preregistration. It preserves the rationale for changing the experiment before training the Qwen decision head or recovery actor. No learned-model performance result is claimed.

## Candidate budget: 5-to-20 becomes a diagnostic; 1-to-5 becomes primary

**[Initial plan]** Start with five naturally retrieved candidates and expand to twenty. The model architecture has a maximum of twenty candidate positions plus NONE.

**[Measured diagnostic]** In the pinned frozen-MiniLM retrieval preparation, the deduplicated policy-fitting partition contained 2,429 queries. Nine were in scope with the correct label absent from the top five but present in the top twenty. Re-ranking the same fitting queries by the recorded retrieval score and label tie-break showed 243 with the correct label absent from the top one but present in the top five. These are approximately 0.37% and 10.00% of all policy-fitting arrivals, respectively. OOS rows remain in the denominator but cannot be recoverable in-catalog cases.

The source was the original 5-to-20 `data/processed/policy_train.jsonl`, SHA-256 `2b7bac71c9834a9cb3edd6e87d4356c120b19b9fa8a73b347d78ad7fd1596a7d`, with preparation manifest SHA-256 `72182d8fd49a11afc1ca6f41a4acdd23314cb5980f280f06a8813ea9a7323890`. Retrieval used `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The regenerated primary artifacts have different hashes and should not be presented as this original snapshot.

**[Decision]** Use 1-to-5 as the primary cheap-first-check/recovery schedule. Retain 5-to-20 as a retrieval-only saturation diagnostic. Natural candidates remain label-independent; no gold label is inserted to create recoverability. This provides enough naturally recoverable fitting cases to study the policy, without using deliberate candidate-removal interventions as the headline performance distribution.

**[Disclosure]** Whole-corpus retrieval summaries had already been produced and visible when this decision was made. The explicit justification uses the policy-fitting partition, and no trained decision-model or actor test metrics existed. Nevertheless, do not claim that all test information was untouched or call the revision a blind public preregistration. Seal the final configuration before evaluating the trained system and retain this history.

## What the change does not establish

**[Limitation]** The 243 cases are possible candidate-recall repairs, not 243 correct predictions after expansion. The selector can still be wrong, expansion can harm an initially correct decision, and an actor can choose poorly. The count does not establish calibration, improved utility or an RL advantage.

**[Required comparison]** Always retrieve/score five candidates and apply a simple cost-aware confidence gate. If that baseline is as cheap and effective as the learned two-stage system, there is no practical case for the learned recovery policy at that operating point. Simulated expansion penalties do not establish dollar savings; measure actual latency and report every declared cost scenario.

**[Research scope]** The strongest question is cost-conditioned behavior without refitting when costs and the catalog change, followed by transfer to BANKING77. Adaptive retrieval and RL routing already have prior art. The schedule change is not a new algorithm or proof of global novelty.

## Positional-head support and the saturation control

**[Architecture fact]** A positional head trained on candidate counts 1 and 5 receives no valid-class training for positions 6 through 20. Merely allocating twenty columns does not train them. Therefore a 1/5 checkpoint must not be used to claim a fair learned-policy result on a 5/20 schedule.

**[Decision]** Record `trained_candidate_counts` in each model bundle. The online CLI checks that the manifest's two candidate counts are supported. The default 5/20 control is retrieval-only. A future full learned-policy control requires a separately trained and calibrated 5/20 model, or an explicitly declared mixed-count training design evaluated as a different experiment.

## Training effort

**[User hardware correction, before any GPU training]** The available device is A100 80 GB. `configs/a100.json` keeps the same model, data, epochs, batches and evaluation, updates the published hourly rate to $3.18 and permits a three-hour main-training ceiling inside the unchanged nine-hour VM cap. It uses up to 50 disposable smoke updates to estimate throughput. The following two-hour rule remains the historical H100 configuration. See `A100_ESTIMATES.md`; no GPU timing or outcome informed this revision.

**[Plan, not measured duration]** Adapt the decision model for up to two passes or a 7,200-second training-loop limit. This was chosen before decision-model weight training. Setup, downloads, preprocessing, checkpoint writes, actor fitting and evaluation all still count toward the total twelve hours and $100. Time-cap stopping is an admissible outcome and must be reported; it is not permission to claim two completed passes.
