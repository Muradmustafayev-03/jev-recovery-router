# jev-recovery-router: prospective research protocol

Status: **DRAFT / NOT SEALED / RESULTS PENDING**. This document specifies intended experiments. It is not a timestamped public preregistration. Mark a run preregistered only after the final configuration, split manifest, source revision and this document have been sealed before test evaluation. Record deviations separately; do not rewrite the original protocol after seeing results.

**[Hardware revision, before GPU training]** The available device is A100 80 GB, not H100. The current A100 preset changes only resource allowances and pilot length: up to three hours for the same two epochs, a maximum 50-update disposable smoke test, and $3.18/hour assumed billing. The original two-hour H100 references below are historical and are superseded by this paragraph for the A100 run. The nine-hour VM envelope, twelve-hour total ceiling and final two-hour evaluation reserve remain. See `A100_ESTIMATES.md` for sourced prices and explicitly unmeasured runtime assumptions.

Claim labels used here: **Protocol** = intended procedure; **Hypothesis** = testable prediction; **Assumption** = unmeasured modeling choice; **Derivation** = mathematical consequence; **Limitation** = what the experiment cannot establish. No trained-model performance result is reported here; the explicitly labeled retrieval diagnostic below documents a design revision.

**[Design revision / measured training-partition diagnostic]** Before decision-model or actor training, frozen MiniLM retrieval left only 9 of 2,429 policy-fitting queries recoverable by expanding 5 candidates to 20. Expanding 1 to 5 made 243 of those queries recoverable. The primary schedule is therefore 1-to-5; 5-to-20 is retained as a retrieval-only saturation diagnostic. These counts measure candidate availability, not trained routing accuracy, utility or an RL gain. Whole-corpus retrieval summaries had already been visible; this is an openly revised protocol, not an untouched public preregistration. See `DESIGN_DECISIONS.md` for provenance and limitations.

## 1. Questions and falsifiable claims

**[Hypothesis H1]** On the natural `test_base` split (120-intent catalog), the prespecified RL actor seed 42 improves mean utility over the declared validation-selected one-threshold confidence gate at the sealed primary cost vector. The primary endpoint uses the empirical test mixture and a paired 95% bootstrap interval. The gate commits above a tuned initial threshold and otherwise expands; its terminal commit/defer threshold is cost-derived. It is a restricted comparator, not the best possible threshold policy. Mixture-standardized results, the updated catalog, other costs, other seeds and intervention arms are secondary. The intervention arm alone cannot establish H1.

**[Hypothesis H2]** The same fitted policy transfers to 30 additional CLINC intents through catalog addition without updating model, calibration or actor parameters. Report performance on old and added intents separately; a pooled gain can conceal regressions on the original catalog.

**[Hypothesis H3]** Some recovery behavior transfers to BANKING77 with a new labeled catalog and no parameter update. This is a second-domain stress test, not proof of general industry usefulness.

**[Open question H4]** Does sampled-action RL improve on supervised action-value learning or exact expected-utility optimization when complete training feedback is available? No superiority is presumed. The exact comparator uses the same actor architecture; a tie or loss argues for the simpler training method.

**[Success criterion]** Report H1 as supported only if the predefined main comparison favors recovery with a paired interval excluding zero, without concealing higher error through lower coverage. Report effect size, interval, baseline and operating point together. For H2/H3 report both point estimates and uncertainty; do not claim generalization from a favorable subset. All outcomes, including failed hypotheses, belong in the release.

## 2. System under test

**[Protocol]** A decision model sees the utterance and the current candidate catalog and returns candidate probabilities plus a NONE event. A frozen feature representation is passed to an actor conditioned on a declared cost vector. At the first state the actor may COMMIT, EXPAND once, or DEFER. Expansion retrieves a larger candidate set and rescoring creates the second state. The final state permits COMMIT or DEFER. Never expose expanded-state probabilities, gold labels, intervention flags or recoverability labels in first-state features.

**[Definition]** NONE means that the correct supported intent is absent from the supplied candidate set, or that the utterance is out of scope. Those causes are distinguished in evaluation metadata, not magically known to the actor. NONE is a belief event; it is not an automatic successful customer resolution. A user-facing escalation is logged as DEFER. All output probabilities retain their event definition and catalog version.

**[Protocol]** Freeze the probability model and its calibration before fitting the actor. The actor's categorical probabilities are action-selection probabilities and must never be presented as confidence that an intent is correct. Cost conditioning is supplied to the actor, not used to rewrite factual probabilities.

## 3. Data partitions and leakage controls

**[Protocol]** Use the pinned original CLINC OOS+ source corresponding to the `clinc/clinc_oos` `plus` configuration. Persist the exact 120 fit-intent names and 30 held-out-intent names chosen deterministically using seed 42, plus the selected row IDs/hashes. Exclude all 30 held-out intents from the recovery system's belief fitting, actor fitting, calibration, hyperparameter selection and threshold selection. Do not label them OOS during fit and still call them unseen in this experiment. The explicitly labeled all-training-data classifier reference is an exception: it sees all 150 intents and must not be called a matched-data transfer baseline.

**[Protocol]** Keep query examples used for belief fitting separate from actor-fitting queries. Split each seen intent's 100 official training examples into 20 index exemplars, 60 belief-fit queries and 20 actor-fit queries. Split the 250 official OOS training examples into 200 belief-fit and 50 actor-fit queries. Index examples are never used as evaluation queries. Record exact realized counts and duplicate checks in the manifest. Any change must be sealed before training and charged to the time budget.

**[Protocol]** Split official validation rows deterministically into a calibration partition and a policy-selection partition. Calibration data fit temperatures or other declared calibrators; policy-selection data choose thresholds and checkpoints. Test rows are used once for the frozen comparison. OOS data follow the same disjoint roles. Cluster all views of an utterance under one group ID; exact/normalized duplicates cannot straddle fit, calibration, selection and test unnoticed.

**[Protocol]** On the primary task, the catalog contains only the 120 fit intents. For catalog-addition transfer, freeze everything and append train-only catalog exemplars for the 30 held-out intents, yielding 150 intents. Test old and added intent slices under the enlarged catalog. Recalibrating with added-intent labels is a separate adaptation experiment, not this transfer result.

**[Protocol]** For BANKING77, build the catalog from 20 deterministic official-training examples per class. No BANKING77 examples may update Qwen, MiniLM, the actor or calibration in the primary transfer arm. Labels and examples are available through the catalog; call this **labeled-catalog transfer without parameter fine-tuning**. An optional BANKING77-trained logistic-regression baseline must be labeled as a domain-trained reference with more adaptation. BANKING77 supplies no native OOS test class; do not invent an OOS score for it.

**[Limitation]** Pretraining contamination is unknown. Holding out labels during this project's fine-tuning does not establish that Qwen or MiniLM have never encountered their text, concepts or benchmark examples. The benchmark languages and domains are narrow.

## 4. Natural and intervention arms

**[Protocol]** Produce natural top-1 candidate lists with the frozen MiniLM retriever and sealed index; expansion uses the top 5. This arm is the main performance estimate. Record these values in the resolved run configuration and data manifest. Report natural recall@1 and recall@5 first. Keep the 5-to-20 retrieval saturation diagnostic separate. TF-IDF retrieval is an explicitly labeled diagnostic/fallback, not a silent replacement of the primary retriever.

**[Architecture limitation / protocol]** The model has capacity for 20 positional candidate columns, but training only on 1- and 5-candidate views does not train positions 6 through 20. Primary inference supports the recorded trained counts, 1 and 5; it must reject a manifest requesting untrained counts. The default 5-to-20 control therefore measures retrieval only, not learned-policy performance. A future full policy control requires its own matching training/calibration artifacts, or a separately declared mixed-count training experiment.

**[Protocol]** Separately construct paired interventions on eligible queries: deliberately omit the true intent from the initial candidate set while leaving the normal expanded catalog mechanism available. A second diagnostic may omit it from the complete catalog, making retrieval unable to repair the error. Every pair has the same original query and group ID. Never silently inject the gold intent into the expanded candidates of the natural arm. Do not expose intervention labels to the model.

**[Limitation]** Controlled removal tests response to a known failure mechanism. It is not an estimate of how often a production retriever fails or how large a real-world benefit will be. Its altered class mixture can also change calibration. Report natural and intervention arms in separate tables; no pooled headline number.

## 5. Costs and policy fitting

**[Assumption]** Wrong-routing, expansion and review costs are simulated utility units, not measured dollars or employee time. The primary configuration gives +1 for a correct automatic route, -5 for a wrong automatic route, -0.2 for deferral, and subtracts 0.05 when expanding. Train/evaluate the full cost grid: wrong penalty {2, 5, 10}, defer cost {0.1, 0.2, 0.4}, expansion cost {0.01, 0.05, 0.15}; correct-route reward stays +1. The secondary held-out interpolation combination is wrong=7, defer=0.3, expand=0.08. This is an unseen cost combination, not a new benchmark. Freeze the cost sampling distribution and full grid before test evaluation; never select a favorable headline triple afterward.

**[Definition]** Per-query utility is `I(correct automatic route) - wrong_cost * I(wrong automatic route) - defer_cost * I(defer) - expand_cost * I(expand)`. Higher is better; a corresponding loss is its negative. The +1 correct-route reward is essential and must not be dropped when deriving decision thresholds. A deferred query is unresolved by the model. Review utility is a simulated outside option, not a claim of perfect human accuracy. For a simulated end-to-end accuracy calculation, any assumed reviewer accuracy must be explicit and swept separately.

**[Protocol]** Fit a small actor with REINFORCE on sampled one- or two-decision episodes using frozen observations. Cache deterministic decision-model observations to reduce repeated expensive encoding, but disclose the counterfactual collection cost. Training sees only rewards for sampled actions in the RL update; the full-feedback baseline can use all training action outcomes. Use comparable feature access and compute limits. Full-feedback advantages must not be hidden.

**[Protocol]** The backbone adaptation schedule permits up to two passes through its training views or a 7,200-second training-loop limit, whichever stops it first. The revision was fixed before training the decision-model weights. This is a stopping rule, not a claim that setup, training and checkpointing will finish within exactly two hours. All of that time counts toward the overall 12-hour constraint.

**[Protocol]** Seed 42 is the main backbone/split run. Fit actor seeds 42, 43 and 44 with identical data and features. Report each seed and a summary; do not report only the best actor. Three actor seeds do not measure backbone-training uncertainty.

## 6. Required comparisons and ablations

| Comparison | Purpose and fairness constraint |
|---|---|
| Never expand, cost-implied confidence/defer gate | Tests whether retrieval action is needed. Same belief model and calibration; derive its threshold from the full utility, including +1 for correct routing. |
| Always expand to 5, then cost-implied confidence/defer gate | Essential baseline: tests whether the 1-candidate first pass is actually useful. Charge expansion on every query and measure the total latency. A cheap direct 5-candidate service may be preferable. |
| Confidence-triggered expansion and deferral | Tune initial expansion threshold on policy-selection data; the initial implementation uses the cost-implied terminal commitment threshold. A separately validation-tuned terminal threshold is a useful stronger sensitivity baseline, not an assumed completed comparison. |
| Full-feedback supervised action-value model | Regress realized current/expanded/deferred utilities from stage-0 observations plus cost, using a fixed stage-1 cost-threshold policy. Expanded targets include its contingent commit/defer decision after observing stage 1. No gold or future observation is available to the stage-0 predictor at test time. |
| Exact full-information expected-utility optimizer | Same actor architecture, data/features and cost grid as sampled RL; differentiate the exact return summed over the bounded action paths. It can learn both stages, so this comparison separates sampled feedback from policy expressiveness. This is not sampled-action RL. |
| Cost-conditioned RL actor | Sampled episodes, all actor seeds, same serving information. |
| Cost-independent actor or removal of cost input | Tests whether conditioning actually permits useful cost changes. |
| TF-IDF plus SGD logistic classifier on all cleaned original training queries | Explicit stronger-data classifier reference: 15,154 rows after overlap removal, including the 30 added intents. It cannot establish a matched-data transfer comparison. |
| Frozen MiniLM retrieval and logistic-regression reference | Tests whether Qwen is worth its training and inference cost. Label any supervised adaptation. |
| Unadapted Qwen with the same candidates and output definition | Separates backbone prior capability from decision-model adaptation. |
| Hindsight oracle | Evaluation-only upper bound over the same permitted action paths, same model predictions and costs, using held-out labels. Not deployable and not a trained baseline. |
| Optional Jev API | Only after explicit configured cost cap, available credentials and recorded served model version. Otherwise mark NOT RUN. Use matched query/candidate definitions; do not compare incompatible published numbers. |

**[Protocol]** If deadline pressure removes a comparison, mark it missing and narrow the paper claim. Do not disguise a synthetic or random-weight smoke test as a trained baseline.

**[Limitation]** More recoverable fitting examples make the recovery question estimable; they do not establish that manufacturing a smaller initial shortlist improves a real service. The 1-to-5 system must justify itself against always using 5, and its simulated expansion penalty must not substitute for measured latency. The main research question is cost-conditioned, no-refit behavior under catalog/domain transfer, not simply showing that a larger list contains more correct labels.

**[Comparator definition]** Do not replace contingent expansion targets with `max(E[commit utility after expansion | s0], defer utility) - expand_cost`: taking the maximum before observing the second state loses the value of changing the action after that observation. The implemented regression comparator uses realized outcomes of a frozen terminal policy; it does not train that policy on its own outcome targets. The exact optimizer learns both actor stages and provides the matched-capacity comparison. Record optimizer and critic-loss differences, and use matching entropy objectives when comparing sampled and exact updates.

## 7. Metrics, intervals and calibration

**[Protocol]** Primary metrics are operational utility and its paired difference from the selected simple gate on natural candidates. Also report automatic coverage, conditional error among automatic routes, wrong routes per all arrivals, defer rate, expansion rate, correct resolutions per all arrivals, p50/p95 latency and cost per query. An always-defer system must show zero automatic coverage, not 100% model accuracy. A zero denominator produces an undefined metric, not zero error.

**[Protocol]** Diagnose: (a) catalog-membership/OOS status; (b) natural candidate recall; (c) selector accuracy conditional on true candidate presence; (d) expansion recovery and harm; (e) unnecessary expansion. A stronger score after expansion can reflect added information and still be miscalibrated. Break out original intents, added intents, OOS and second-dataset results.

**[Protocol]** Use paired bootstrap intervals over original utterance group IDs, preserving every method prediction and every intervention view in its group. The small-budget configuration uses 1,000 resamples and bootstrap seed 2026; report this finite Monte Carlo resolution and sample size. Where feasible, add an intent-cluster sensitivity interval, since examples share intent vocabulary. Neither interval captures unknown pretraining contamination or all training variance. Identify one primary comparison; label cost-grid and subgroup analyses exploratory and report all of them.

**[Protocol]** Report Brier score and negative log-likelihood with the outcome space defined at each stage; ECE is a secondary descriptive statistic with binning recorded. Fit calibration on the calibration partition only. Report raw and calibrated models, plus stage- and membership-specific results. Test-specific fitted temperature is leakage.

**[Assumption / protocol]** The primary target OOS arrival prevalence is 2/11 (approximately 18.18%), matching the original 150-intent test mixture, not an observed customer prior. The 120-intent test subset has a different raw mixture. If its sample OOS rate is `q` and target is `p`, use weights `p/q` for OOS and `(1-p)/(1-q)` for in-scope examples when computing target-mixture averages. Do not pool candidate-removal interventions into those weights or apply OOS weights to BANKING77's in-scope-only test. This standardization assumes within-group distributions are representative; changing mixture weights alone does not prove transport calibration. Any prior-probability correction applied to outputs is a separate fitted/assumed procedure and must be specified. Retain unweighted benchmark metrics too. Additional prevalence sensitivity settings must be declared before test evaluation.

## 8. Seal, resources and stopping rules

**[Protocol]** Before opening decision-model/actor test metrics, record a SHA-256 seal covering resolved configuration, this protocol and design history, split/group manifests, data and model revisions, catalog IDs, calibration parameters, selected thresholds/checkpoints, source commit or source-tree manifest, and environment versions. Store the UTC sealing timestamp and subsequent evaluation timestamp. Earlier retrieval diagnostics remain disclosed. A hash is an audit record, not independent proof of non-access; public preregistration needs an externally timestamped artifact. File output paths are implementation details; the run manifest must link every actual artifact.

**[Constraint]** The experiment has a total $100 and 12-hour ceiling, including preparation, coding, debugging, data processing, downloads, evaluation and documentation. Keep a wall-clock ledger from the beginning of work, and a separate billed-resource ledger. Concurrent tasks consume wall time once; their machine charges all count. Earlier preparation is recorded as elapsed preparation, not reset to zero when the GPU starts. Unknown historical durations stay unknown until recovered; do not fabricate them.

**[Arithmetic from supplied prices]** One H100 costs $5.95/hour; 12 billed hours are $71.40 before extras. Eight H100s cost $47.60/hour; two hours cost $95.20 before extras. Use one H100 and reserve funds for storage/taxes/other actual charges. These are supplied rates, not a verified invoice. Run a brief throughput pilot and adapt training volume only under the predeclared deadline rule.

**[Protocol]** Preserve at least the final two hours for frozen evaluation, artifact validation and reporting. At the budget/deadline limit stop training and publish the achieved state, including unrun experiments. Never silently spend the reserve to manufacture a favorable result. If the primary experiment cannot finish, release a reproducible protocol/code artifact with pending metrics, not a trained-model success claim.
