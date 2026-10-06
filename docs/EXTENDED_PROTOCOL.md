# Extended validation-selected training protocol

Status: follow-up experiment specified 2026-09-29 after viewing v1 test results, at the user's request for longer, better-instrumented training. This is exploratory; no fresh confirmatory test set is claimed. Original v1 results and frozen sources remain unchanged.

## Selection fixed before new training

Start from the same immutable Qwen3-1.7B revision and the same LoRA/head architecture. Keep all original query partitions, retrieval outputs and candidate meanings unchanged. Train for up to 12 epochs, with a 3,600-second training-loop ceiling and the original absolute cloud deadline. Peak learning rate is 1e-4, 5% linear warmup, cosine decay to 10% of peak, AdamW weight decay .01 and gradient clipping 1.0. Candidate order and corresponding target positions are regenerated deterministically each epoch.

Use the original tuning partition (1,249 queries) for checkpoint selection, as allowed in the original protocol. The separate calibration partition remains reserved for temperature fitting. Select the trained checkpoint minimizing the mean of first-view and expanded-view raw FP32 NLL, with each view weighted to the declared OOS prior 2/11. No temperature is fitted for checkpoint selection. Evaluate every half epoch; require an improvement greater than 1e-4 to replace an earlier best checkpoint. Step zero is diagnostic only and cannot be released as trained. Log unweighted and weighted validation metrics as well as example-weighted training loss. Validation inputs retain fixed candidate positions and never receive optimizer updates.

Reuse of tuning for checkpoint selection and routing-threshold selection is disclosed; this may overfit that validation partition. The test partitions are excluded from selection. Do not alter training choices in response to follow-up test outcomes. BANKING77 and the 600 novel-intent validation rows remain excluded from fitting and checkpoint selection.

## Comparisons and interpretation

Read-only scoring of v1 saved checkpoints diagnoses its learning curve. The original final 922-step v1 model remains the recorded baseline, regardless of those diagnostics. The follow-up changes duration, learning-rate schedule, candidate-order augmentation and checkpoint selection together; any improvement cannot be attributed to additional epochs alone. Report the entire curve and selected epoch, including the possibility that an early checkpoint wins.

After selection, freeze the selected belief model, fit calibration and policies with unchanged fitting partitions and hyperparameters, then run the same natural and stress evaluations. Reuse completed unadapted-Qwen, classical and Jev comparator artifacts only when their dataset hashes and utility costs match. Disclose that v1 supplied the benchmark evidence that motivated this follow-up. A validation-selected model improving on already-inspected tests is exploratory evidence, not an untouched confirmation.

## Constraints

All source/configuration/validation artifacts are fingerprinted. Original cloud start is 06:18:12 UTC; the hard cloud deadline remains 15:18:12 UTC. Preserve at least two hours for evaluation at the outer deadline, and bound this follow-up to two hours independently. Rate assumption remains $3.18/GPU-hour; no new Jev API calls are required for comparing the follow-up. All cost assumptions and actual elapsed time must be reported separately from invoices.
