"""Build PNG/SVG figures from recorded artifacts, without training or test tuning.

Requires matplotlib only for rendering. Missing results remain visibly pending;
no fabricated observations, imputed losses, or aggregate leaderboard are drawn.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPLETE = "measured_offline_evaluation_complete"
COLORS = {"v1": "#177E89", "v2": "#7555A3", "qwen": "#596579", "jev": "#CC7430",
          "v1_gate": "#619AA1", "v1_never": "#97B8BC", "v2_gate": "#9880B8", "v2_never": "#BEAFD0",
          "v2_exact": "#634F87", "v2_full_feedback": "#AD8EA7", "v2_fixed_cost": "#8D6396",
          "direct": "#3775A9", "nearest": "#789ABB"}
METHOD_COLOR_SUFFIX = {"rl_seed42": "", "exact_seed42": "_exact", "full_feedback": "_full_feedback",
                       "fixed_cost_seed42": "_fixed_cost", "confidence_gate": "_gate", "never_expand": "_never"}
DATASETS = [("test_base", "CLINC · 120 intents"), ("test", "CLINC · 150 intents"),
            ("transfer", "BANKING77 · catalog transfer")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self):
        self.sources = {}
        self.notes = []

    def read(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        raw = path.read_bytes()
        try:
            label = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            label = str(path.resolve())
        self.sources[label] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return json.loads(raw)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def checked_nll(row, field, weighted_stage_field):
    value = row.get(field)
    if not finite(value) or value < 0:
        raise ValueError("Curve has missing, negative or non-finite NLL")
    reconstructed = .5 * sum(row["stages"][stage][weighted_stage_field] for stage in ("first", "expanded"))
    if not math.isclose(value, reconstructed, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Recorded selection NLL disagrees with its two stage means")
    return float(value)


def learning_data(inputs, baseline, followup, baseline_config):
    v1 = inputs.read(followup / "v1_validation_curve.json")
    summary = inputs.read(followup / "main/training_summary.json")
    live = None if summary else inputs.read(followup / "main/validation_curve.json")
    result = {"v1": [], "v2": [], "training_epochs": [], "v2_status": "pending",
              "validation_sha256": None, "target_oos_prior": 2 / 11}
    if v1:
        if v1.get("status") != "complete":
            inputs.notes.append("V1 checkpoint curve is incomplete; available diagnostic points only.")
        config = inputs.read(baseline_config)
        metadata = inputs.read(baseline / "main/model/metadata.json")
        training = inputs.read(baseline / "main/model/training_report.json")
        if not config or not metadata or not training or digest(baseline_config) != metadata.get("training_config_sha256"):
            raise ValueError("Cannot verify v1 configuration for optimizer-step to epoch conversion")
        epochs = config["training"]["epochs"]
        steps_per_epoch = training["planned_optimizer_steps"] / epochs
        if steps_per_epoch <= 0 or int(steps_per_epoch) != steps_per_epoch:
            raise ValueError("V1 planned steps do not define complete equal-size epochs")
        result["v1_epoch_conversion"] = {"steps_per_epoch": steps_per_epoch,
                                          "config_sha256": digest(baseline_config)}
        result["validation_sha256"] = v1["validation_sha256"]
        if not math.isclose(v1["target_oos_prior"], 2 / 11, rel_tol=0, abs_tol=1e-12):
            raise ValueError("V1 curve does not use the declared OOS prior 2/11")
        for row in v1["checkpoints"]:
            result["v1"].append({"epoch": row["step"] / steps_per_epoch, "step": row["step"],
                                 "nll": checked_nll(row, "selector_weighted_nll", "weighted_nll"),
                                 "final_original_baseline": row["is_final_baseline"]})
        result["v1_best_diagnostic_step"] = v1.get("best_checkpoint_diagnostic_only", {}).get("step")
    rows = summary.get("validation_curve", []) if summary else (live or [])
    if rows:
        if summary:
            validation_sha = summary["validation_data_sha256"]
            result["v2_status"] = "completed_training_validation_selected"
            result["selected_step"] = summary["selected_step"]
            result["training_stop_reason"] = summary["stop_reason"]
            result["training_epochs"] = summary["epochs"]
            result["v2_last_epoch_fraction"] = rows[-1]["epoch_fraction"]
        else:
            ledger = inputs.read(followup / "run_ledger.json")
            if not ledger:
                raise ValueError("Live v2 curve requires its run ledger to verify validation data identity")
            validation_sha = ledger["signature"]["data"]["tuning.jsonl"]
            result["v2_status"] = "in_progress_snapshot_no_final_selection"
        if result["validation_sha256"] is not None and validation_sha != result["validation_sha256"]:
            raise ValueError("V1 and v2 validation populations differ; refusing an overlaid curve")
        result["validation_sha256"] = validation_sha
        for row in rows:
            if not math.isclose(row["target_oos_prior"], 2 / 11, rel_tol=0, abs_tol=1e-12):
                raise ValueError("V2 curve does not use the common declared OOS prior 2/11")
            result["v2"].append({"epoch": row["epoch_fraction"], "step": row["step"],
                                 "nll": checked_nll(row, "selection_mean_nll", "oos_standardized_nll"),
                                 "diagnostic_only": row["step"] == 0})
        if summary:
            selected = [r for r in result["v2"] if r["step"] == result["selected_step"]]
            if (len(selected) != 1 or selected[0]["step"] <= 0
                    or selected[0]["nll"] != summary["selected_validation_mean_nll"]):
                raise ValueError("Selected v2 checkpoint is not a positive-step member of the recorded curve")
    return result


def utility_data(inputs, baseline, followup):
    providers = {"v1": inputs.read(baseline / "main/results/results.json"),
                 "v2": inputs.read(followup / "main/results/results.json"),
                 "qwen": inputs.read(baseline / "qwen_base/results/results.json"),
                 "jev": inputs.read(baseline / "jev/results/results.json")}
    classical = inputs.read(baseline / "classical/results.json")
    if not providers["v1"] or providers["v1"].get("status") != COMPLETE:
        raise ValueError("A completed baseline main evaluation is required for utility plots")
    # A completed test file must describe the same selected model as the curve.
    # Incomplete/missing follow-up evaluations remain pending, never zero-valued.
    if providers["v2"] and providers["v2"].get("status") == COMPLETE:
        summary = inputs.read(followup / "main/training_summary.json")
        if not summary:
            raise ValueError("Completed v2 results require training_summary.json selection provenance")
        step = summary.get("selected_step")
        fingerprint = summary.get("selected_model_fingerprint")
        if (not isinstance(step, int) or isinstance(step, bool) or step <= 0
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)):
            raise ValueError("Completed v2 results require a positive selected step and model fingerprint")
        if providers["v2"].get("provider_fingerprint") != fingerprint:
            raise ValueError("Completed v2 results provider_fingerprint differs from the selected model")
    result = []
    for dataset, title in DATASETS:
        reference = providers["v1"]["datasets"][dataset]["scenarios"]["primary"]
        expected_hash = providers["v1"]["test_data_hashes"][dataset]
        rows = []
        for key, method, label in [
            ("v1", "rl_seed42", "V1 adapted · RL seed 42"),
            ("v1", "confidence_gate", "V1 confidence gate"),
            ("v1", "never_expand", "V1 never expand"),
            ("v2", "rl_seed42", "V2 adapted · RL seed 42†"),
            ("v2", "exact_seed42", "V2 exact optimizer · seed 42†"),
            ("v2", "full_feedback", "V2 full-feedback†"),
            ("v2", "fixed_cost_seed42", "V2 fixed-cost actor · seed 42†"),
            ("v2", "confidence_gate", "V2 confidence gate†"),
            ("v2", "never_expand", "V2 never expand†"),
            ("qwen", "rl_seed42", "Unadapted Qwen · RL"),
            ("jev", "rl_seed42", "Jev · RL"),
        ]:
            document = providers[key]
            value = None
            if document and document.get("status") == COMPLETE:
                if document["test_data_hashes"].get(dataset) != expected_hash:
                    raise ValueError(f"{key}/{dataset}: mismatched benchmark hash")
                scenario = document["datasets"][dataset]["scenarios"]["primary"]
                if scenario["costs"] != reference["costs"]:
                    raise ValueError(f"{key}/{dataset}: mismatched primary utility costs")
                value = scenario["methods"][method]["overall"]["mean_utility"]
            color_key = key + METHOD_COLOR_SUFFIX[method]
            rows.append({"key": color_key, "group": key if key in {"v1", "v2"} else "comparators",
                         "provider": key, "method": method, "label": label, "utility": value})
        for kind, label, suffix in [
            ("direct", "MiniLM direct classifier*", "direct_always_route" if dataset == "transfer" else "direct_calibrated_gate"),
            ("nearest", "MiniLM nearest + gate", "nearest_frozen_gate"),
        ]:
            value = None
            if classical and f"{dataset}_{suffix}" in classical.get("evaluations", {}):
                if classical["test_data_hashes"].get(dataset) != expected_hash:
                    raise ValueError("Classical comparison uses a different test file")
                entry = classical["evaluations"][f"{dataset}_{suffix}"]
                classic_costs = classical["fitting"]["utility"]
                if (classic_costs != {"correct": reference["costs"]["correct"],
                                      "wrong": -reference["costs"]["wrong"],
                                      "handoff": -reference["costs"]["defer"]}
                        or entry["overall"]["expand_rate"] != 0):
                    raise ValueError("Classical comparison uses different primary utility costs")
                value = entry["overall"]["mean_utility"]
            rows.append({"key": kind, "group": "comparators", "label": label, "utility": value})
        if any(r["utility"] is not None and not finite(r["utility"]) for r in rows):
            raise ValueError("Utility contains non-finite values")
        result.append({"dataset": dataset, "title": title, "test_sha256": expected_hash,
                       "costs": reference["costs"], "rows": rows})
    return result


def save_figure(fig, output, stem, plt):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_learning(data, output, plt):
    if not data["v1"] and not data["v2"]:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1))
    left, right = axes
    for key, label in (("v1", "V1: retrospectively scored"), ("v2", "V2: fresh training")):
        rows = [r for r in data[key] if r["step"] > 0]
        if rows:
            left.plot([r["epoch"] for r in rows], [r["nll"] for r in rows], color=COLORS[key],
                      marker="o", markersize=4, linewidth=1.8, label=label)
    initial = [r for r in data["v2"] if r["step"] == 0]
    if initial:
        left.scatter([initial[0]["epoch"]], [initial[0]["nll"]], facecolors="none", edgecolors=COLORS["v2"],
                     s=55, label="V2 step 0: diagnostic only", zorder=5)
    selected = [r for r in data["v2"] if r["step"] == data.get("selected_step")]
    if selected:
        left.scatter([selected[0]["epoch"]], [selected[0]["nll"]], marker="*", s=170,
                     color=COLORS["v2"], edgecolors="white", linewidth=.6, label="V2 selected on validation", zorder=6)
    final = [r for r in data["v1"] if r["final_original_baseline"]]
    if final:
        left.scatter([final[0]["epoch"]], [final[0]["nll"]], marker="s", s=62, facecolors="none",
                     edgecolors=COLORS["v1"], linewidths=1.5, label="V1 original final checkpoint", zorder=6)
    if all(r["nll"] > 0 for key in ("v1", "v2") for r in data[key]):
        from matplotlib.ticker import LogLocator, FuncFormatter
        left.set_yscale("log")
        left.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        left.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        left.set_ylabel("OOS-standardized raw validation NLL (log scale)")
    else:
        left.set_ylabel("OOS-standardized raw validation NLL")
    left.set_title("Validation objective · lower is better", loc="left", fontweight="bold")
    left.set_xlabel("Training epoch")
    left.legend(loc="best", fontsize=8, frameon=False)
    recorded = [r for r in data["training_epochs"] if finite(r.get("mean_nll"))]
    if recorded:
        xs = [r["epoch"] if r.get("complete") else data["v2_last_epoch_fraction"] for r in recorded]
        right.plot(xs, [r["mean_nll"] for r in recorded], color=COLORS["v2"], marker="o", linewidth=1.8,
                   label="V2 actual-example-weighted epoch mean")
        right.legend(loc="best", fontsize=8, frameon=False)
    else:
        right.text(.5, .55, "Epoch-mean training losses pending", ha="center", va="center",
                   transform=right.transAxes, color="#56616D")
    right.set_title("Training objective · separate scale/population", loc="left", fontweight="bold")
    right.set_xlabel("Completed epoch (fraction for final partial epoch)")
    right.set_ylabel("Unweighted raw training NLL")
    for ax in axes:
        ax.grid(alpha=.2, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Recorded learning curves", x=.065, ha="left", fontsize=16, fontweight="bold")
    fig.text(.065, .01, "Validation: fixed tuning queries/candidate order; mean of first and expanded NLL; OOS prior 2/11; no temperature.\n"
             "Training: observed sample mixture; never overlaid with validation. V1 epoch means were not recorded. "
             + ("V2 selection is final." if selected else "V2 is pending/in progress; no final selection."), fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, .09, 1, .94))
    save_figure(fig, output, "learning_curves", plt)
    return True


def plot_utility(data, output, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 9.0), sharey=True, sharex=True)
    values = [r["utility"] for panel in data for r in panel["rows"] if r["utility"] is not None]
    lower, upper = min(0, min(values)) - .09, max(0, max(values)) + .12
    positions, separators, offset = [], [], 0.
    for index, row in enumerate(data[0]["rows"]):
        if index and row["group"] != data[0]["rows"][index - 1]["group"]:
            separators.append(index + offset - .175)
            offset += .65
        positions.append(index + offset)
    for ax, panel in zip(axes, data):
        for position, row in zip(positions, panel["rows"]):
            value = row["utility"]
            if value is None:
                ax.text(.97, position, "pending", ha="right", va="center", color="#9299A2",
                        fontsize=8, transform=ax.get_yaxis_transform())
            else:
                ax.barh(position, value, color=COLORS[row["key"]], height=.57)
                ax.text(value + (.012 if value >= 0 else -.012), position, f"{value:.3f}",
                        va="center", ha="left" if value >= 0 else "right", fontsize=8)
        for separator in separators:
            ax.axhline(separator, color="#D5DAE0", linewidth=.7)
        ax.set_title(panel["title"], loc="left", fontweight="bold", fontsize=11)
        ax.axvline(0, color="#7B8490", linewidth=.8)
        ax.set_xlim(lower, upper)
        ax.set_xlabel("Mean utility per arrival")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(positions, [r["label"] for r in data[0]["rows"]], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Utility under declared costs · separate benchmark populations", x=.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(.06, .035, "Correct +1; wrong −5; defer −0.2; expand −0.05. Paired intervals: A100_RESULTS.md (V1), extended-summary.md (V2).\n"
             "* MiniLM direct: CLINC has supervision for all 150 intents; BANKING77 is trained on labeled catalog examples and routes automatically.\n"
             "† V2 follows inspection of V1 test outcomes; repeated benchmark results are exploratory. Fixed method order; no aggregate ranking.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, .12, 1, .94))
    save_figure(fig, output, "utility_comparison", plt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs/a100.json")
    args = parser.parse_args(argv)
    inputs = Inputs()
    learning = learning_data(inputs, args.baseline, args.followup, args.baseline_config)
    utility = utility_data(inputs, args.baseline, args.followup)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Install the optional report dependency: python -m pip install matplotlib==3.10.3") from error
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.labelcolor": "#283442", "text.color": "#283442"})
    args.output.mkdir(parents=True, exist_ok=True)
    rendered_learning = plot_learning(learning, args.output, plt)
    plot_utility(utility, args.output, plt)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "recorded_artifact_snapshot",
               "script_sha256": digest(__file__), "matplotlib_version": matplotlib.__version__,
               "sources": inputs.sources, "notes": inputs.notes, "learning": learning, "utility": utility,
               "learning_plot_rendered": rendered_learning,
               "limitations": ["No missing results are treated as zero or as a win", "No test metrics select a checkpoint",
                               "V1 checkpoint curve is retrospective; original final model stays the v1 baseline",
                               "V2 repeated benchmark comparisons are exploratory after v1 inspection",
                               "Utility bars show point estimates, not uncertainty or a total ranking"]}
    (args.output / "figure_data.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    caption = """# Figure captions and reproduction

**Learning curves — measured snapshot.** The left panel shows raw FP32 validation NLL, with equal weight for first/expanded stages and each stage standardized to the fixed OOS prior 2/11. NONE remains in the distribution, and no temperature scaling enters selection. Candidate order and validation data identity are fixed. V1 checkpoints are scored retrospectively; the square marks its original final baseline, which is not replaced by a retrospectively better checkpoint. A hollow V2 step-zero point is diagnostic only and cannot be selected. The star appears only when a completed V2 training summary identifies the selected positive-step checkpoint. Missing later epochs are not extrapolated.

**Training loss — different population.** The right panel separately shows actual-example-weighted V2 epoch means under the training sample mixture. These are not standardized to the validation OOS prior. V1 epoch means were not recorded; its final minibatch loss is never substituted for an epoch mean. A partial final epoch is plotted at its recorded fraction.

**Utility comparison — measured point estimates.** Each panel is a separate benchmark population under the same declared costs. V1 and V2 are shown in separate groups, each with its own RL, confidence-gate and never-expand policies. V2 also includes its seed-42 exact optimizer, full-feedback policy and seed-42 fixed-cost actor; comparator models follow. Fixed-cost describes how that actor was trained; all bars use the same primary evaluation costs within each panel. Bars are not a pooled leaderboard and include no invented uncertainty. Read paired confidence intervals and transfer limitations in the [V1 report](../A100_RESULTS.md) and [V2 follow-up summary](../extended-summary.md); absent V2/Jev evaluations remain pending. Completed V2 results must match the selected model fingerprint in the training summary before they can be plotted. The MiniLM direct classifier has labeled supervision for all 150 CLINC intents, including when the router catalog has only 120, and domain labels through BANKING77 catalog examples. V2 follows inspection of V1 test outcomes, so repeated-test comparisons are exploratory. No claim about real business savings follows from simulated utility.

Each figure is supplied as 240-dpi PNG and editable-text SVG. `figure_data.json` retains plotted values, availability status, input hashes and renderer version. Rerun after downloading completed follow-up artifacts:

```bash
python reports/build_figures.py --baseline runs/a100-001 --followup runs/a100-extended-001 --output reports/figures
```
"""
    (args.output / "CAPTIONS.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "learning_plot": rendered_learning,
                      "v2_training_status": learning["v2_status"], "sources": len(inputs.sources)}))


if __name__ == "__main__":
    main()
