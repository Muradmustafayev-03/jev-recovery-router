"""Metrics and paired uncertainty. No invented measured GPU latency."""
from __future__ import annotations
import numpy as np
from .calibration import checked_probs, target_index
from .io import join_predictions


def probability_metrics(rows, predictions, expanded=False, bins=10):
    key="expanded" if expanded else "first"
    nll=[]; brier=[]; correctness=[]; confidences=[]
    for row,pred in join_predictions(rows,predictions):
        p=checked_probs(pred[key+"_probs"],len(row["expanded_candidates" if expanded else "candidates"]))
        y=target_index(row,expanded)
        target=np.zeros_like(p); target[y]=1
        nll.append(-np.log(max(p[y],1e-12)))
        brier.append(np.square(p-target).sum())
        correctness.append(int(p.argmax()==y)); confidences.append(float(p.max()))
    if not nll:
        return {"n":0}
    confidences=np.asarray(confidences); correctness=np.asarray(correctness)
    ece=0.
    for i in range(bins):
        mask=(confidences>=i/bins)&((confidences<(i+1)/bins) if i<bins-1 else (confidences<=1))
        if mask.any(): ece+=mask.mean()*abs(confidences[mask].mean()-correctness[mask].mean())
    return {"n":len(nll),"nll":float(np.mean(nll)),"multiclass_brier_sum":float(np.mean(brier)),"ece":float(ece),
            "ece_bins":bins,"ece_binning":"equal-width confidence bins on [0,1]", "accuracy_including_none":float(correctness.mean())}


def grouped_probability_metrics(rows, raw_predictions, calibrated_predictions):
    """Describe calibration by catalog/OOS and per-stage candidate membership.

    These labels partition evaluation results only; they fit no calibrator and
    are never passed to a deployed policy.
    """
    raw_by_id={r["id"]:p for r,p in join_predictions(rows,raw_predictions)}
    calibrated_by_id={r["id"]:p for r,p in join_predictions(rows,calibrated_predictions)}

    def scores(selected):
        raw=[raw_by_id[r["id"]] for r in selected]
        calibrated=[calibrated_by_id[r["id"]] for r in selected]
        return {"raw_initial":probability_metrics(selected,raw),
                "scaled_initial":probability_metrics(selected,calibrated),
                "raw_expanded":probability_metrics(selected,raw,True),
                "scaled_expanded":probability_metrics(selected,calibrated,True)}

    result={"group":{group:scores([r for r in rows if r.get("group","unknown")==group])
                     for group in sorted({r.get("group","unknown") for r in rows})},
            "catalog_membership":{name:scores([r for r in rows if bool(r["is_oos"])==is_oos])
                                  for name,is_oos in (("in_scope",False),("out_of_scope",True))},
            "candidate_membership":{},
            "scope":"Descriptive held-out subgroups; no test-fitted calibration or prior correction."}
    for stage,key in (("initial","candidates"),("expanded","expanded_candidates")):
        selected={"gold_present":[],"in_scope_gold_absent":[],"out_of_scope":[]}
        for row in rows:
            group=("out_of_scope" if row["is_oos"] else
                   "gold_present" if any(c["label"]==row["label"] for c in row[key]) else "in_scope_gold_absent")
            selected[group].append(row)
        result["candidate_membership"][stage]={name:scores(partition) for name,partition in selected.items()}
    return result


def expansion_diagnostics(outcomes):
    """Ex-post selector changes among expansions actually chosen by a policy."""
    expanded=[r for r in outcomes if r["expanded"]]
    if any("first_would_be_correct" not in r or "expanded_would_be_correct" not in r for r in expanded):
        return {"status":"unavailable_counterfactual_selector_correctness", "expanded_n":len(expanded)}
    counts={"helpful_selector_change":0,"harmful_selector_change":0,
            "unnecessary_ex_post_both_correct":0,"unresolved_both_wrong":0}
    accepted_recoveries=accepted_harms=0
    for row in expanded:
        before,after=row["first_would_be_correct"],row["expanded_would_be_correct"]
        name=("unnecessary_ex_post_both_correct" if before and after else "harmful_selector_change" if before else
              "helpful_selector_change" if after else "unresolved_both_wrong")
        counts[name]+=1
        accepted_recoveries+=int(row["accepted"] and not before and after)
        accepted_harms+=int(row["accepted"] and before and not after)
    return {"status":"measured_ex_post_selector_diagnostic","expanded_n":len(expanded),"counts":counts,
            "fractions_of_expanded":{key:value/len(expanded) if expanded else None for key,value in counts.items()},
            "accepted_recovery_n":accepted_recoveries,"accepted_harm_n":accepted_harms,
            "expanded_then_deferred_n":sum(not r["accepted"] for r in expanded),
            "definition":"Potential top-candidate commitment correctness before/after expansion, among queries actually expanded. Unnecessary means both candidates would be correct; it does not establish zero information value, zero calibration benefit, or causal utility harm."}


def summarize(outcomes):
    if not outcomes: return {"n":0}
    accepted=[x for x in outcomes if x["accepted"]]
    oos=[x for x in outcomes if x["is_oos"]]
    latencies=[x["accounted_ms"] for x in outcomes if x.get("accounted_ms") is not None]
    return {"n":len(outcomes),"correct_route_rate":float(np.mean([x["accepted"] and x["correct"] for x in outcomes])),
            "wrong_route_rate":float(np.mean([x["accepted"] and not x["correct"] for x in outcomes])),
            "coverage":len(accepted)/len(outcomes),"accepted_error":float(np.mean([not x["correct"] for x in accepted])) if accepted else None,
            "handoff_rate":1-len(accepted)/len(outcomes),"expand_rate":float(np.mean([x["expanded"] for x in outcomes])),
            "oos_false_acceptance":float(np.mean([x["accepted"] for x in oos])) if oos else None,
            "mean_utility":float(np.mean([x["utility"] for x in outcomes])),
            "expansion_diagnostics":expansion_diagnostics(outcomes),
            "accounted_model_ms_p50":float(np.percentile(latencies,50)) if latencies else None,
            "accounted_model_ms_p95":float(np.percentile(latencies,95)) if latencies else None,
            "latency_note":"Sum of cached inference component times for actions actually taken; not an online serving benchmark. Retrieval/queue overhead is excluded unless explicitly recorded."}


def grouped_summary(outcomes):
    result={"overall":summarize(outcomes)}
    for key in ["group","recovery_case","intervention"]:
        result[key]={str(value):summarize([x for x in outcomes if x.get(key)==value]) for value in sorted({x.get(key,"unknown") for x in outcomes})}
    return result


def standardized_summary(outcomes, target_oos_prior):
    """Secondary mixture-standardized rates; never invent absent OOS cases."""
    oos=[r for r in outcomes if r["is_oos"]]; inscope=[r for r in outcomes if not r["is_oos"]]
    if not oos or not inscope:
        return {"status":"not_estimable_missing_mixture_group"}
    weights=np.array([target_oos_prior/len(oos) if r["is_oos"] else (1-target_oos_prior)/len(inscope) for r in outcomes])
    average=lambda values:float(np.dot(weights,np.asarray(values)))
    coverage=average([r["accepted"] for r in outcomes])
    wrong=average([r["accepted"] and not r["correct"] for r in outcomes])
    return {"target_oos_prior":target_oos_prior,"mean_utility":average([r["utility"] for r in outcomes]),
            "coverage":coverage,"wrong_route_rate":wrong,"accepted_error":wrong/coverage if coverage else None,
            "correct_route_rate":average([r["accepted"] and r["correct"] for r in outcomes]),
            "expand_rate":average([r["expanded"] for r in outcomes]),
            "status":"secondary_assumed_mixture_standardization; primary paired intervals use empirical benchmark mixture"}


def candidate_diagnostics(rows,predictions):
    """Separate candidate availability from selection quality and recovery."""
    present=[[],[]]; correct=[[],[]]; supported=[]
    for row,pred in join_predictions(rows,predictions):
        supported.append(not row["is_oos"])
        for stage,(key,candidates_key) in enumerate((("first","candidates"),("expanded","expanded_candidates"))):
            candidates=row[candidates_key]; p=checked_probs(pred[key+"_probs"],len(candidates))
            present[stage].append(not row["is_oos"] and any(c["label"]==row["label"] for c in candidates))
            correct[stage].append(not row["is_oos"] and candidates[int(p[:-1].argmax())]["label"]==row["label"])
    present=np.asarray(present,dtype=bool); correct=np.asarray(correct,dtype=bool); supported=np.asarray(supported)
    result={"n":len(rows),"supported_n":int(supported.sum()),"scope":"counterfactual candidate/selector diagnostic; ignores routing acceptance"}
    for i,key in enumerate(("first","expanded")):
        result[key]={"candidate_recall_in_scope":float(present[i,supported].mean()) if supported.any() else None,
                     "selector_accuracy_when_present":float(correct[i,present[i]].mean()) if present[i].any() else None,
                     "candidate_present_n":int(present[i].sum())}
    result.update(helpful_selector_changes=int((~correct[0]&correct[1]).sum()),harmful_selector_changes=int((correct[0]&~correct[1]).sum()),
                  recoverable_candidate_n=int((~present[0]&present[1]).sum()))
    return result


def paired_bootstrap(left,right,samples=1000,seed=2026):
    right_by_id={x["id"]:x for x in right}
    if set(right_by_id)!={x["id"] for x in left}: raise ValueError("Paired comparison needs the same IDs")
    clusters={}
    for row in left:
        clusters.setdefault(row.get("parent_id",row["id"]),[]).append(row["utility"]-right_by_id[row["id"]]["utility"])
    values=[np.asarray(v) for v in clusters.values()]
    sums=np.array([v.sum() for v in values]); counts=np.array([len(v) for v in values])
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(samples):
        ix=rng.integers(0,len(values),len(values)); diffs.append(sums[ix].sum()/counts[ix].sum())
    return {"metric":"paired_mean_utility_difference", "difference":float(sums.sum()/counts.sum()),
            "ci95":[float(x) for x in np.quantile(diffs,[.025,.975])],"resampling_unit":"parent_query", "bootstrap_samples":samples,
            "limitation":"Descriptive interval; no multiple-comparison correction or guarantee under deployment shift."}


def grouped_paired_bootstrap(left,right,samples=1000,seed=2026):
    """Exploratory main-versus-gate intervals within predefined data groups."""
    right_by_id={r["id"]:r for r in right}
    if set(right_by_id)!={r["id"] for r in left}:
        raise ValueError("Paired comparison needs the same IDs")
    result={}
    for key in ("group","recovery_case","intervention"):
        result[key]={}
        for value in sorted({r.get(key,"unknown") for r in left}):
            selected=[r for r in left if r.get(key,"unknown")==value]
            result[key][value]=dict(paired_bootstrap(selected,[right_by_id[r["id"]] for r in selected],samples,seed),
                                    n=len(selected),analysis_status="exploratory_subgroup_no_multiplicity_adjustment")
    return result


def actor_seed_summary(methods):
    """Descriptive actor-seed variability, holding backbone and split fixed."""
    result={}
    for family in ("rl","exact"):
        utilities={name:float(np.mean([r["utility"] for r in rows])) for name,rows in methods.items()
                   if name.startswith(family+"_seed") and rows}
        values=list(utilities.values())
        result[family]={"n_actor_seeds":len(values),"per_seed_mean_utility":utilities,
                        "mean_utility_across_actor_seeds":float(np.mean(values)) if values else None,
                        "sample_std_across_actor_seeds":float(np.std(values,ddof=1)) if len(values)>1 else None,
                        "scope":"Descriptive variability across actor initializations only; fixed belief model, dataset and calibration. Not backbone-training uncertainty or a confidence interval."}
    return result


def accepted_subset_pruning_curve(outcomes):
    """Prune an existing accepted set, never extend it into deferred cases."""
    accepted=sorted([x for x in outcomes if x["accepted"]],key=lambda x:-x["confidence"])
    result=[]
    for fraction in [0.1,0.25,0.5,0.75,1.0]:
        n=max(1,int(len(accepted)*fraction)) if accepted else 0
        subset=accepted[:n]
        result.append({"coverage":n/len(outcomes) if outcomes else 0,"accepted_error":float(np.mean([not x["correct"] for x in subset])) if n else None})
    return {"points":result,"original_policy_coverage":len(accepted)/len(outcomes) if outcomes else 0.,
            "description":"Sort only already-accepted queries by confidence and retain fractions of that subset. This is not a full risk-coverage curve and cannot measure performance when accepting formerly deferred queries."}


def risk_coverage(outcomes):
    """Legacy points-only alias; interpretation is accepted-subset pruning."""
    return accepted_subset_pruning_curve(outcomes)["points"]
