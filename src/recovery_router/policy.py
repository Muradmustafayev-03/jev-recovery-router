"""Cost-conditioned two-step RL with a separate, frozen probabilistic belief model.

Stage zero never receives future-state features. Counterfactual caches are used
by the offline environment only; expansion must be selected before observation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file
from sklearn.ensemble import ExtraTreesRegressor

from .calibration import apply_calibration, checked_probs, mixture_weights
from .io import read_json, read_jsonl, write_json, join_predictions, sha256_file

COMMIT, EXPAND, DEFER = 0, 1, 2
FEATURE_NAMES = ["top_p", "second_p", "margin", "none_p", "normalized_entropy", "candidate_count_div_20",
                 "retrieval_top_score", "retrieval_score_gap", "retrieval_mean_score", "expanded_stage",
                 "wrong_cost_div_10", "defer_cost", "expand_cost", "correct_reward"]


@dataclass(frozen=True)
class Costs:
    correct: float = 1.0
    wrong: float = 5.0
    defer: float = 0.2
    expand: float = 0.05

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Costs/reward must be finite and nonnegative")


def state_features(row, pred, expanded=False, costs=Costs()):
    candidates = row["expanded_candidates" if expanded else "candidates"]
    p = checked_probs(pred["expanded_probs" if expanded else "first_probs"],len(candidates))
    scores = sorted([float(c.get("score",0.0)) for c in candidates],reverse=True)
    ranked = np.sort(p[:-1])[::-1]
    second = ranked[1] if len(ranked)>1 else 0.
    values = [ranked[0],second,ranked[0]-second,p[-1],-np.sum(p*np.log(np.clip(p,1e-12,1)))/np.log(len(p)),
              len(candidates)/20,scores[0],scores[0]-(scores[1] if len(scores)>1 else 0.),np.mean(scores),float(expanded),
              costs.wrong/10,costs.defer,costs.expand,costs.correct]
    return np.asarray(values,dtype=np.float32)


def commit_result(row,pred,expanded=False):
    candidates=row["expanded_candidates" if expanded else "candidates"]
    p=checked_probs(pred["expanded_probs" if expanded else "first_probs"],len(candidates))
    index=int(p[:-1].argmax())  # NONE mass is kept in p, never renormalized away.
    label=candidates[index]["label"]
    correct=not row["is_oos"] and label==row["label"]
    return label,float(p[index]),bool(correct)


def correctness(rows,predictions,expanded=False):
    return np.asarray([commit_result(r,p,expanded)[2] for r,p in join_predictions(rows,predictions)],dtype=np.float32)


class Actor(nn.Module):
    def __init__(self,hidden=48):
        super().__init__()
        self.hidden=hidden
        self.trunk=nn.Sequential(nn.Linear(len(FEATURE_NAMES),hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.actions=nn.Linear(hidden,3)
        self.value=nn.Linear(hidden,1)

    def forward(self,features):
        h=self.trunk(features)
        logits=self.actions(h)
        mask=torch.zeros_like(logits,dtype=torch.bool)
        mask[:,EXPAND]=features[:,9]>0.5
        logits=logits.masked_fill(mask,torch.finfo(logits.dtype).min)
        return logits,self.value(h).squeeze(-1)

    def choose(self,features):
        self.eval()
        if getattr(self,"serving_cost_override",None):
            features=condition_features(features,Costs(**self.serving_cost_override))
        with torch.no_grad(): return self(torch.as_tensor(features,dtype=torch.float32))[0].argmax(-1).numpy()


def base_features(rows,predictions,expanded=False):
    return np.stack([state_features(r,p,expanded) for r,p in join_predictions(rows,predictions)])


def condition_features(features,costs):
    out=np.array(features,copy=True)
    out[:,10:]=[costs.wrong/10,costs.defer,costs.expand,costs.correct]
    return out


def train_actor(rows,predictions,config,seed=42,full_information=False):
    """Sampled REINFORCE with learned baseline; no exact action rewards in actor loss."""
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    actor=Actor(); optimizer=torch.optim.Adam(actor.parameters(),lr=config.get("learning_rate",.001))
    f0=base_features(rows,predictions,False); f1=base_features(rows,predictions,True)
    ok0=correctness(rows,predictions,False); ok1=correctness(rows,predictions,True)
    weights=mixture_weights(rows,config.get("target_oos_prior",2/11)); weights=weights/weights.sum()
    grid=list(product(config.get("wrong_costs",[2,5,10]),config.get("defer_costs",[.1,.2,.4]),config.get("expand_costs",[.01,.05,.15])))
    logs=[]; started=time.monotonic()
    for step in range(config.get("steps",1000)):
        ix=rng.choice(len(rows),size=config.get("batch_size",256),p=weights)
        cost=Costs(wrong=grid[step%len(grid)][0],defer=grid[step%len(grid)][1],expand=grid[step%len(grid)][2])
        x0=torch.from_numpy(condition_features(f0[ix],cost)); x1=torch.from_numpy(condition_features(f1[ix],cost))
        logits0,v0=actor(x0); d0=torch.distributions.Categorical(logits=logits0); a0=d0.sample()
        # The environment has cached x1, but only selected expansion trajectories
        # contribute second-state policy loss/observations. No x1 is in x0.
        logits1,v1=actor(x1); d1=torch.distributions.Categorical(logits=logits1); a1=d1.sample()
        commit0=torch.from_numpy(np.where(ok0[ix]>0,cost.correct,-cost.wrong).astype(np.float32))
        commit1=torch.from_numpy(np.where(ok1[ix]>0,cost.correct,-cost.wrong).astype(np.float32))
        r1=torch.where(a1==COMMIT,commit1,-cost.defer)
        expanded=(a0==EXPAND)
        r0=torch.where(a0==COMMIT,commit0,torch.where(expanded,r1-cost.expand,-cost.defer))
        if full_information:
            # Matched architecture, exact full-information expected return. This
            # low-variance comparator must not be called sampled RL.
            probs0=d0.probs; probs1=d1.probs
            expected1=probs1[:,COMMIT]*commit1-probs1[:,DEFER]*cost.defer
            expected0=probs0[:,COMMIT]*commit0+probs0[:,EXPAND]*(expected1-cost.expand)-probs0[:,DEFER]*cost.defer
            entropy=d0.entropy()
            loss=(-expected0-config.get("entropy_weight",.005)*entropy).mean()
        else:
            pg0=-d0.log_prob(a0)*(r0-v0).detach()
            pg1=-d1.log_prob(a1)*(r1-v1).detach()*expanded
            critic=(v0-r0.detach()).square()+expanded*(v1-r1.detach()).square()
            entropy=d0.entropy()
            loss=(pg0+pg1+.5*critic-config.get("entropy_weight",.005)*entropy).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); optimizer.step()
        if step%100==0 or step==config.get("steps",1000)-1:
            logs.append({"step":step,"loss":float(loss.detach()),"sampled_return":float(r0.mean()),"expanded_fraction":float(expanded.float().mean())})
    return actor,{"seed":seed,"seconds":time.monotonic()-started,"training_rows":len(rows),"target_oos_prior":config.get("target_oos_prior",2/11),
                  "features":FEATURE_NAMES,"cost_grid":grid,"logs":logs,"method":"exact_full_information_expected_utility" if full_information else "two_step_REINFORCE_with_value_baseline"}


def save_actor(actor,path,metadata):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in actor.state_dict().items()},str(path/"policy.safetensors"))
    write_json(path/"policy.json",dict(metadata,hidden=actor.hidden,feature_names=FEATURE_NAMES))


def load_actor(path):
    path=Path(path); meta=read_json(path/"policy.json")
    if meta["feature_names"]!=FEATURE_NAMES: raise ValueError("Policy feature schema differs")
    actor=Actor(meta["hidden"]); actor.load_state_dict(load_file(str(path/"policy.safetensors")))
    actor.serving_cost_override=meta.get("serving_cost_override")
    actor.eval(); return actor


def fit_full_feedback(rows,predictions,target_oos_prior=2/11,seed=42,config=None):
    """Regress counterfactual action returns, including contingent deferral at s1.

    Targets use a fixed second-stage rule on frozen belief forecasts. We regress
    E[realized continuation return | s0,cost], not max(E[correctness],defer),
    preserving the value of observing the expanded state.
    """
    config=config or {}
    weights=mixture_weights(rows,target_oos_prior)
    y0=correctness(rows,predictions,False); y1=correctness(rows,predictions,True)
    x0=base_features(rows,predictions,False)
    p1=np.array([commit_result(r,p,True)[1] for r,p in join_predictions(rows,predictions)])
    xs=[]; ys=[]
    grid=list(product(config.get("wrong_costs",[2,5,10]),config.get("defer_costs",[.1,.2,.4]),config.get("expand_costs",[.01,.05,.15])))
    for wrong,defer,expand in grid:
        cost=Costs(wrong=wrong,defer=defer,expand=expand)
        current=np.where(y0>0,1.,-wrong)
        continuation=np.where(p1>=(wrong-defer)/(1+wrong),np.where(y1>0,1.,-wrong),-defer)-expand
        xs.append(condition_features(x0,cost)); ys.append(np.stack([current,continuation,np.full(len(rows),-defer)],axis=1))
    model=ExtraTreesRegressor(n_estimators=80,max_depth=12,min_samples_leaf=30,n_jobs=2,random_state=seed)
    model.fit(np.concatenate(xs),np.concatenate(ys),sample_weight=np.tile(weights,len(grid)))
    return model


def recovery_case(row):
    if row["is_oos"]: return "out_of_scope"
    if row["label"] in [c["label"] for c in row["candidates"]]: return "present_initially"
    if row["label"] in [c["label"] for c in row["expanded_candidates"]]: return "recoverable_candidate"
    return "absent_after_expansion"


def outcomes(rows,predictions,actions0,actions1,costs=Costs()):
    output=[]
    for (row,pred),a0,a1 in zip(join_predictions(rows,predictions),actions0,actions1):
        expanded=int(a0)==EXPAND; action=int(a1) if expanded else int(a0)
        if action not in (COMMIT,DEFER): raise ValueError("Terminal action must commit or defer")
        label,confidence,correct=commit_result(row,pred,expanded)
        first_would_be_correct=commit_result(row,pred,False)[2]
        expanded_would_be_correct=commit_result(row,pred,True)[2]
        accepted=action==COMMIT
        utility=(costs.correct if correct else -costs.wrong) if accepted else -costs.defer
        if expanded: utility-=costs.expand
        t0=pred.get("first_ms"); t1=pred.get("expanded_ms")
        elapsed=float(t0)+(float(t1) if expanded else 0) if t0 is not None and (not expanded or t1 is not None) else None
        output.append({"id":row["id"],"parent_id":row.get("parent_id",row["id"]),"group":row.get("group","unknown"),
                       "intervention":row.get("intervention","natural"),"recovery_case":recovery_case(row),"is_oos":bool(row["is_oos"]),
                       "accepted":accepted,"expanded":expanded,"prediction":label if accepted else None,"correct":correct if accepted else False,
                       "first_would_be_correct":first_would_be_correct,"expanded_would_be_correct":expanded_would_be_correct,
                       "confidence":confidence,"utility":float(utility),"accounted_ms":elapsed})
    return output


def run_policy(kind,rows,predictions,costs=Costs(),actor=None,full_feedback=None,threshold=.8):
    pairs=join_predictions(rows,predictions); n=len(rows)
    p0=np.array([commit_result(r,p,False)[1] for r,p in pairs]); p1=np.array([commit_result(r,p,True)[1] for r,p in pairs])
    pstar=(costs.wrong-costs.defer)/(costs.correct+costs.wrong) if costs.correct+costs.wrong else 1.
    a1=np.where(p1>=pstar,COMMIT,DEFER)
    if kind=="never_expand": a0=np.where(p0>=pstar,COMMIT,DEFER)
    elif kind=="always_expand": a0=np.full(n,EXPAND)
    elif kind=="confidence_gate": a0=np.where(p0>=threshold,COMMIT,EXPAND)
    elif kind=="actor":
        if actor is None: raise ValueError("Missing actor")
        a0=actor.choose(condition_features(base_features(rows,predictions,False),costs))
        a1=actor.choose(condition_features(base_features(rows,predictions,True),costs))
    elif kind=="full_feedback":
        if full_feedback is None: raise ValueError("Missing fitted full-feedback policy")
        estimated=full_feedback.predict(condition_features(base_features(rows,predictions,False),costs))
        a0=estimated.argmax(1)
    elif kind=="oracle_upper_bound":
        # Ground truth allowed ONLY for this explicitly labeled nondeployable ceiling.
        now=np.where(correctness(rows,predictions,False)>0,costs.correct,-costs.wrong)
        after=np.where(correctness(rows,predictions,True)>0,costs.correct,-costs.wrong)
        a0=np.stack([now,np.maximum(after,-costs.defer)-costs.expand,np.full(n,-costs.defer)],axis=1).argmax(1)
        a1=np.where(after>=-costs.defer,COMMIT,DEFER)
    else: raise ValueError(f"Unknown policy {kind}")
    return outcomes(rows,predictions,a0,a1,costs)


def select_threshold(rows,predictions,costs,target_prior):
    weights=mixture_weights(rows,target_prior)
    scores=[]
    for threshold in np.linspace(.3,.99,24):
        out=run_policy("confidence_gate",rows,predictions,costs,threshold=float(threshold))
        scores.append((float(np.average([r["utility"] for r in out],weights=weights)),float(threshold)))
    return max(scores)[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True); parser.add_argument("--predictions",required=True)
    parser.add_argument("--calibration",required=True); parser.add_argument("--config",default="configs/h100.json")
    parser.add_argument("--output",required=True); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--full-information",action="store_true",help="Train matched-architecture exact expected utility comparator")
    args=parser.parse_args()
    rows=read_jsonl(args.data); calibration=read_json(args.calibration)
    preds=[apply_calibration(p,calibration) for p in read_jsonl(args.predictions)]
    config=read_json(args.config)["policy"]
    actor,metadata=train_actor(rows,preds,config,args.seed,args.full_information)
    metadata.update(data_sha256=sha256_file(args.data),predictions_sha256=sha256_file(args.predictions),calibration_sha256=sha256_file(args.calibration))
    save_actor(actor,args.output,metadata)
    print(f"Saved actor to {args.output}; {metadata['seconds']:.1f}s")


if __name__=="__main__": main()
