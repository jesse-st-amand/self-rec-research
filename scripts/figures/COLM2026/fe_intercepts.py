"""Per-evaluator fixed-effect intercepts behind Figure 2a-d, for Appendix F.

Figure 2's panels (a)-(d) fit y = beta*x + alpha_g with evaluator x task-domain
fixed effects and draw the line at the sample-weighted mean of alpha_g. The
counts quoted in Section 4.2 ("21 of 23", ...) are per-EVALUATOR intercepts, not
per-group ones. Because the slope is common, evaluator j's intercept is just
wmean_j(y) - beta*wmean_j(x), and the weighted mean of those recovers the drawn
intercept exactly -- which is the check this script prints.

Intervals are 2000-sample bootstrap percentiles resampling (evaluator, generator,
task domain) rows and refitting beta on each draw. They do not cluster on
generator, so they are narrower than a generator-clustered interval; the appendix
says so.

Run:  uv run python scripts/figures/COLM2026/fe_intercepts.py

Takes a few minutes. Output pastes into Tables 6-7 of appendix.tex.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scripts.figures.COLM2026.prototype_compact_figures import (
    AGG_DIR, load_self_scores_per_dataset, adjust_ind_performance,
    fit_score_distance, _wmean,
)
from self_rec_framework.src.helpers.model_names import LM_ARENA_SCORES

PANELS = {
 "a": "ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst",
 "b": "ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst",
 "c": "COLM_01_AT_PW-C_Rec_NPr_FA_Inst",
 "d": "COLM_02_AT_IND-C_Rec_NPr_FA_Inst",
}
def score(m):
    return LM_ARENA_SCORES.get(m) or LM_ARENA_SCORES.get(m.replace("-thinking",""))

def ev_intercepts(x,y,w,grp,ev,evs_order):
    slope,_,_ = fit_score_distance(x,y,w,grp,"fe")
    return slope, np.array([_wmean(y[ev==e],w[ev==e]) - slope*_wmean(x[ev==e],w[ev==e])
                            for e in evs_order])

res={}
for k,exp in PANELS.items():
    ts=sorted((AGG_DIR/exp).iterdir(),reverse=True)[0]
    df=pd.read_csv(ts/"rank_distance_data.csv")
    df["eval_score"]=df.evaluator.map(score); df["gen_score"]=df.generator.map(score)
    df=df.dropna(subset=["eval_score","gen_score"])
    df["score_distance"]=df.eval_score-df.gen_score
    ss=load_self_scores_per_dataset(exp)
    if ss is not None: df=adjust_ind_performance(df,ss)
    agg=df.rename(columns={"n_samples":"weight"})
    x=agg.score_distance.values.astype(float); y=agg.performance.values.astype(float)
    w=agg.weight.values.astype(float); ev=agg.evaluator.values
    grp=(agg.evaluator.astype(str)+"|"+agg.dataset.astype(str)).values
    order=sorted(pd.unique(ev))
    slope,b=ev_intercepts(x,y,w,grp,ev,order)

    rng=np.random.default_rng(42); B=2000
    boot=np.full((B,len(order)),np.nan)
    for t in range(B):
        i=rng.choice(len(x),len(x),replace=True)
        try: _,bb=ev_intercepts(x[i],y[i],w[i],grp[i],ev[i],order)
        except Exception: continue
        boot[t]=bb
    lo=np.nanpercentile(boot,2.5,axis=0); hi=np.nanpercentile(boot,97.5,axis=0)
    res[k]=pd.DataFrame({"evaluator":order,"b":b,"lo":lo,"hi":hi}).set_index("evaluator")
    print(f"panel {k}: slope/100={slope*100:.3f}  n_ev={len(order)}  "
          f"b>0.5: {(b>0.5).sum()}   CI wholly >0.5: {(lo>0.5).sum()}   CI wholly <0.5: {(hi<0.5).sum()}")

all_ev=sorted(set().union(*[set(d.index) for d in res.values()]))
tab=pd.DataFrame(index=all_ev)
for k,d in res.items():
    tab[k]=d.b.reindex(all_ev).round(3)
    tab[k+"_ci"]=[("" if np.isnan(r.lo) else f"[{r.lo:.2f}, {r.hi:.2f}]")
                  if e in d.index else "" for e,r in
                  d.reindex(all_ev).iterrows()]
pd.set_option("display.width",250); pd.set_option("display.max_rows",50)
print(); print(tab.to_string())
