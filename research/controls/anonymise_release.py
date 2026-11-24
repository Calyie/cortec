"""Map a CoRTeC release of `adult` onto the `adult_anon` schema (post-processing of released numbers: no privacy
budget is spent) or map an `adult_anon` draw back to `adult` names for scoring with the standard evaluator.
  python anonymise_release.py release  <adult cohort_statistics.json> <out json>
  python anonymise_release.py draw     <adult_anon draw csv> <out csv>
  python anonymise_release.py prompt   <adult_anon cohort_statistics.json>     (prints a prompt and the leak check)"""
import json, re, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import pandas as pd
from src.dataset_spec import get_spec
import src.datasets_anon  # noqa: F401
from src.datasets_anon import anon_maps
m = anon_maps(); base = get_spec("adult")
def key_map(k: str, sep: str) -> str:
    return sep.join(m["bands"].get(p, p) for p in k.split(sep))
def anon_block(blk, col_map_only=False):
    out = {"numerical": {m["cols"][c]: v for c, v in blk.get("numerical", {}).items()},
           "categorical": {m["cols"][c]: {m["cats"][c][lab]: p for lab, p in d.items()} for c, d in blk.get("categorical", {}).items()}}
    return out
def anon_release(rel):
    out = []
    for r in rel:
        s = dict(r)
        s["cohort_name"] = key_map(r["cohort_name"], " & ")
        s.update(anon_block(r))
        s["class_balance"] = {m["cats"][base.target_col][k]: v for k, v in r["class_balance"].items()}
        if r.get("by_class"):
            s["by_class"] = {m["cats"][base.target_col][cls]: anon_block(blk) for cls, blk in r["by_class"].items()}
        for fk in ("_floored_categories",):
            if r.get(fk): s[fk] = {m["cols"][c]: [m["cats"][c][v] for v in vs] for c, vs in r[fk].items()}
        if r.get("_floored_categories_by_class"):
            s["_floored_categories_by_class"] = {m["cats"][base.target_col][cls]: {m["cols"][c]: [m["cats"][c][v] for v in vs] for c, vs in d.items()}
                                                 for cls, d in r["_floored_categories_by_class"].items()}
        s["conditional_target"] = {key_map(k, "|"): v for k, v in r["conditional_target"].items()}
        s["conditional_target_levels"] = {lv: {key_map(k, "|"): v for k, v in t.items()} for lv, t in r["conditional_target_levels"].items()}
        out.append(s)
    return out
def deanon_draw(df):
    df = df.rename(columns=m["cols_inv"])
    for c in base.categorical_cols + [base.target_col]:
        inv = m["cats_inv"][m["cols"][c]]; df[c] = df[c].astype(str).map(lambda v: inv.get(v, v))
    return df[base.column_names]
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "release":
        rel = json.load(open(sys.argv[2])); out = anon_release(rel)
        os.makedirs(os.path.dirname(sys.argv[3]), exist_ok=True); json.dump(out, open(sys.argv[3], "w"), indent=2)
        print(f"wrote {sys.argv[3]}: {len(out)} cohorts")
    elif cmd == "draw":
        df = pd.read_csv(sys.argv[2], keep_default_na=False); deanon_draw(df).to_csv(sys.argv[3], index=False); print(f"wrote {sys.argv[3]}: {len(df)} rows")
    elif cmd == "prompt":
        from src.generic_pipeline import build_cortec_prompt, batch_quotas
        import numpy as np
        spec = get_spec("adult_anon"); rel = json.load(open(sys.argv[2]))
        for cs in rel[:2]:
            q = batch_quotas(spec, cs, 25, np.random.default_rng(42), by_class=bool(cs.get("by_class")))
            p = build_cortec_prompt(spec, cs, n_rows=25, quotas=q)
            vocab = set(base.column_names) | {v for c in base.categorical_cols for v in m["cats"][c]} | set(m["bands"]) | {"Adult", "census", ">50K", "<=50K", "income", "UCI", "Married", "edu", "hours"}
            leaks = sorted(t for t in vocab if re.search(r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])", p))
            print(f"--- cohort {cs['cohort_name']}: {len(p)} chars; leaked real tokens: {leaks if leaks else 'NONE'}")
            if cs is rel[0]: print(p[:1800])
