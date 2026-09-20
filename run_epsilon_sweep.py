"""
run_epsilon_sweep.py — how does each method degrade as the privacy budget tightens?

Every CoRTeC number in this project is at eps_total = 2.0 (CORTEC_MEMORY.md §28.3). That is a
single point on the privacy-utility curve, and it is not the point the baseline papers report:
MST publishes at eps in {0.3, 1.0, 8.0} and AIM's "practical regime" is [0.1, 10]. A method that
looks good at eps=2 can be useless at eps=0.3, which is where a cautious regulator would sit.

For CoRTeC the budget affects ONLY Stage A — the released statistics. Generation is
post-processing, so one generation run per epsilon is all that is needed, and drawing more rows
from a given release costs no further privacy.

The sweep runs, per epsilon: DP release -> CoRTeC generation -> MST/PATE-CTGAN at the same
epsilon -> full evaluation. AIM is excluded by default because its fit did not complete within
two hours on these datasets (§28.1); pass it in --methods if there is time to spare.

The quantity to watch is NOT which method wins at each epsilon but how each one's error GROWS as
epsilon shrinks. CoRTeC's conditional table has Laplace noise 1/(n_cell * eps), so its
conditional fidelity should degrade gracefully while cells stay large; the marginal methods
degrade across all their measured marginals at once.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")


def run(cmd, log: Path):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.3, 1.0, 8.0])
    ap.add_argument("--n-synthetic", type=int, default=300)
    ap.add_argument("--budget-per-eps", type=float, default=2.60)
    ap.add_argument("--methods", nargs="+", default=["mst", "patectgan"])
    ap.add_argument("--rows-per-call", type=int, default=25)
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "openai", "gemini", "ollama", "mock"])
    ap.add_argument("--model", default=None,
                    help="generator model; passed through to run_dataset.py")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()

    root = Path(a.root or f"results/{a.dataset}_eps")
    root.mkdir(parents=True, exist_ok=True)
    log = root / "sweep.log"
    py = "./venv/bin/python"
    spent = 0.0

    for eps in a.epsilons:
        tag = f"eps{eps:g}".replace(".", "p")
        gdir = root / tag
        bdir = root / f"{tag}_baselines"
        gdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*78}\n=== {a.dataset} @ eps_total={eps}\n{'='*78}", flush=True)

        if not a.skip_generate:
            t0 = time.time()
            cmd = [py, "run_dataset.py", "--dataset", a.dataset, "--stage", "generate",
                   "--n-synthetic", str(a.n_synthetic), "--draws", str(a.draws),
                   "--rows-per-call", str(a.rows_per_call), "--skip-header-only",
                   "--epsilon-total", str(eps), "--budget-usd", f"{a.budget_per_eps:.2f}",
                   "--outdir", str(gdir), "--backend", a.backend]
            if a.model:
                cmd += ["--model", a.model]
            if a.effort:
                cmd += ["--effort", a.effort]
            run(cmd, log)
            man = gdir / "generation_manifest.json"
            if man.exists():
                m = json.load(open(man))
                spent += float(m.get("spend_usd", 0.0))
                print(f"  generated: {m.get('rows')} rows, {m.get('api_calls')} calls, "
                      f"${m.get('spend_usd', 0):.2f} in {time.time()-t0:.0f}s "
                      f"| cumulative ${spent:.2f}", flush=True)
                if m.get("aborted"):
                    print(f"  !! aborted: {m['aborted'][:160]}", flush=True)

        if not a.skip_baselines:
            t0 = time.time()
            run([py, "run_dataset.py", "--dataset", a.dataset, "--stage", "baselines",
                 "--n-out", str(a.n_synthetic), "--draws", "3",
                 "--epsilon-total", str(eps), "--methods", *a.methods,
                 "--outdir", str(bdir)], log)
            print(f"  baselines done in {time.time()-t0:.0f}s", flush=True)

    # ── one evaluation per epsilon ───────────────────────────────────────────────────
    print(f"\n{'='*78}\n=== EVALUATION\n{'='*78}", flush=True)
    for eps in a.epsilons:
        tag = f"eps{eps:g}".replace(".", "p")
        gdir, bdir = root / tag, root / f"{tag}_baselines"
        spec_file = root / f"spec_{tag}.json"
        cond = {"cortec": [str(gdir / "synthetic_cortec_draw*.csv")]}
        for m in a.methods:
            cond[m] = [str(bdir / f"{m}_seed*.csv")]
        json.dump(cond, open(spec_file, "w"), indent=2)
        run([py, "evaluate_generic.py", "--dataset", a.dataset,
             "--spec-json", str(spec_file), "--out", str(root / f"eval_{tag}.json")], log)

    print(f"\ntotal generation spend: ${spent:.2f}")
    print(f"logs: {log}")
    print(f"per-epsilon results: {root}/eval_eps*.json")


if __name__ == "__main__":
    main()
