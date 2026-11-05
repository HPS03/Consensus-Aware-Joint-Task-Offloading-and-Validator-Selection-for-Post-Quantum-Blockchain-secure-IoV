
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = [("completion", "Completion (s)", 1), ("throughput", "Capacity (kbit/s)", 1e-3),
           ("objective", "P2 objective", 1), ("energy", "Energy (J)", 1),
           ("deadline_miss", "Deadline miss (%)", 100), ("cons_ms", "Consensus (ms)", 1),
           ("hidden", "Hidden (%)", 100), ("self_val", "Self-validation (%)", 100),
           ("tampered", "Tampered caught", 1), ("tamper_committed", "Tampered committed", 1)]


def load(folder):
    path = os.path.join(folder, "results.json")
    with open(path) as fh:
        d = json.load(fh)
    c = d["cfg"]
    return c.get("seed"), d["fig3"]["test"], (c.get("n_veh"), c.get("n_res"), c.get("pbft_f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    folders = sorted({f for p in args.folders for f in (glob.glob(p) or [p])
                      if os.path.exists(os.path.join(f, "results.json"))})
    runs = [load(f) for f in folders]
    if not runs:
        raise SystemExit("no results.json found in the given folders")
    setups = {f: st for f, (_, _, st) in zip(folders, runs)}
    if len(set(setups.values())) > 1:
        for f, st in setups.items():
            print(f"  {f}: vehicles N={st[0]}, servers M={st[1]}, f={st[2]}")
        raise SystemExit("these runs use different settings (N, M or f); merge only runs with the same setup")
    runs = [(s, r) for s, r, _ in runs]
    seeds = [s for s, _ in runs]
    if len(set(seeds)) < len(seeds):
        raise SystemExit(f"the same seed appears twice: {seeds}")
    st = next(iter(setups.values()))
    print(f"{len(runs)} runs, seeds {seeds}, vehicles N={st[0]}, servers M={st[1]}")
    algs = [a for a in runs[0][1] if all(a in r for _, r in runs)]
    rows = []
    for a in algs:
        row = {"Method": a}
        for key, name, sc in METRICS:
            v = np.array([r[a].get(key, np.nan) for _, r in runs], float) * sc
            row[name] = f"{np.nanmean(v):.3f} +- {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.3f}"
        if a != "TaBRL" and "TaBRL" in algs and len(runs) > 1:
            x = np.array([r[a]["completion"] for _, r in runs])
            y = np.array([r["TaBRL"]["completion"] for _, r in runs])
            try:
                from scipy.stats import ttest_ind
                row["p vs TaBRL (completion)"] = round(float(ttest_ind(x, y, equal_var=False).pvalue), 4)
            except ImportError:
                pass
        rows.append(row)
    t = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    t.to_csv(os.path.join(args.out, "seeds_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
