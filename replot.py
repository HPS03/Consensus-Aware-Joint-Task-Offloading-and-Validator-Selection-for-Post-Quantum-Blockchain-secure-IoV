
import argparse
import glob
import json
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import cosched_gnn as C


def num(k):
    try:
        f = float(k)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return k


def numkeys(d):
    return {a: {num(k): v for k, v in pts.items()} for a, pts in d.items()}


def from_json(path):
    with open(path) as fh:
        d = json.load(fh)
    s = d.get("summary", {})
    out = dict(res=d.get("fig3", {}), hists=d.get("history", {}), cfg=d.get("cfg", {}))
    if "main" in s and s["main"].get("fig4"):
        out["fig4"] = {num(k): v for k, v in s["main"]["fig4"].items()}
    if "pqc" in s:
        out["pqc"] = s["pqc"]
    if "hidden" in s:
        out["hidden"] = numkeys(s["hidden"])
    if "scaling" in s:
        out["scaling"] = {k: numkeys(v) for k, v in s["scaling"].items()}
    if "ablation" in s:
        out["ablation"] = s["ablation"]
    return out


SCHEME_KEY = {v[0]: k for k, v in C.SIG.items()}   # display name -> scheme key


def from_csv(folder):
    """Tables transcribed from a run log (results_kaggle_seed42)."""
    def metrics(r):
        m = dict(completion=r["completion"])
        if "throughput_kbit" in r: m["throughput"] = r["throughput_kbit"] * 1e3
        for k in ("energy", "objective", "offload", "cons_ms", "tampered", "tamper_committed",
                  "forge_attempts", "forge_accepted"):
            if k in r: m[k] = r[k]
        if "sys_throughput" in r: m["sys_throughput"] = r["sys_throughput"] * 1e6
        for k in ("deadline_miss", "hidden", "self_val"):
            if f"{k}_pct" in r: m[k] = r[f"{k}_pct"] / 100
        if "val_load" in r: m["val_util"] = r["val_load"]
        return m
    out = {}
    mt = pd.read_csv(os.path.join(folder, "main_tables.csv"))
    out["res"] = {sp: {r.Algorithm: metrics(r._asdict()) for r in g.itertuples()} for sp, g in mt.groupby("Split")}
    p = os.path.join(folder, "pqc.csv")
    if os.path.exists(p):
        t = pd.read_csv(p); out["pqc"] = {}
        for r in t.itertuples():
            out["pqc"].setdefault(r.Algorithm, {})[SCHEME_KEY.get(r.Scheme, r.Scheme)] = metrics(r._asdict())
    p = os.path.join(folder, "hidden.csv")
    if os.path.exists(p):
        t = pd.read_csv(p); out["hidden"] = {}
        for r in t.itertuples():
            out["hidden"].setdefault(r.Algorithm, {})[num(r.Level)] = metrics(r._asdict())
    p = os.path.join(folder, "scaling.csv")
    if os.path.exists(p):
        t = pd.read_csv(p); sc = {}
        for r in t.itertuples():
            sc.setdefault(r.Sweep, {}).setdefault(r.Algorithm, {})[num(r.Size)] = metrics(r._asdict())
        out["scaling"] = sc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="figures_clean")
    ap.add_argument("--drop", default="CP-Greedy+Post-load")
    args = ap.parse_args()
    drop = {m.strip() for m in args.drop.split(",") if m.strip()}
    js = glob.glob(os.path.join(args.folder, "results.json"))
    d = from_json(js[0]) if js else from_csv(args.folder)
    os.makedirs(args.out, exist_ok=True)
    keep = lambda names: [a for a in C.CO_ORDER + C.CO_EXTRA + list(names) if a in names and a not in drop]
    order = lambda names: list(dict.fromkeys(keep(names)))
    o = lambda f: os.path.join(args.out, f)
    made = []

    res = {sp: {a: m for a, m in v.items() if a not in drop} for sp, v in d["res"].items()}
    algs = order(res.get("test", {}))
    if "train" in res and "test" in res:
        C.plot_co_fig3(res, algs, o("fig3_completion_capacity.png")); made.append("fig3_completion_capacity.png")
    C.plot_consensus(res, algs, o("consensus_security.png")); made.append("consensus_security.png")
    C.plot_paper_comparison(res, algs, o("paper_comparison.png")); made.append("paper_comparison.png")
    C.paper_table(res, algs).to_csv(o("paper_comparison.csv"), index=False)
    if d.get("fig4"):
        f4 = {nv: {a: m for a, m in v.items() if a not in drop} for nv, v in d["fig4"].items()}
        C.with_order(algs, C.plot_fig4, f4, o("fig4_backlog_energy.png")); made.append("fig4_backlog_energy.png")
    if d.get("hists"):
        h = {a: v for a, v in d["hists"].items() if a not in drop and a in algs}
        C.with_order(algs, C.plot_training, h, o("training_curves.png")); made.append("training_curves.png")
    if d.get("pqc"):
        pm = order(d["pqc"])
        schemes = [s for s in C.SIG if all(s in d["pqc"][a] for a in pm)]
        C.plot_pqc(d["pqc"], pm, schemes, o("pqc_signatures.png")); made.append("pqc_signatures.png")
    if d.get("hidden"):
        hd = {a: d["hidden"][a] for a in order(d["hidden"])}
        lv = sorted(next(iter(hd.values())))
        C.plot_lines(lv, hd, "Consensus-delay level (x RTT and result-check cycles)", o("hidden_vs_level.png"),
                     "Consensus hidden in DAG slack",
                     [("hidden", "Hidden consensus fraction (%)", 100), ("completion", "Mean completion time (s)", 1)])
        made.append("hidden_vs_level.png")
    sc = d.get("scaling", {})
    cfg = d.get("cfg", {})
    spec = {"vehicles": ("scale_vehicles.png", "Number of vehicles N",
                         f"Zero-shot: trained with N = {cfg.get('n_veh', 10)}",
                         [("completion", "Mean completion time (s)", 1), ("throughput", "Processing capacity (bit/s)", 1)]),
            "servers": ("scale_servers.png", "Number of edge servers M",
                        f"Zero-shot: trained with M = {cfg.get('n_res', 8)} (fixed-size FC agents cannot run at other M)",
                        [("completion", "Mean completion time (s)", 1), ("cons_ms", "Consensus delay (ms)", 1)]),
            "dag_size": ("scale_dag_size.png", "Largest DAG size (subtasks)",
                         f"Zero-shot: trained with DAGs of up to {cfg.get('k_max', 8)} subtasks",
                         [("completion", "Mean completion time (s)", 1), ("hidden", "Hidden consensus fraction", 1)])}
    for key, (fn, xl, title, keys) in spec.items():
        if key in sc:
            data = {a: sc[key][a] for a in order(sc[key])}
            keys = [k for k in keys if all(k[0] in m for pts in data.values() for m in pts.values())]
            if not data or not keys:
                continue
            xs = sorted({x for pts in data.values() for x in pts})
            C.plot_lines(xs, data, xl, o(fn), title, keys); made.append(fn)
    print(f"dropped {sorted(drop)}; wrote {len(made)} figures to {args.out}/:")
    for f in made:
        print("  " + f)


if __name__ == "__main__":
    main()
