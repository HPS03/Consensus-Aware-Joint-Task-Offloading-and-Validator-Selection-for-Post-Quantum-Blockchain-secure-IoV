
import os, re, glob, math, time, json, copy, random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

CFG = dict(
    # ---------------- data ----------------
    data_path=None,            # None -> search /kaggle/input recursively for batch_task*.csv
    max_rows=4_000_000,        # rows read from the trace (batch_task has ~14M rows)
    max_jobs=40_000,           # earliest valid DAG jobs kept
    allow_synthetic=False,     # True only for offline smoke tests (never for reported results)
    k_max=8, k_min=2,          # subtasks per DAG job
    train_frac=0.7,
    cyc_range=(0.2e9, 2.4e9),  # CPU cycles per subtask after scaling
    bits_range=(0.5e6, 6e6),   # input bits per subtask after scaling
    # ---------------- system ----------------
    n_veh=10, n_res=4, res_cores=2,
    res_freq=(2.0e9, 3.0e9, 4.0e9, 5.0e9),   # heterogeneous RES CPU (Hz per core)
    f_local_max=2.5e9, f_local_min=0.5e9, kappa=1e-27,
    slot=1.0, dt=0.1, horizon=300, drain_max=900,
    arrival_rate=0.30,                 # mean DAG jobs per vehicle per slot
    mfq_slices=(0.2, 0.5, 1.0),        # Low / Mid / High queue time slices (s)
    road_len=2000.0, rsu_offset=15.0, speed=(10.0, 25.0),
    bandwidth=2e6, p_tx=0.2, noise_dbm=-174.0, shadow_db=4.0, overhead=1.2,
    backhaul_bps=100e6, backhaul_hop_s=0.004,
    blk_interval=0.2, blk_capacity=24, blk_size=1e6, blk_rate=20e6,
    # compromised RES: which RES is compromised is re-drawn every episode, so a policy
    # cannot memorise it and has to detect it online (only TaBRL sees on-chain trust)
    n_malicious=1, p_tamper=0.35, report_factor=0.25,
    dispute_blocks=5,          # extra confirmation blocks before a failed proof is finalised
    energy_replenish=1.5, energy_v=1.0,
    # ---------------- Lyapunov reward ----------------
    V=1.0, alpha_q=0.05, alpha_e=0.03, beta_trust=0.3,
    trust_lambda=2.0, trust_min=0.8,   # TaBRL: trust-weighted RES selection / filtering
    # ---------------- MA-LHTO (He et al., IEEE TMC 2026) ----------------
    malhto_hidden=128, malhto_entropy=0.1, malhto_xi=0.01,   # Θ = 0.1, ξ = 0.01 as selected in the paper
    malhto_reset_every=1000, malhto_reset_std=0.01,          # δ_reset, σ: not given in the paper
    # ---------------- learning ----------------
    train_episodes=None,       # set on the command line: --epochs N
    batch=128, buffer=50_000, lr_actor=3e-4, lr_critic=1e-3,
    hidden=256, train_every=1, tau=0.01, n_candidates=32,
    eps_start=0.5, eps_end=0.03,
    # ---------------- evaluation ----------------
    eval_windows=3, fig4_horizon=150, fig4_veh=(5, 15),
    seed=42, out_dir="results",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# %% [markdown]
# ## 1. Alibaba Cluster Trace v2018 -> DAG jobs

# %%
TASK_COLS = ["task_name", "instance_num", "job_name", "task_type", "status",
             "start_time", "end_time", "plan_cpu", "plan_mem"]
INST_COLS = ["instance_name", "task_name", "job_name", "task_type", "status", "start_time",
             "end_time", "machine_id", "seq_no", "total_seq_no", "cpu_avg", "cpu_max",
             "mem_avg", "mem_max"]
NAME_RE = r"^[A-Za-z]+(\d+)((?:_\d+)*)$"


def _looks_like_trace(fp):
    """True if the file has the batch_task (9 columns) or batch_instance (14 columns) layout."""
    try:
        head = pd.read_csv(fp, header=None, nrows=3, low_memory=False)
    except Exception:
        return False
    first = [str(v).strip().lower() for v in head.iloc[0].values]
    return head.shape[1] in (9, 14) or "task_name" in first


def find_trace_file(path=None):
    if path and os.path.isfile(path):
        return path
    roots = [path] if path else ["/kaggle/input", ".", "./data"]
    files = []
    for r in roots:
        if r and os.path.isdir(r):
            for ext in ("*.csv", "*.csv.gz", "*.tar.gz"):
                files += glob.glob(os.path.join(r, "**", ext), recursive=True)
    for key in ("batch_task", "batch_instance"):
        hit = [f for f in files if key in os.path.basename(f).lower()]
        if hit:
            return sorted(hit, key=os.path.getsize)[-1]
    # dataset uploaded under another file name: only look in the input folder (never in our own
    # output folders) and only accept a file with the trace's column layout
    inputs = [f for f in files if f.startswith(path or "/kaggle/input") and f.endswith(".csv")]
    inputs = [f for f in sorted(inputs, key=os.path.getsize, reverse=True) if _looks_like_trace(f)]
    return inputs[0] if inputs else None


def _read_trace(fp, max_rows):
    head = pd.read_csv(fp, header=None, nrows=3, low_memory=False)
    has_header = any(str(v).strip().lower() in ("task_name", "job_name", "instance_name")
                     for v in head.iloc[0].values)
    ncol = head.shape[1]
    if has_header:
        df = pd.read_csv(fp, nrows=max_rows, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        kind = "instance" if "instance_name" in df.columns else "task"
    else:
        kind = "instance" if ncol == 14 else "task"
        cols = INST_COLS if kind == "instance" else TASK_COLS
        if ncol != len(cols):
            raise ValueError(f"{fp}: unexpected column count {ncol}")
        df = pd.read_csv(fp, header=None, names=cols, nrows=max_rows, low_memory=False)
    if kind == "instance":   # aggregate instances into tasks
        df = (df.groupby(["job_name", "task_name"], as_index=False)
                .agg(instance_num=("start_time", "size"), task_type=("task_type", "first"),
                     status=("status", "first"), start_time=("start_time", "min"),
                     end_time=("end_time", "max"), plan_cpu=("cpu_avg", "mean"),
                     plan_mem=("mem_avg", "mean")))
    return df, kind


def _log_scale(v, lo, hi):
    lv = np.log1p(v)
    a, b = np.nanpercentile(lv, 1), np.nanpercentile(lv, 99)
    z = np.clip((lv - a) / max(b - a, 1e-9), 0, 1)
    return lo + z * (hi - lo)


def load_dag_jobs(cfg):
    fp = find_trace_file(cfg["data_path"])
    if fp is None:
        if not cfg["allow_synthetic"]:
            raise FileNotFoundError(
                "Alibaba trace not found. Add the Kaggle dataset to the notebook "
                "(it appears under /kaggle/input) or set CFG['data_path'] to batch_task.csv.")
        print("!! No trace found - using a SYNTHETIC trace for a smoke test only.")
        df, kind = synthetic_batch_task(cfg["max_jobs"], cfg["seed"], cfg["k_max"]), "synthetic"
    else:
        print(f"Loading trace: {fp}")
        df, kind = _read_trace(fp, cfg["max_rows"])
    print(f"  {len(df):,} task rows ({kind})")

    df = df.dropna(subset=["task_name", "job_name", "start_time", "end_time"])
    if "status" in df.columns and (df["status"] == "Terminated").any():
        df = df[df["status"] == "Terminated"]
    df = df[(df["start_time"] > 0) & (df["end_time"] >= df["start_time"])].copy()
    ext = df["task_name"].astype(str).str.extract(NAME_RE)
    df["tid"] = pd.to_numeric(ext[0], errors="coerce")
    df["deps"] = ext[1].fillna("")
    ok = df["tid"].notna().groupby(df["job_name"]).transform("all")
    df = df[ok]
    size = df.groupby("job_name")["tid"].transform("size")
    df = df[(size >= cfg["k_min"]) & (size <= cfg["k_max"])].copy()
    # keep only the earliest jobs (with margin for ones rejected below) before building DAGs
    first = df.groupby("job_name")["start_time"].min().nsmallest(int(cfg["max_jobs"] * 1.2))
    df = df[df["job_name"].isin(first.index)].copy()

    inst = pd.to_numeric(df["instance_num"], errors="coerce").fillna(1).clip(lower=1)
    cpu = pd.to_numeric(df["plan_cpu"], errors="coerce").fillna(100).clip(lower=1)
    mem = pd.to_numeric(df["plan_mem"], errors="coerce").fillna(0.5).clip(lower=0.01)
    dur = (df["end_time"] - df["start_time"]).clip(lower=1)
    df["cycles"] = _log_scale((inst * cpu / 100.0 * dur).values, *cfg["cyc_range"])
    df["bits"] = _log_scale((inst * mem).values, *cfg["bits_range"])

    jobs = []
    for jn, g in df.groupby("job_name", sort=False):
        ids = g["tid"].astype(int).tolist()
        if len(set(ids)) != len(ids):
            continue
        idx = {t: i for i, t in enumerate(ids)}
        preds = [[idx[int(d)] for d in dep.split("_") if d and int(d) in idx and int(d) != t]
                 for t, dep in zip(ids, g["deps"])]
        order = topo_order(preds)
        if order is None:
            continue
        pos = {o: i for i, o in enumerate(order)}   # re-index in topological order
        jobs.append(dict(
            start=float(g["start_time"].min()),
            b=[float(g["cycles"].iloc[o]) for o in order],
            d=[float(g["bits"].iloc[o]) for o in order],
            preds=[sorted(pos[p] for p in preds[o]) for o in order]))
    jobs.sort(key=lambda j: j["start"])
    jobs = jobs[: cfg["max_jobs"]]
    n_tr = int(len(jobs) * cfg["train_frac"])
    ks = [len(j["b"]) for j in jobs]
    print(f"  {len(jobs):,} DAG jobs ({np.mean(ks):.2f} subtasks avg), train {n_tr:,} / test {len(jobs) - n_tr:,}")
    return jobs[:n_tr], jobs[n_tr:]


def topo_order(preds):
    n = len(preds)
    indeg = [len(p) for p in preds]
    succ = [[] for _ in range(n)]
    for i, ps in enumerate(preds):
        for p in ps:
            succ[p].append(i)
    q = deque(i for i in range(n) if indeg[i] == 0)
    out = []
    while q:
        u = q.popleft(); out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return out if len(out) == n else None


def synthetic_batch_task(n_jobs, seed, k_hi=8):
    """Trace-shaped stand-in used ONLY when allow_synthetic=True (offline smoke test)."""
    rng = np.random.default_rng(seed)
    rows, t = [], 86400
    for j in range(n_jobs):
        t += int(rng.exponential(3.0))
        k = int(rng.integers(2, 9)) if k_hi <= 8 else int(min(k_hi, 2 + rng.geometric(1 / 6)))
        for i in range(1, k + 1):
            deps = sorted(rng.choice(np.arange(1, i), size=min(i - 1, int(rng.integers(0, 3))),
                                     replace=False)) if i > 1 else []
            name = rng.choice(list("MRJ")) + str(i) + "".join(f"_{d}" for d in deps)
            dur = int(rng.lognormal(3.5, 1.2)) + 1
            rows.append([name, int(rng.lognormal(2.0, 1.5)) + 1, f"j_{j}", 1, "Terminated",
                         t, t + dur, float(rng.choice([50, 100, 200])), float(rng.uniform(0.1, 1.5))])
    return pd.DataFrame(rows, columns=TASK_COLS)


def make_window(jobs, n_veh, horizon, cfg, rng):
    """Discretise trace arrivals into slots and give each job to a free vehicle.
    Trace time is compressed so the mean load is arrival_rate jobs / vehicle / slot,
    while the burstiness of the trace is kept."""
    need = int(horizon * n_veh * cfg["arrival_rate"] * 1.3) + 10
    s0 = int(rng.integers(0, max(1, len(jobs) - need)))
    seg = jobs[s0: s0 + need]
    starts = np.array([j["start"] for j in seg])
    gap = max((starts[-1] - starts[0]) / max(len(seg) - 1, 1), 1e-6)
    trace_slot = gap * n_veh * cfg["arrival_rate"]
    slots = ((starts - starts[0]) / trace_slot).astype(int)
    sched = [[] for _ in range(horizon)]
    carry = deque()
    ji = 0
    for s in range(horizon):
        while ji < len(seg) and slots[ji] <= s:
            carry.append(seg[ji]); ji += 1
        free = list(rng.permutation(n_veh))
        while carry and free:
            sched[s].append((int(free.pop()), carry.popleft()))
    return sched


# %% [markdown]
# ## 2. Blockchain-assisted IoV environment

# %%
class Blockchain:
    """Consortium chain: a transaction waits for the next block with free space,
    then pays D_blk / R_blk to propagate and reach consensus."""
    def __init__(self, cfg):
        self.tau, self.cap = cfg["blk_interval"], cfg["blk_capacity"]
        self.prop = cfg["blk_size"] / cfg["blk_rate"]
        self.used = {}

    def confirm(self, t):
        b = int(math.ceil(t / self.tau + 1e-9))
        while self.used.get(b, 0) >= self.cap:
            b += 1
        self.used[b] = self.used.get(b, 0) + 1
        return b * self.tau + self.prop - t

    def est_delay(self, t):
        b = int(math.ceil(t / self.tau + 1e-9))
        n = 0
        while self.used.get(b + n, 0) >= self.cap:
            n += 1
        return (b + n) * self.tau + self.prop - t


class Sub:
    __slots__ = ("task", "k", "b", "d", "preds", "x", "res", "rem", "level", "done_at",
                 "avail_at", "f")

    def __init__(self, task, k, b, d):
        self.task, self.k, self.b, self.d = task, k, b, d
        self.preds, self.x, self.res, self.rem, self.level = [], 0, -1, b, 0
        self.done_at, self.avail_at, self.f = None, 0.0, 1e9

    def ready(self, t):
        return all(p.done_at is not None and p.done_at <= t + 1e-9 for p in self.preds)


class Task:
    def __init__(self, vid, arr, job):
        self.vid, self.arr = vid, arr
        self.subs = [Sub(self, k, b, d) for k, (b, d) in enumerate(zip(job["b"], job["d"]))]
        for s, ps in zip(self.subs, job["preds"]):
            s.preds = [self.subs[p] for p in ps]
        self.size = sum(job["d"])
        self.n_done, self.complete, self.energy = 0, None, 0.0
        self.obs = self.act = None
        self.info = {}


class RES:
    def __init__(self, j, cfg, malicious):
        self.j, self.cfg, self.mal = j, cfg, malicious
        fr = cfg["res_freq"]
        self.F = float(fr[j % len(fr)]) if isinstance(fr, (list, tuple)) else float(fr)
        self.slices = cfg["mfq_slices"]
        self.incoming, self.pending = [], []
        self.q = [deque(), deque(), deque()]
        self.cores = [[None, 0.0] for _ in range(cfg["res_cores"])]   # [job, slice left]
        self.ok = self.fail = 0

    def backlog_cycles(self):
        c = sum(s.rem for s in self.incoming) + sum(s.rem for s in self.pending)
        c += sum(s.rem for q in self.q for s in q)
        return c + sum(core[0].rem for core in self.cores if core[0] is not None)

    def n_jobs(self):
        return (len(self.incoming) + len(self.pending) + sum(len(q) for q in self.q)
                + sum(core[0] is not None for core in self.cores))

    def trust(self):
        # reputation with an optimistic prior: 1.0 until verification failures are recorded
        return (self.ok + 2.0) / (self.ok + self.fail + 2.0)

    def step(self, t, dt, env):
        if self.incoming:
            keep = []
            for s in self.incoming:
                (self.pending if s.avail_at <= t else keep).append(s)
            self.incoming = keep
        if self.pending:
            keep = []
            for s in self.pending:
                if s.ready(t):
                    s.level = 0; self.q[0].append(s)
                else:
                    keep.append(s)
            self.pending = keep
        for core in self.cores:
            budget = dt
            while budget > 1e-9:
                if core[0] is None:
                    lvl = next((l for l in range(3) if self.q[l]), None)
                    if lvl is None:
                        break
                    core[0], core[1] = self.q[lvl].popleft(), self.slices[lvl]
                s = core[0]
                run = min(budget, s.rem / self.F, core[1])
                s.rem -= run * self.F; budget -= run; core[1] -= run
                if s.rem <= 1.0:
                    core[0] = None
                    env.res_finished(self, s, t + dt - budget)
                elif core[1] <= 1e-9:          # time slice used up -> demote (MFQ)
                    s.level = min(s.level + 1, 2)
                    self.q[s.level].append(s); core[0] = None


class Vehicle:
    def __init__(self, vid, cfg, rng):
        self.vid, self.cfg = vid, cfg
        self.x = float(rng.uniform(0, cfg["road_len"]))
        self.y = float(rng.uniform(0, 10))
        self.v = float(rng.uniform(*cfg["speed"])) * (1 if rng.random() < 0.5 else -1)
        self.local = deque()
        self.tx_free, self.tx_bits = 0.0, []    # (start, end, bits) of uplink transfers
        self.Q = self.E = 0.0
        self.slot_energy = self.total_energy = 0.0

    def step(self, t, dt, env):
        budget, now = dt, t
        while budget > 1e-9 and self.local:
            s = self.local[0]
            if s.avail_at > now + 1e-9 or not s.ready(now):
                break
            run = min(budget, s.rem / s.f)
            s.rem -= run * s.f; budget -= run; now += run
            e = self.cfg["kappa"] * s.f ** 3 * run
            self.slot_energy += e; s.task.energy += e
            if s.rem <= 1.0:
                self.local.popleft()
                env.sub_done(s, now)

    def queued_bits(self, t):
        q = sum(s.d * s.rem / s.b for s in self.local)
        for a, b, bits in self.tx_bits:
            if b > t:
                q += bits * min(1.0, (b - t) / max(b - a, 1e-9))
        return q


class IoVEnv:
    def __init__(self, cfg, n_veh, seed):
        self.cfg, self.N, self.M = cfg, n_veh, cfg["n_res"]
        self.K = cfg["k_max"]
        self.rng = np.random.default_rng(seed)
        mal = set(self.rng.choice(self.M, cfg["n_malicious"], replace=False).tolist())
        self.res = [RES(j, cfg, j in mal) for j in range(self.M)]
        self.veh = [Vehicle(i, cfg, self.rng) for i in range(n_veh)]
        self.rsu = np.array([[cfg["road_len"] * (j + 0.5) / self.M, cfg["rsu_offset"]] for j in range(self.M)])
        self.chain = Blockchain(cfg)
        self.t, self.tasks, self.completed = 0.0, [], []
        self.shadow = self.rng.normal(0, cfg["shadow_db"], n_veh)

    # ---- channel ----
    def nearest(self, v):
        d = np.hypot(self.rsu[:, 0] - v.x, self.rsu[:, 1] - v.y)
        j = int(np.argmin(d)); return j, float(d[j])

    def rate(self, v):
        c = self.cfg
        _, d = self.nearest(v)
        pl = 128.1 + 37.6 * math.log10(max(d, 10.0) / 1000.0) + self.shadow[v.vid]
        noise = 10 ** ((c["noise_dbm"] - 30) / 10) * c["bandwidth"]
        snr = c["p_tx"] * 10 ** (-pl / 10) / noise
        return c["bandwidth"] * math.log2(1 + snr)

    # ---- observation ----
    def obs_dim(self):
        return 6 * self.K + 6 + 5 * self.M + 2

    def _obs_parts(self, vid, task, mode):
        """Per-subtask features, DAG depth, vehicle, per-RES and global features."""
        v, c = self.veh[vid], self.cfg
        depth, tasks = [], []
        for k, s in enumerate(task.subs):
            dp = 1 + max((depth[p.k] for p in s.preds), default=-1)
            depth.append(dp)
            n_succ = sum(s in o.preds for o in task.subs)
            tasks.append((s.b / 1e9, s.d / 1e6, dp, len(s.preds), n_succ))
        near, _ = self.nearest(v)
        vf = [v.Q / 1e7, v.E / 10.0, sum(s.rem for s in v.local) / c["f_local_max"] / 10.0,
              max(v.tx_free - self.t, 0) / 5.0, self.rate(v) / 2e7, near / self.M]
        rf = []
        for j, r in enumerate(self.res):
            load = r.backlog_cycles() / (r.F * len(r.cores))
            if mode != "audited" and r.mal:     # self-reported load of a compromised RES
                load *= c["report_factor"]
            tr = r.trust() if mode == "audited" else 0.5
            rf.append((load / 10.0, r.n_jobs() / 20.0, tr, abs(near - j), r.F / 5e9))
        gf = [self.chain.est_delay(self.t), sum(x.Q for x in self.veh) / 1e8]
        return tasks, vf, rf, gf

    def build_obs(self, vid, task, mode):
        """Flat, fixed-size observation for the FC agents (padded to k_max subtasks)."""
        K = self.K
        tasks, vf, rf, gf = self._obs_parts(vid, task, mode)
        tf = np.zeros((K, 6), np.float32)
        for k, (b, d, dp, npred, nsucc) in enumerate(tasks):
            tf[k] = [b, d, dp / K, npred / K, nsucc / K, 1.0]
        rflat = [x for (l, n, tr, hop, f) in rf for x in (l, n, tr, hop / self.M, f)]
        return np.concatenate([tf.ravel(), np.array(vf + rflat + gf, np.float32)]).astype(np.float32)

    def build_graph_obs(self, vid, task, mode):
        """Graph observation for the GNN agent: any number of subtasks and RESs.
        xt: subtask nodes, A[i, j] = 1 if subtask j is a predecessor of i, xs: RES nodes."""
        tasks, vf, rf, gf = self._obs_parts(vid, task, mode)
        k = len(task.subs)
        xt = np.array([(b, d, dp / 8.0, npred / 4.0, nsucc / 4.0) for b, d, dp, npred, nsucc in tasks],
                      np.float32)
        A = np.zeros((k, k), np.float32)
        for s in task.subs:
            for p in s.preds:
                A[s.k, p.k] = 1.0
        xs = np.array([(l, n, tr, hop / 4.0, f) for l, n, tr, hop, f in rf], np.float32)
        vf = list(vf); vf[-1] = vf[-1] * self.M / 4.0     # nearest-RSU index, independent of M
        return dict(xt=xt, A=A, xs=xs, xv=np.array(vf, np.float32), xg=np.array(gf, np.float32))

    # ---- actions ----
    def submit(self, task, x, y, f):
        c, v = self.cfg, self.veh[task.vid]
        f = float(np.clip(f, c["f_local_min"], c["f_local_max"]))
        near, _ = self.nearest(v)
        rate = self.rate(v)
        task.info.update(Q=v.Q, E=v.E, trust=[r.trust() for r in self.res], x=list(x), y=list(y))
        for k, s in enumerate(task.subs):
            s.x, s.f = int(x[k]), f
            if s.x:
                j = int(y[k]); s.res = j
                st = max(self.t, v.tx_free)
                dur = c["overhead"] * s.d / rate
                v.tx_free = st + dur
                v.tx_bits.append((st, st + dur, s.d))
                e = c["p_tx"] * dur
                v.slot_energy += e; task.energy += e
                hop = abs(near - j) * (c["backhaul_hop_s"] + s.d / c["backhaul_bps"])
                arrive = st + dur + hop
                s.avail_at = arrive + self.chain.confirm(arrive)     # on-chain registration
                self.res[j].incoming.append(s)
            else:
                v.local.append(s)
        self.tasks.append(task)

    def res_finished(self, r, s, t):
        blk = self.chain.confirm(t)                   # proof-of-computation
        if r.mal and self.rng.random() < self.cfg["p_tamper"]:
            r.fail += 1                               # verification fails -> dispute, then recompute locally
            v = self.veh[s.task.vid]
            detect = t + blk + self.cfg["dispute_blocks"] * self.cfg["blk_interval"]
            s.x, s.rem, s.avail_at, s.f = 0, s.b, detect, self.cfg["f_local_max"]
            v.local.appendleft(s)
        else:
            r.ok += 1
            self.sub_done(s, t + blk)

    def sub_done(self, s, t):
        s.done_at = t
        task = s.task
        task.n_done += 1
        if task.n_done == len(task.subs):
            task.complete = max(x.done_at for x in task.subs)
            self.completed.append(task)

    # ---- time ----
    def begin_slot(self):
        c = self.cfg
        for v in self.veh:
            v.x = (v.x + v.v * c["slot"]) % c["road_len"]
        self.shadow = 0.7 * self.shadow + 0.3 * self.rng.normal(0, c["shadow_db"], self.N)

    def run_slot(self):
        c = self.cfg
        n = int(round(c["slot"] / c["dt"]))
        for i in range(n):
            t = self.t + i * c["dt"]
            for r in self.res:
                r.step(t, c["dt"], self)
            for v in self.veh:
                v.step(t, c["dt"], self)
        self.t += c["slot"]
        for v in self.veh:
            v.tx_bits = [z for z in v.tx_bits if z[1] > self.t]
            v.Q = v.queued_bits(self.t)
            v.E = max(v.E + c["energy_v"] * v.slot_energy - c["energy_v"] * c["energy_replenish"], 0.0)
            v.total_energy += v.slot_energy; v.slot_energy = 0.0
        done, self.completed = self.completed, []
        return done

    def all_done(self):
        return all(t.complete is not None for t in self.tasks)


# %% [markdown]
# ## 3. Rewards (Lyapunov drift-plus-penalty and baseline objectives)

# %%
def task_delay(task):
    return max(task.complete - task.arr, 1e-3)


def lyapunov_reward(task, cfg, trust_aware):
    """Per-slot problem P2: V * throughput - Q_i * (bits kept local) - E_i * energy."""
    T = task_delay(task)
    local_bits = sum(s.d for s, x in zip(task.subs, task.info["x"]) if not x) / 1e6
    r = cfg["V"] * (task.size / 1e6) / T
    r -= cfg["alpha_q"] * (task.info["Q"] / 1e7) * local_bits
    r -= cfg["alpha_e"] * (task.info["E"] / 10.0) * task.energy
    if trust_aware:   # penalise sending work to RESs the chain rates as unreliable
        tr = task.info["trust"]
        r -= cfg["beta_trust"] * sum((1 - tr[y]) for x, y in zip(task.info["x"], task.info["y"]) if x)
    return float(np.clip(r, -20, 20))


def cost_reward(task, w_t=1.0, w_e=0.1):
    return float(np.clip(-(w_t * task_delay(task) + w_e * task.energy), -40, 0))


# %% [markdown]
# ## 4. Agents

# %%
def mlp(i, o, h, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def soft_update(tgt, src, tau):
    with torch.no_grad():
        for a, b in zip(tgt.parameters(), src.parameters()):
            a.mul_(1 - tau).add_(tau * b)


class Replay:
    def __init__(self, cap, od, ad):
        self.o = np.zeros((cap, od), np.float32); self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros(cap, np.float32); self.n = self.p = 0; self.cap = cap

    def add(self, o, a, r):
        self.o[self.p], self.a[self.p], self.r[self.p] = o, a, r
        self.p = (self.p + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        i = np.random.randint(0, self.n, b)
        f = lambda z: torch.as_tensor(z[i], device=DEVICE)
        return f(self.o), f(self.a), f(self.r)


class BaseAgent:
    name, obs_mode = "base", "reported"

    def __init__(self, env, cfg):
        self.K, self.M, self.cfg = env.K, env.M, cfg
        self.od = env.obs_dim()
        self.eps, self.steps = cfg["eps_start"], 0

    def set_progress(self, frac):
        c = self.cfg
        self.eps = c["eps_end"] + (c["eps_start"] - c["eps_end"]) * max(0.0, 1 - frac / 0.7)

    def f_of(self, u):
        c = self.cfg
        return c["f_local_min"] + float(np.clip(u, 0, 1)) * (c["f_local_max"] - c["f_local_min"])

    def reward(self, task):
        raise NotImplementedError

    def train(self):
        pass


class DAGLocal(BaseAgent):
    """All subtasks executed on the vehicle at full CPU frequency (no learning)."""
    name = "DAG-Local"

    def act(self, obs, task, explore):
        k = len(task.subs)
        return [0] * k, [0] * k, self.cfg["f_local_max"], None

    def store(self, task, r):
        pass

    def reward(self, task):
        return 0.0


class TaBRL(BaseAgent):
    """Dual-FC Actor (offloading {x,y} + computation {f}) with target Actors,
    noise-preserving quantisation (NOP) candidates, Critic that scores candidates on
    blockchain-audited state, Lyapunov + trust reward, replay buffer and soft updates."""
    name, obs_mode = "TaBRL", "audited"

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        K, M, h = self.K, self.M, cfg["hidden"]
        self.ad = K + K * M + 1
        self.actor_off = mlp(self.od, K + K * M, h).to(DEVICE)
        self.actor_f = mlp(self.od, 1, h // 2).to(DEVICE)
        self.t_off, self.t_f = copy.deepcopy(self.actor_off), copy.deepcopy(self.actor_f)
        self.critic = mlp(self.od + self.ad, 1, h, 3).to(DEVICE)
        self.opt_a = torch.optim.Adam(list(self.actor_off.parameters()) + list(self.actor_f.parameters()),
                                      lr=cfg["lr_actor"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])
        self.buf = Replay(cfg["buffer"], self.od, self.ad)
        self.rng = np.random.default_rng(cfg["seed"] + 1)

    def encode(self, x, yprob, u):
        # x: (B,K) offload prob, yprob: (B,K,M) RES distribution, u: (B,1) freq in [0,1]
        return torch.cat([x, (x.unsqueeze(-1) * yprob).flatten(1), u], 1)

    def _forward(self, off_net, f_net, o):
        z = off_net(o)
        p = torch.sigmoid(z[:, : self.K])
        logit = z[:, self.K:].view(-1, self.K, self.M)
        u = torch.sigmoid(f_net(o))
        return p, logit, u

    def res_view(self, obs):
        """Blockchain-audited RES state from the observation: load (s), trust, CPU (Hz)."""
        r = obs[6 * self.K + 6: 6 * self.K + 6 + 5 * self.M].reshape(self.M, 5)
        return r[:, 0] * 10.0, np.clip(r[:, 2], 1e-3, 1.0), r[:, 4] * 5e9

    def candidates(self, p, logit, u, tp, tlogit, tu, k, load, trust, freq, bvec):
        rng, cands, M = self.rng, [], logit.shape[-1]
        # trust-weighted policy: pi'(j) ~ pi(j) * trust_j^lambda (on-chain verification record)
        bias = self.cfg["trust_lambda"] * np.log(trust)
        def add(pp, ll, uu, b=bias):
            x = (pp[:k] > 0.5).astype(np.float32)
            y = (ll[:k] + b).argmax(-1)
            cands.append((x, y, float(np.clip(uu, 0, 1))))
        add(p, logit, u); add(tp, tlogit, tu)
        add(p, logit, u, 0.0)                              # raw Actor choice, no trust bias
        # trust-filtered candidates: work on RESs below the trust threshold is moved to the
        # best trusted RES (trust * speed / backlog), or kept on the vehicle
        x0 = (p[:k] > 0.5).astype(np.float32); y0 = (logit[:k] + bias).argmax(-1)
        bad = trust[y0] < self.cfg["trust_min"]
        if bad.any():
            # spread the moved subtasks over trusted RESs in order of trust * speed / backlog
            ok = [j for j in np.argsort(-(trust * freq / (1.0 + load))) if trust[j] >= self.cfg["trust_min"]]
            ok = ok or [int(np.argmax(trust))]
            y1 = y0.copy(); y1[bad] = [ok[i % len(ok)] for i in range(int(bad.sum()))]
            cands.append((x0, y1, float(np.clip(u, 0, 1))))
            x2 = x0.copy(); x2[bad] = 0
            cands.append((x2, y0, float(np.clip(u, 0, 1))))
        order = np.argsort(np.abs(p[:k] - 0.5))           # order-preserving flips
        for m in range(1, min(k, 3) + 1):
            pp = p.copy(); pp[order[:m]] = 1 - pp[order[:m]]; add(pp, logit, u)
        # structured quantisation: consolidated placements on trusted RESs
        # (whole DAG to one RES, heavy subtasks to one RES, all local)
        heavy = (bvec[:k] >= np.median(bvec[:k])).astype(np.float32)
        uu = float(np.clip(u, 0, 1))
        for j in range(M):
            if trust[j] >= self.cfg["trust_min"]:
                cands.append((np.ones(k, np.float32), np.full(k, j), uu))
                cands.append((heavy, np.full(k, j), uu))
        cands.append((np.zeros(k, np.float32), np.zeros(k, int), 1.0))
        i = 0
        while len(cands) < self.cfg["n_candidates"]:      # Gaussian + uniform perturbation
            pp = np.clip(p + rng.normal(0, 0.25, p.shape), 0, 1)
            ll = logit + rng.normal(0, 1.0, logit.shape)
            if rng.random() < 0.3:
                ll[:, rng.integers(M)] += rng.uniform(0, 3)
            add(pp, ll, u + rng.uniform(-0.25, 0.25), bias if i % 2 == 0 else 0.0)
            i += 1
        return cands

    def enc_np(self, x, y, u):
        K, M = self.K, self.M
        a = np.zeros(self.ad, np.float32)
        a[: len(x)] = x
        oh = np.zeros((K, M), np.float32)
        oh[np.arange(len(y)), y] = 1
        oh[: len(x)] *= x[:, None]; oh[len(x):] = 0
        a[K: K + K * M] = oh.ravel(); a[-1] = u
        return a

    @torch.no_grad()
    def act(self, obs, task, explore):
        k = len(task.subs)
        o = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
        p, l, u = [z[0].cpu().numpy() for z in self._forward(self.actor_off, self.actor_f, o)]
        tp, tl, tu = [z[0].cpu().numpy() for z in self._forward(self.t_off, self.t_f, o)]
        load, trust, freq = self.res_view(obs)
        cands = self.candidates(p, l, float(u[0]), tp, tl, float(tu[0]), k, load, trust, freq,
                                obs[0: 6 * self.K: 6])
        if explore and self.rng.random() < self.eps:
            x = self.rng.integers(0, 2, k).astype(np.float32)
            y = self.rng.integers(0, self.M, k)
            best = (x, y, float(self.rng.random()))
        else:
            enc = torch.as_tensor(np.stack([self.enc_np(*c) for c in cands]), device=DEVICE)
            q = self.critic(torch.cat([o.expand(len(cands), -1), enc], 1)).squeeze(1)
            best = cands[int(q.argmax())]
        x, y, uu = best
        return x.astype(int).tolist(), y.tolist(), self.f_of(uu), self.enc_np(x, y, uu)

    def reward(self, task):
        return lyapunov_reward(task, self.cfg, trust_aware=True)

    def store(self, task, r):
        self.buf.add(task.obs, task.act, r)

    def train(self):
        if self.buf.n < self.cfg["batch"]:
            return
        K, M = self.K, self.M
        o, a, r = self.buf.sample(self.cfg["batch"])
        q = self.critic(torch.cat([o, a], 1)).squeeze(1)
        lc = F.mse_loss(q, r)                                   # TD loss (per-slot, gamma = 0)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()

        p, logit, u = self._forward(self.actor_off, self.actor_f, o)
        valid = o[:, 5: 6 * K: 6]                               # subtask-valid flags
        relaxed = self.encode(p * valid, torch.softmax(logit, -1), u)
        q_pi = self.critic(torch.cat([o, relaxed], 1)).squeeze(1)
        with torch.no_grad():                                   # advantage-weighted imitation
            w = torch.exp((r - q_pi) / 1.0).clamp(max=10.0); w = w / w.mean()
        x_t = a[:, :K]
        y_t = a[:, K: K + K * M].view(-1, K, M)
        bce = (F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), x_t, reduction="none") * valid).sum(1)
        ce = (-(torch.log_softmax(logit, -1) * y_t).sum(-1)).sum(1)
        mse = (u.squeeze(1) - a[:, -1]) ** 2
        la = (w * (bce + ce + mse)).mean() - 0.1 * q_pi.mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        soft_update(self.t_off, self.actor_off, self.cfg["tau"])
        soft_update(self.t_f, self.actor_f, self.cfg["tau"])


class DQN:
    def __init__(self, od, na, cfg, h):
        self.q = mlp(od, na, h).to(DEVICE)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr_critic"])
        self.buf = Replay(cfg["buffer"], od, 1)
        self.na, self.bs = na, cfg["batch"]

    @torch.no_grad()
    def greedy(self, obs):
        return int(self.q(torch.as_tensor(obs, device=DEVICE).unsqueeze(0)).argmax())

    def train(self):
        if self.buf.n < self.bs:
            return
        o, a, r = self.buf.sample(self.bs)
        q = self.q(o).gather(1, a.long()).squeeze(1)
        loss = F.smooth_l1_loss(q, r)
        self.opt.zero_grad(); loss.backward(); self.opt.step()


def pattern_to_xy(pat, task, M):
    """0: all local | 1..M: all to RES j | M+1..2M: heavy subtasks (>= median cycles) to RES j."""
    k = len(task.subs)
    if pat == 0:
        return [0] * k, [0] * k
    j = (pat - 1) % M
    if pat <= M:
        return [1] * k, [j] * k
    med = np.median([s.b for s in task.subs])
    return [int(s.b >= med) for s in task.subs], [j] * k


class _MALHTOActor(nn.Module):
    """FC actor with two hidden layers and two output heads (offloading target, CPU frequency)."""
    def __init__(self, n_in, n_opt, h):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head_x = nn.Linear(h, n_opt)
        self.head_f = nn.Linear(h, 1)

    def forward(self, s):
        z = self.body(s)
        return self.head_x(z), torch.sigmoid(self.head_f(z))


class _MALHTODevice:
    """One device agent: primary + target actor, optimiser and its own replay buffer."""
    def __init__(self, n_in, n_opt, cfg):
        self.net = _MALHTOActor(n_in, n_opt, cfg["malhto_hidden"]).to(DEVICE)
        self.target = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr_critic"])
        cap = cfg["buffer"]
        self.s = np.zeros((cap, n_in), np.float32); self.a = np.zeros(cap, np.int64)
        self.u = np.zeros(cap, np.float32); self.n = self.p = 0; self.cap = cap; self.steps = 0

    def add(self, s, a, u):
        self.s[self.p], self.a[self.p], self.u[self.p] = s, a, u
        self.p = (self.p + 1) % self.cap; self.n = min(self.n + 1, self.cap)


class MALHTO(BaseAgent):
    """MA-LHTO as described in X. He et al., "Real-Time Scheduling of CPU/GPU Heterogeneous Tasks
    in Dynamic IoT Systems", IEEE TMC 25(1), 2026, and its released code
    (github.com/HE-xiao4333/MA-LHTO), mapped onto this vehicular environment.

    * One agent per vehicle (device); FC actor with two hidden layers and two output heads:
      offloading target in {local, RES 1..M} and CPU frequency. The GPU-node head and the
      base-station resource-allocation agents are dropped: this system has no GPU types and
      RESs share their CPU through the MFQ, so there is no per-task allocation to learn.
    * Local state (eq. 22): the subtask, the other subtasks generated with it, the device
      queue, and the queue length / pending-task count of every RES (self-reported, no trust).
      Each subtask is decided on its own; the DAG edges are not observed.
    * Centralised training, decentralised execution: during training the top-K ranked actions
      (K = 3 / 2 / 1 for slots < 50 / < 200 / later, as in the authors' code) are scored by a
      centralised model-based critic with the team reward (eq. 28, value = 10*0.7*sigmoid(psi)
      - sigmoid(stable) as in the released code, GPU-fragmentation term = 0); the best one is
      executed and stored. Loss (eq. 36): cross-entropy to the stored action - Θ * entropy,
      plus Huber loss for the frequency head. Soft-updated target actor (ξ = 0.01) and a
      periodic reset of the last hidden layer (eq. 38).
    * Execution (evaluation): every vehicle takes its own actor's most probable action."""
    name, obs_mode = "MA-LHTO", "reported"

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.n_in = 7 + 2 * self.M
        self.agents = [_MALHTODevice(self.n_in, self.M + 1, cfg) for _ in range(env.N)]
        self._last = 0

    def _reported_load(self, r):
        load = r.backlog_cycles() / (r.F * len(r.cores))
        return load * self.cfg["report_factor"] if r.mal else load

    def _states(self, task):
        env, c = self.env, self.cfg
        v, subs = env.veh[task.vid], task.subs
        B, D = sum(s.b for s in subs), sum(s.d for s in subs)
        common = [v.Q / 1e7, sum(s.rem for s in v.local) / c["f_local_max"] / 10.0]
        for r in env.res:
            common += [self._reported_load(r) / 10.0, r.n_jobs() / 20.0]
        return np.array([[s.b / 1e9, s.d / 1e6, (len(subs) - 1) / 8.0, (B - s.b) / 1e9, (D - s.d) / 1e6]
                         + common for s in subs], np.float32)

    def _value(self, task, opt, f):
        """Centralised critic: estimated completion time of the job under `opt`
        (0 = local, j + 1 = RES j) and the RES backlog it leaves, from the reported state."""
        env, c = self.env, self.cfg
        v, now = env.veh[task.vid], env.t
        local_free = now + sum(s.rem / s.f for s in v.local)
        tx_free, rate = max(now, v.tx_free), env.rate(v)
        near, _ = env.nearest(v)
        blk = env.chain.est_delay(now)
        load = [self._reported_load(r) for r in env.res]
        free = [now + l for l in load]
        fin = []
        for s, o in zip(task.subs, opt):
            ready = max((fin[p.k] for p in s.preds), default=now)
            if o == 0:
                start = max(ready, local_free); local_free = start + s.b / f; fin.append(local_free)
            else:
                j = int(o) - 1; r = env.res[j]
                tx_free += c["overhead"] * s.d / rate
                arrive = tx_free + abs(near - j) * (c["backhaul_hop_s"] + s.d / c["backhaul_bps"]) + blk
                start = max(arrive, ready, free[j])
                free[j] += s.b / (r.F * len(r.cores)); load[j] += s.b / (r.F * len(r.cores))
                fin.append(start + s.b / r.F + blk)
        T = max(max(fin) - now, 1e-3)
        psi = (task.size / 1e6) / T
        stable = sum(load) / (10.0 * len(load))
        sig = lambda z: 1.0 / (1.0 + math.exp(-z))
        return 10.0 * 0.7 * sig(psi) - sig(stable)

    @torch.no_grad()
    def act(self, obs, task, explore):
        ag = self.agents[task.vid % len(self.agents)]
        st = self._states(task)
        logits, fu = ag.net(torch.as_tensor(st, device=DEVICE))
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        u = float(fu.mean())
        if explore:                                     # centralised training: pick best of top-K
            slot = int(self.env.t)
            K = min(3 if slot < 50 else 2 if slot < 200 else 1, self.M + 1)
            cands = [order[:, i] for i in range(K)]
            vals = [self._value(task, cand, self.f_of(u)) for cand in cands]
            opt = cands[int(np.argmax(vals))]
            for k in range(len(task.subs)):
                ag.add(st[k], opt[k], u)
        else:                                           # decentralised execution
            opt = order[:, 0]
        x = [int(o > 0) for o in opt]
        y = [max(int(o) - 1, 0) for o in opt]
        return x, y, self.f_of(u), np.array([1.0], np.float32)

    def reward(self, task):                             # logged only; learning uses the critic above
        return lyapunov_reward(task, self.cfg, trust_aware=False)

    def store(self, task, r):
        self._last = task.vid % len(self.agents)

    def train(self):
        c, ag = self.cfg, self.agents[self._last]
        if ag.n < c["batch"]:
            return
        i = np.random.randint(0, ag.n, c["batch"])
        s = torch.as_tensor(ag.s[i], device=DEVICE)
        a = torch.as_tensor(ag.a[i], device=DEVICE)
        u = torch.as_tensor(ag.u[i], device=DEVICE)
        logits, f = ag.net(s)
        p = torch.softmax(logits, 1)
        entropy = -(p * torch.log(p + 1e-8)).sum(1).mean()
        loss = F.cross_entropy(logits, a) + F.smooth_l1_loss(f.squeeze(1), u) - c["malhto_entropy"] * entropy
        ag.opt.zero_grad(); loss.backward(); ag.opt.step()
        soft_update(ag.target, ag.net, c["malhto_xi"])
        ag.steps += 1
        if ag.steps % c["malhto_reset_every"] == 0:     # eq. (38): reset the last hidden layer
            with torch.no_grad():
                last = ag.net.body[2]
                last.weight.normal_(0.0, c["malhto_reset_std"]); last.bias.zero_()


class IOPO(BaseAgent):
    """Two-stage DRL: stage 1 DQN picks the offloading pattern, stage 2 sets the local
    CPU frequency in closed form to minimise w_t*delay + w_e*energy."""
    name = "IOPO"
    W_T, W_E = 1.0, 0.1

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.na = 1 + 2 * self.M
        self.dqn = DQN(self.od, self.na, cfg, cfg["hidden"])
        self.rng = np.random.default_rng(cfg["seed"] + 3)
        # argmin_f  W_T*b/f + W_E*kappa*b*f^2  ->  f* = (W_T / (2 W_E kappa))^(1/3)
        self.f_star = float(np.clip((self.W_T / (2 * self.W_E * cfg["kappa"])) ** (1 / 3),
                                    cfg["f_local_min"], cfg["f_local_max"]))

    def act(self, obs, task, explore):
        a = int(self.rng.integers(self.na)) if explore and self.rng.random() < self.eps else self.dqn.greedy(obs)
        x, y = pattern_to_xy(a, task, self.M)
        return x, y, self.f_star, np.array([a], np.float32)

    def reward(self, task):
        return cost_reward(task, self.W_T, self.W_E)

    def store(self, task, r):
        self.dqn.buf.add(task.obs, task.act, r)

    def train(self):
        self.dqn.train()


class DECO(BaseAgent):
    """TD3 offloading: continuous action [offload(K), RES(K), f] quantised to decisions;
    twin critics, target policy smoothing and delayed actor updates."""
    name = "DECO"

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        K, h = self.K, cfg["hidden"]
        self.ad = 2 * K + 1
        self.actor = mlp(self.od, self.ad, h).to(DEVICE)
        self.q1 = mlp(self.od + self.ad, 1, h).to(DEVICE)
        self.q2 = mlp(self.od + self.ad, 1, h).to(DEVICE)
        self.actor_t = copy.deepcopy(self.actor)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])
        self.opt_c = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["lr_critic"])
        self.buf = Replay(cfg["buffer"], self.od, self.ad)
        self.rng = np.random.default_rng(cfg["seed"] + 4)
        self.it = 0

    @torch.no_grad()
    def act(self, obs, task, explore):
        a = torch.sigmoid(self.actor(torch.as_tensor(obs, device=DEVICE).unsqueeze(0)))[0].cpu().numpy()
        if explore:
            a = np.clip(a + self.rng.normal(0, 0.1 + 0.4 * self.eps, a.shape), 0, 1)
        k, K = len(task.subs), self.K
        x = (a[:k] > 0.5).astype(int).tolist()
        y = np.minimum((a[K: K + k] * self.M).astype(int), self.M - 1).tolist()
        return x, y, self.f_of(a[-1]), a.astype(np.float32)

    def reward(self, task):
        return cost_reward(task, 1.0, 0.1)

    def store(self, task, r):
        self.buf.add(task.obs, task.act, r)

    def train(self):
        if self.buf.n < self.cfg["batch"]:
            return
        o, a, r = self.buf.sample(self.cfg["batch"])
        oa = torch.cat([o, a], 1)
        lc = F.mse_loss(self.q1(oa).squeeze(1), r) + F.mse_loss(self.q2(oa).squeeze(1), r)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()
        self.it += 1
        if self.it % 2 == 0:                                   # delayed policy update
            pi = torch.sigmoid(self.actor(o))
            la = -self.q1(torch.cat([o, pi], 1)).mean()
            self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
            soft_update(self.actor_t, self.actor, self.cfg["tau"])


ALGOS = {"TaBRL": TaBRL, "MA-LHTO": MALHTO, "IOPO": IOPO, "DECO": DECO, "DAG-Local": DAGLocal}



# %% [markdown]
# ## 5. Episode runner, training and evaluation

# %%
def run_episode(agent, sched, cfg, n_veh, seed, train, horizon=None, trace=False):
    env = IoVEnv(cfg, n_veh, seed)
    agent.env = env
    horizon = horizon or len(sched)
    backlog, energy, rewards = [], [], []
    stores = 0

    def handle(done):
        nonlocal stores
        for task in done:
            r = agent.reward(task); rewards.append(r)
            if train and task.act is not None:
                agent.store(task, r); stores += 1
                if stores % getattr(agent, "train_every", cfg["train_every"]) == 0:
                    agent.train()

    for s in range(horizon):
        env.begin_slot()
        for vid, job in sched[s]:
            task = Task(vid, env.t, job)
            build = env.build_graph_obs if getattr(agent, "graph_obs", False) else env.build_obs
            task.obs = build(vid, task, agent.obs_mode)
            x, y, f, task.act = agent.act(task.obs, task, explore=train)
            env.submit(task, x, y, f)
        handle(env.run_slot())
        if trace:
            backlog.append(sum(v.Q for v in env.veh))
            energy.append(sum(v.total_energy for v in env.veh))
    extra = 0
    while not env.all_done() and extra < cfg["drain_max"]:
        env.begin_slot(); handle(env.run_slot()); extra += 1

    fin = [t for t in env.tasks if t.complete is not None]
    delays = [task_delay(t) for t in fin]
    # unfinished tasks (only if drain_max hit) are counted with their age so they are not hidden
    delays += [env.t - t.arr for t in env.tasks if t.complete is None]
    makespan = max([t.complete for t in fin] + [env.t if len(fin) < len(env.tasks) else 0, 1.0])
    return dict(
        completion=float(np.mean(delays)) if delays else 0.0,
        # processing capacity = mean per-DAG processing rate (bits / completion time),
        # i.e. the throughput term size/T of problem P2
        throughput=float(np.mean([t.size / task_delay(t) for t in fin])) if fin else 0.0,
        sys_throughput=float(sum(t.size for t in fin) / makespan),
        reward=float(np.mean(rewards)) if rewards else 0.0,
        energy=float(sum(v.total_energy for v in env.veh)),
        offload=float(np.mean([np.mean(t.info["x"]) for t in env.tasks])) if env.tasks else 0.0,
        tampered=int(sum(r.fail for r in env.res)),
        n_tasks=len(env.tasks), backlog=backlog, energy_t=energy)


def train_agent(name, train_jobs, cfg, n_veh):
    set_seed(cfg["seed"])
    probe = IoVEnv(cfg, n_veh, 0)
    agent = ALGOS[name](probe, cfg)
    hist = []
    if name == "DAG-Local":
        return agent, hist
    rng = np.random.default_rng(cfg["seed"] + 100)
    E = cfg["train_episodes"]
    if not E:
        raise ValueError("Number of training epochs not set: run with --epochs N "
                         "(e.g. python tabrl_iov.py --epochs 150)")
    t0 = time.time()
    for ep in range(E):
        agent.set_progress(ep / E)
        sched = make_window(train_jobs, n_veh, cfg["horizon"], cfg, rng)
        m = run_episode(agent, sched, cfg, n_veh, seed=10_000 + ep, train=True)
        hist.append(dict(ep=ep, completion=m["completion"], throughput=m["throughput"], reward=m["reward"]))
        if ep % 10 == 0 or ep == E - 1:
            print(f"  [{name:9s}] epoch {ep + 1:3d}/{E}  T_c={m['completion']:.3f}s  "
                  f"thr={m['throughput'] / 1e6:.2f} Mbit/s  r={m['reward']:.3f}  "
                  f"off={m['offload']:.2f}  eps={agent.eps:.2f}  ({time.time() - t0:.0f}s)")
    return agent, hist


def evaluate(agent, windows, cfg, n_veh, horizon=None, trace=False):
    ms = [run_episode(agent, sched, cfg, n_veh, seed=seed, train=False, horizon=horizon, trace=trace)
          for seed, sched in windows]
    out = {k: float(np.mean([m[k] for m in ms])) for k in
           ("completion", "throughput", "sys_throughput", "reward", "energy", "offload", "tampered")}
    out["completion_std"] = float(np.std([m["completion"] for m in ms]))
    out["throughput_std"] = float(np.std([m["throughput"] for m in ms]))
    if trace:
        out["backlog"] = np.mean([m["backlog"] for m in ms], 0).tolist()
        out["energy_t"] = np.mean([m["energy_t"] for m in ms], 0).tolist()
    return out


def fixed_windows(jobs, n_veh, horizon, cfg, n, seed):
    rng = np.random.default_rng(seed)
    return [(seed * 100 + i, make_window(jobs, n_veh, horizon, cfg, rng)) for i in range(n)]


# %% [markdown]
# ## 6. Figures

# %%
ORDER = ["TaBRL", "MA-LHTO", "IOPO", "DECO", "DAG-Local"]
C_TIME, C_CAP = "#B08FAE", "#7FB8D8"      # mauve / light blue as in the paper
C_LINES = {"TaBRL": "#1a1a1a", "MA-LHTO": "#E03C31", "IOPO": "#9B59B6", "DECO": "#4FB6E0"}
# Axis ranges as multiples of each panel's largest value: (completion-time axis, capacity axis).
# They only set how tall the bars look, never the numbers. The ratios are read off the paper's
# Fig. 3: (a) training, time axis 0-20 for a max of 18.19, capacity axis 0-5000 for a max of
# 2693; (b) test, time axis ~0-17.5 for a max of 16.67, capacity axis ~0-3500 for a max of 3128.
AXIS_HEADROOM = {"train": (1.10, 1.86), "test": (1.05, 1.12)}


def _bar_panel(ax, data, title=None, xlabel=None, headroom=(1.10, 1.86)):
    """Grouped bars on twin axes: completion time (left) and processing capacity (right)."""
    algs = [a for a in ORDER if a in data]
    ct = np.array([data[a]["completion"] for a in algs])
    cap = np.array([data[a]["throughput"] / 1e3 for a in algs])        # kbit/s
    x, w = np.arange(len(algs)), 0.4
    b1 = ax.bar(x - w / 2, ct, w, color=C_TIME, label="Completion time")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, cap, w, color=C_CAP, label="Processing capacity")
    for b, v in zip(b1, ct):
        ax.text(b.get_x() + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(b2, cap):
        ax2.text(b.get_x() + w / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, ct.max() * headroom[0]); ax2.set_ylim(0, cap.max() * headroom[1])
    ax.set_xticks(x); ax.set_xticklabels(algs, rotation=15)
    ax.set_ylabel("Mean completion time (s)"); ax2.set_ylabel("Processing capacity (kbit/s)")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    return b1, b2


def plot_fig3(res, path, title="Impact of different algorithms on train and test set"):
    """Side-by-side layout: Training set | Test set."""
    plt.rcParams.update(plt.rcParamsDefault)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, split, name in zip(axes, ("train", "test"), ("Training set", "Test set")):
        b1, b2 = _bar_panel(ax, res[split], title=name, headroom=AXIS_HEADROOM[split])
    fig.legend([b1, b2], ["Completion time", "Processing capacity"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_fig3_paper_style(res, path):
    """Stacked layout as printed in the paper: (a) training set over (b) test set."""
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.4))
    for ax, split, tag in zip(axes, ("train", "test"), ("(a)", "(b)")):
        b1, b2 = _bar_panel(ax, res[split], xlabel=f"Algorithms\n{tag}", headroom=AXIS_HEADROOM[split])
        ax.legend([b1, b2], ["Completion time", "Processing capacity"], loc="upper center",
                  ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300); plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def replot_fig3(out_dir, train=None, test=None):
    """Redraw Fig. 3 of a finished run with other y-axis ranges, without retraining.

    Display only: every plotted value is read unchanged from <out_dir>/results.json, and
    comparisons between algorithms on the same axis are not affected. train / test are
    (completion-time axis, capacity axis) ranges as multiples of the panel's largest bar,
    e.g. train=(1.10, 2.2). Writes fig3_completion_capacity_custom.png and
    fig3_paper_style_custom.png next to the original figures."""
    res = json.load(open(os.path.join(out_dir, "results.json")))["fig3"]
    saved, saved_order = dict(AXIS_HEADROOM), list(ORDER)
    if train:
        AXIS_HEADROOM["train"] = tuple(train)
    if test:
        AXIS_HEADROOM["test"] = tuple(test)
    ORDER.extend(a for a in res["train"] if a not in ORDER)     # e.g. TaBRL-GNN
    try:
        plot_fig3(res, os.path.join(out_dir, "fig3_completion_capacity_custom.png"))
        plot_fig3_paper_style(res, os.path.join(out_dir, "fig3_paper_style_custom.png"))
        for split in ("train", "test"):
            d, (ht, hc) = res[split], AXIS_HEADROOM[split]
            ct_max = max(m["completion"] for m in d.values())
            cap_max = max(m["throughput"] for m in d.values())
            t, c = d["TaBRL"]["completion"], d["TaBRL"]["throughput"]
            brown, blue = t / (ht * ct_max), c / (hc * cap_max)
            equal_at = ht * (c / cap_max) * (ct_max / t)
            print(f"{split:5s}: axes (time {ht:.2f}, capacity {hc:.2f})  TaBRL bar heights: "
                  f"completion {brown:.0%}, capacity {blue:.0%}  -> "
                  f"{'brown taller' if brown > blue else 'blue taller'}; "
                  f"the two are equal at capacity axis {equal_at:.2f}")
    finally:
        AXIS_HEADROOM.clear(); AXIS_HEADROOM.update(saved)
        ORDER[:] = saved_order


def plot_security(res, path):
    """Tampered results returned by compromised RESs (caught by on-chain verification)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split, name in zip(axes, ("train", "test"), ("Training set", "Test set")):
        algs = [a for a in ORDER if a in res[split] and a != "DAG-Local"]
        v = [res[split][a]["tampered"] for a in algs]
        bars = ax.bar(algs, v, 0.55, color=[C_TIME if a == "TaBRL" else C_CAP for a in algs])
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(name)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Tampered subtask results per episode")
    fig.suptitle("Exposure to compromised edge servers", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_fig4(res4, path):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    for ax, (nv, data), tag in zip(axes, res4.items(), ("(a)", "(b)")):
        ax2 = ax.twinx()
        for a in [a for a in ORDER if a != "DAG-Local"]:
            if a not in data:
                continue
            t = np.arange(1, len(data[a]["backlog"]) + 1)
            ax.plot(t, np.cumsum(data[a]["backlog"]), color=C_LINES.get(a), lw=1.4, label=a)
            ax2.plot(t, data[a]["energy_t"], color=C_LINES.get(a), lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel(f"Time (slot)\n{tag} Vehicles = {nv}")
        ax.set_ylabel("Accumulated awaiting task size (bit·slot)")
        ax2.set_ylabel("Energy consumption (J, dashed)")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_training(hists, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, h in hists.items():
        if not h:
            continue
        ep = [z["ep"] for z in h]
        sm = lambda v: pd.Series(v).rolling(8, min_periods=1).mean()
        axes[0].plot(ep, sm([z["completion"] for z in h]), label=a, color=C_LINES.get(a))
        axes[1].plot(ep, sm([z["throughput"] / 1e3 for z in h]), label=a, color=C_LINES.get(a))
    axes[0].set_ylabel("Completion time (s)"); axes[1].set_ylabel("Processing capacity (kbit/s)")
    for ax in axes:
        ax.set_xlabel("Training epoch"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# %% [markdown]
# ## 7. Run everything

# %%
SETTING_KEYS = ["train_episodes", "seed", "n_veh", "n_res", "horizon", "eval_windows",
                "arrival_rate", "n_malicious", "data_path", "out_dir"]


def print_settings(cfg):
    names = {"train_episodes": "epochs"}
    print("Settings: " + ", ".join(f"{names.get(k, k)}={cfg[k]}" for k in SETTING_KEYS if k in cfg))


def results_table(res):
    """All Fig. 3 values as a table. Two throughput measures are reported:
    processing capacity = mean over DAGs of size / completion time (the bars in Fig. 3),
    system throughput   = all input bits processed / total time until the last DAG finished."""
    return pd.DataFrame([{
        "Split": s, "Algorithm": a,
        "Completion time (s)": round(m["completion"], 3),
        "Processing capacity (kbit/s)": round(m["throughput"] / 1e3, 1),
        "System throughput (Mbit/s)": round(m["sys_throughput"] / 1e6, 3),
        "Energy (J)": round(m["energy"], 1),
        "Offload ratio": round(m["offload"], 3),
        "Tampered results": round(m["tampered"], 1)}
        for s in res for a, m in res[s].items()])


def fig4_table(res4):
    """Fig. 4 values at the end of the horizon: backlog and cumulative energy."""
    return pd.DataFrame([{
        "Vehicles": nv, "Algorithm": a,
        "Accumulated awaiting task size (bit*slot)": float(np.sum(m["backlog"])),
        "Mean awaiting task size (bit)": round(float(np.mean(m["backlog"])), 1),
        "Energy consumption (J)": round(m["energy_t"][-1], 1)}
        for nv, d in res4.items() for a, m in d.items()])


def print_table(title, df):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(df.to_string(index=False))


def apply_cli_args(cfg):
    """Command-line overrides for the script version, e.g.
    python tabrl_iov.py --epochs 100 --seed 7 --data /path/batch_task.csv"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", "--episodes", dest="epochs", type=int, required=True,
                    help="training epochs per algorithm (1 epoch = one simulation run of `horizon` slots)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--vehicles", type=int)
    ap.add_argument("--horizon", type=int, help="time slots per episode")
    ap.add_argument("--eval-windows", type=int)
    ap.add_argument("--data", help="path to batch_task.csv")
    ap.add_argument("--out", help="output directory")
    args, _ = ap.parse_known_args()
    for flag, key in [("epochs", "train_episodes"), ("seed", "seed"), ("vehicles", "n_veh"),
                      ("horizon", "horizon"), ("eval_windows", "eval_windows"), ("data", "data_path"),
                      ("out", "out_dir")]:
        if getattr(args, flag) is not None:
            cfg[key] = getattr(args, flag)
    return cfg


class _Tee:
    """Send every printed line to the notebook/terminal and to a log file, flushing each
    write so no line is held back or lost if the notebook display drops output."""
    def __init__(self, path, stream):
        self.f, self.stream = open(path, "a", encoding="utf-8"), stream

    def write(self, text):
        self.stream.write(text); self.stream.flush()
        self.f.write(text); self.f.flush()

    def flush(self):
        self.stream.flush(); self.f.flush()


def logged(fn, log_name):
    """Run fn(cfg, ...) with all output also saved to <out_dir>/<log_name>."""
    import sys

    def wrapper(cfg=CFG, *args, **kwargs):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        old, tee = sys.stdout, _Tee(os.path.join(cfg["out_dir"], log_name), sys.stdout)
        sys.stdout = tee
        try:
            return fn(cfg, *args, **kwargs)
        finally:
            sys.stdout = old; tee.f.close()
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def main(cfg=CFG, return_agents=None):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg["seed"])
    print(f"Device: {DEVICE}")
    print_settings(cfg)
    train_jobs, test_jobs = load_dag_jobs(cfg)
    N = cfg["n_veh"]
    win_train = fixed_windows(train_jobs, N, cfg["horizon"], cfg, cfg["eval_windows"], seed=1)
    win_test = fixed_windows(test_jobs, N, cfg["horizon"], cfg, cfg["eval_windows"], seed=2)

    agents, hists, res = {}, {}, {"train": {}, "test": {}}
    for name in ORDER:
        print(f"\n=== {name} ===")
        agents[name], hists[name] = train_agent(name, train_jobs, cfg, N)
        for split, win in (("train", win_train), ("test", win_test)):
            res[split][name] = evaluate(agents[name], win, cfg, N)
            m = res[split][name]
            print(f"  eval {split:5s}: completion {m['completion']:.3f}s  "
                  f"capacity {m['throughput'] / 1e3:.0f} kbit/s  energy {m['energy']:.0f} J  "
                  f"offload {m['offload']:.2f}  tampered {m['tampered']:.0f}")

    if return_agents is not None:
        return_agents.update(agents)
    res4 = {}
    for nv in cfg["fig4_veh"]:
        wins = fixed_windows(test_jobs, nv, cfg["fig4_horizon"], cfg, cfg["eval_windows"], seed=3 + nv)
        res4[nv] = {a: evaluate(agents[a], wins, cfg, nv, trace=True)
                    for a in ORDER if a != "DAG-Local"}

    plot_fig3(res, os.path.join(cfg["out_dir"], "fig3_completion_capacity.png"))
    plot_fig3_paper_style(res, os.path.join(cfg["out_dir"], "fig3_paper_style.png"))
    plot_fig4(res4, os.path.join(cfg["out_dir"], "fig4_backlog_energy.png"))
    plot_security(res, os.path.join(cfg["out_dir"], "security_tampered.png"))
    plot_training(hists, os.path.join(cfg["out_dir"], "training_curves.png"))

    table, t4 = results_table(res), fig4_table(res4)
    table.to_csv(os.path.join(cfg["out_dir"], "results.csv"), index=False)
    t4.to_csv(os.path.join(cfg["out_dir"], "fig4_values.csv"), index=False)
    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as fh:
        json.dump(dict(fig3=res, fig4={str(k): v for k, v in res4.items()}, history=hists, cfg=cfg),
                  fh, default=float)
    print_table("Fig. 3 values (train and test)", table)
    print("Processing capacity = mean of size/completion time per DAG (Fig. 3 bars); "
          "system throughput = all bits processed / total time.")
    print_table(f"Fig. 4 values after {cfg['fig4_horizon']} slots", t4)
    print(f"\nFigures and tables written to {cfg['out_dir']}/")
    return res, res4, hists


main = logged(main, "train_log.txt")


if __name__ == "__main__":
    main(apply_cli_args(CFG))
