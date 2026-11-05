
from tabrl_iov import *  # notebook: inlined
from tabrl_iov import _bar_panel  # notebook: inlined

FT, FS, FV, FG = 5, 5, 6, 2          # feature sizes: subtask, RES, vehicle, global


def batch_graphs(obs_list):
    """Pad a list of graph observations into dense tensors with node masks."""
    B = len(obs_list)
    K = max(o["xt"].shape[0] for o in obs_list)
    M = max(o["xs"].shape[0] for o in obs_list)
    xt = np.zeros((B, K, FT), np.float32); A = np.zeros((B, K, K), np.float32)
    mt = np.zeros((B, K), np.float32)
    xs = np.zeros((B, M, FS), np.float32); ms = np.zeros((B, M), np.float32)
    xv = np.zeros((B, FV), np.float32); xg = np.zeros((B, FG), np.float32)
    for i, o in enumerate(obs_list):
        k, m = o["xt"].shape[0], o["xs"].shape[0]
        xt[i, :k] = o["xt"]; A[i, :k, :k] = o["A"]; mt[i, :k] = 1
        xs[i, :m] = o["xs"]; ms[i, :m] = 1
        xv[i] = o["xv"]; xg[i] = o["xg"]
    t = lambda z: torch.as_tensor(z, device=DEVICE)
    return dict(xt=t(xt), A=t(A), mt=t(mt), xs=t(xs), ms=t(ms), xv=t(xv), xg=t(xg))


def repeat_graph(g, n):
    return {k: v.expand(n, *v.shape[1:]) for k, v in g.items()}


def masked_mean(h, m):
    return (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def masked_max(h, m):
    return h.masked_fill(m.unsqueeze(-1) == 0, -1e4).max(1).values


class GNNLayer(nn.Module):
    """One round of message passing: subtask <- predecessors, successors, RESs, context;
    RES <- subtasks."""
    def __init__(self, h):
        super().__init__()
        self.self_t, self.pred, self.succ = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.srv, self.ctx = nn.Linear(h, h), nn.Linear(h, h)
        self.self_s, self.task = nn.Linear(h, h), nn.Linear(h, h)
        self.norm_t, self.norm_s = nn.LayerNorm(h), nn.LayerNorm(h)

    def forward(self, ht, hs, g, c):
        A, mt, ms = g["A"], g["mt"], g["ms"]
        At = A.transpose(1, 2)
        m_pred = (A @ ht) / A.sum(-1, keepdim=True).clamp(min=1)
        m_succ = (At @ ht) / At.sum(-1, keepdim=True).clamp(min=1)
        upd_t = F.relu(self.self_t(ht) + self.pred(m_pred) + self.succ(m_succ)
                       + self.srv(masked_mean(hs, ms)).unsqueeze(1) + self.ctx(c).unsqueeze(1))
        ht = self.norm_t(ht + upd_t) * mt.unsqueeze(-1)
        upd_s = F.relu(self.self_s(hs) + self.task(masked_mean(ht, mt)).unsqueeze(1))
        hs = self.norm_s(hs + upd_s) * ms.unsqueeze(-1)
        return ht, hs


class GraphEncoder(nn.Module):
    """Embeds subtask nodes, RES nodes and the vehicle/global context. With act_in=True
    (Critic) each subtask node also receives its offload decision and the embedding of the
    RES it is sent to."""
    def __init__(self, h, layers, act_in=False):
        super().__init__()
        self.act_in = act_in
        self.t_in = nn.Sequential(nn.Linear(FT + int(act_in), h), nn.ReLU(), nn.Linear(h, h))
        self.s_in = nn.Sequential(nn.Linear(FS, h), nn.ReLU(), nn.Linear(h, h))
        self.c_in = nn.Sequential(nn.Linear(FV + FG, h), nn.ReLU(), nn.Linear(h, h))
        if act_in:
            self.a_proj = nn.Linear(h, h)
        self.layers = nn.ModuleList([GNNLayer(h) for _ in range(layers)])

    def forward(self, g, x=None, ypi=None):
        c = self.c_in(torch.cat([g["xv"], g["xg"]], -1))
        hs = self.s_in(g["xs"]) * g["ms"].unsqueeze(-1)
        if self.act_in:
            ht = self.t_in(torch.cat([g["xt"], x.unsqueeze(-1)], -1))
            ht = ht + self.a_proj(x.unsqueeze(-1) * (ypi @ hs))       # chosen RES embedding
        else:
            ht = self.t_in(g["xt"])
        ht = (ht + c.unsqueeze(1)) * g["mt"].unsqueeze(-1)
        for layer in self.layers:
            ht, hs = layer(ht, hs, g, c)
        return ht, hs, c


class GNNOffloadActor(nn.Module):
    """Actor 1: offload probability per subtask + score per (subtask, RES) pair."""
    def __init__(self, h, layers):
        super().__init__()
        self.enc = GraphEncoder(h, layers)
        self.off = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.pair = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, g):
        ht, hs, _ = self.enc(g)
        B, K, H = ht.shape; M = hs.shape[1]
        p = torch.sigmoid(self.off(ht).squeeze(-1)) * g["mt"]
        pair = torch.cat([ht.unsqueeze(2).expand(B, K, M, H), hs.unsqueeze(1).expand(B, K, M, H)], -1)
        logit = self.pair(pair).squeeze(-1)
        logit = logit.masked_fill(g["ms"].unsqueeze(1) == 0, -1e4)
        return p, logit


class GNNFreqActor(nn.Module):
    """Actor 2: local CPU frequency from the pooled DAG embedding and the context."""
    def __init__(self, h, layers):
        super().__init__()
        self.enc = GraphEncoder(h, layers)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, g):
        ht, _, c = self.enc(g)
        return torch.sigmoid(self.head(torch.cat([masked_mean(ht, g["mt"]), c], -1)))


class GNNCritic(nn.Module):
    def __init__(self, h, layers):
        super().__init__()
        self.enc = GraphEncoder(h, layers, act_in=True)
        self.head = nn.Sequential(nn.Linear(4 * h + 1, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                  nn.Linear(h, 1))

    def forward(self, g, x, ypi, u):
        ht, hs, c = self.enc(g, x, ypi)
        z = torch.cat([masked_mean(ht, g["mt"]), masked_max(ht, g["mt"]),
                       masked_mean(hs, g["ms"]), c, u], -1)
        return self.head(z).squeeze(-1)


class TaBRLGNN(TaBRL):
    """TaBRL with graph networks instead of the dual-FC Actor and FC Critic. Candidate
    generation (NOP + trust coupling), reward and update rules are inherited unchanged."""
    name, obs_mode, graph_obs = "TaBRL-GNN", "audited", True

    def __init__(self, env, cfg):
        BaseAgent.__init__(self, env, cfg)
        h, L = cfg["gnn_hidden"], cfg["gnn_layers"]
        self.actor_off, self.actor_f = GNNOffloadActor(h, L).to(DEVICE), GNNFreqActor(h, L).to(DEVICE)
        self.t_off, self.t_f = copy.deepcopy(self.actor_off), copy.deepcopy(self.actor_f)
        self.critic = GNNCritic(h, L).to(DEVICE)
        self.opt_a = torch.optim.Adam(list(self.actor_off.parameters()) + list(self.actor_f.parameters()),
                                      lr=cfg["lr_actor"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])
        self.buf, self.cap, self.pos = [], cfg["buffer"], 0
        self.rng = np.random.default_rng(cfg["seed"] + 5)
        self.train_every = cfg["gnn_train_every"]    # fewer updates than FC TaBRL (runtime)

    @torch.no_grad()
    def act(self, obs, task, explore):
        k, M = len(task.subs), obs["xs"].shape[0]
        g = batch_graphs([obs])
        p, l = [z[0].cpu().numpy() for z in self.actor_off(g)]
        tp, tl = [z[0].cpu().numpy() for z in self.t_off(g)]
        u, tu = float(self.actor_f(g)[0]), float(self.t_f(g)[0])
        load, trust, freq = obs["xs"][:, 0] * 10.0, np.clip(obs["xs"][:, 2], 1e-3, 1), obs["xs"][:, 4] * 5e9
        cands = self.candidates(p, l, u, tp, tl, tu, k, load, trust, freq, obs["xt"][:, 0])
        if explore and self.rng.random() < self.eps:
            best = (self.rng.integers(0, 2, k).astype(np.float32), self.rng.integers(0, M, k),
                    float(self.rng.random()))
        else:
            n = len(cands)
            x = torch.as_tensor(np.stack([c[0] for c in cands]), device=DEVICE)
            y = torch.as_tensor(np.stack([c[1] for c in cands]), device=DEVICE)
            u_c = torch.as_tensor([[c[2]] for c in cands], dtype=torch.float32, device=DEVICE)
            q = self.critic(repeat_graph(g, n), x, F.one_hot(y, M).float(), u_c)
            best = cands[int(q.argmax())]
        x, y, uu = best
        return x.astype(int).tolist(), [int(v) for v in y], self.f_of(uu), (x.astype(np.float32), np.asarray(y), uu)

    def store(self, task, r):
        item = (task.obs, task.act, r)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
        self.pos = (self.pos + 1) % self.cap

    def train(self):
        B = self.cfg["batch"]
        if len(self.buf) < B:
            return
        batch = [self.buf[i] for i in np.random.randint(0, len(self.buf), B)]
        g = batch_graphs([b[0] for b in batch])
        K, M = g["xt"].shape[1], g["xs"].shape[1]
        x = np.zeros((B, K), np.float32); y = np.zeros((B, K), np.int64)
        for i, (_, (xi, yi, _), _) in enumerate(batch):
            x[i, :len(xi)] = xi; y[i, :len(yi)] = yi
        x = torch.as_tensor(x, device=DEVICE)
        y_oh = F.one_hot(torch.as_tensor(y, device=DEVICE), M).float()
        u = torch.as_tensor([[b[1][2]] for b in batch], dtype=torch.float32, device=DEVICE)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        mt = g["mt"]

        lc = F.mse_loss(self.critic(g, x, y_oh, u), r)                # TD loss (per-slot, gamma = 0)
        self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()

        p, logit = self.actor_off(g)
        uu = self.actor_f(g)
        q_pi = self.critic(g, p, torch.softmax(logit, -1), uu)
        with torch.no_grad():                                         # advantage-weighted imitation
            w = torch.exp(r - q_pi).clamp(max=10.0); w = w / w.mean()
        bce = (F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), x, reduction="none") * mt).sum(1)
        ce = (-(torch.log_softmax(logit, -1) * y_oh).sum(-1) * x * mt).sum(1)
        mse = (uu.squeeze(1) - u.squeeze(1)) ** 2
        la = (w * (bce + ce + mse)).mean() - 0.1 * q_pi.mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        soft_update(self.t_off, self.actor_off, self.cfg["tau"])
        soft_update(self.t_f, self.actor_f, self.cfg["tau"])


ALGOS["TaBRL-GNN"] = TaBRLGNN
if "TaBRL-GNN" not in ORDER:
    ORDER.insert(1, "TaBRL-GNN")
C_LINES["TaBRL-GNN"] = "#E69F00"
CFG.update(gnn_hidden=64, gnn_layers=2, gnn_train_every=2)


# %% [markdown]
# ## Thesis experiments

# %%
def plot_bars(res, algs, path, title):
    """Single-panel grouped bars (completion time, processing capacity) for a test result."""
    fig, ax = plt.subplots(figsize=(1.6 * len(algs) + 3, 4.8))
    _bar_panel(ax, {a: res[a] for a in algs}, title=title, headroom=(1.2, 1.2))
    fig.legend(["Completion time", "Processing capacity"], loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=300); plt.close(fig)


def plot_scaling(xs, curves, xlabel, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for a, pts in curves.items():
        xv = [x for x in xs if x in pts]
        axes[0].plot(xv, [pts[x]["completion"] for x in xv], marker="o", lw=2, label=a, color=C_LINES.get(a))
        axes[1].plot(xv, [pts[x]["throughput"] / 1e3 for x in xv], marker="o", lw=2, label=a,
                     color=C_LINES.get(a))
    axes[0].set_ylabel("Mean completion time (s, log scale)"); axes[1].set_ylabel("Processing capacity (kbit/s)")
    axes[0].set_yscale("log")        # overloaded points (100+ s) would flatten the rest
    for ax in axes:
        ax.set_xlabel(xlabel); ax.set_xticks(xs); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def thesis_main(cfg=CFG, run_e1=True, run_e2=True, run_e3=True):
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    summary = {}

    # ---- E1: base scenario, all algorithms (produces the same figures as the base notebook)
    agents = {}
    if run_e1 or run_e3:
        print("\n######## E1: base scenario ########")
        res, res4, hists = main(dict(cfg, out_dir=os.path.join(out, "E1_base")), return_agents=agents)
        summary["E1"] = res

    # ---- E2: larger DAGs (2..k_large subtasks); load kept comparable by scaling arrivals
    if run_e2:
        print("\n######## E2: larger DAGs ########")
        c2 = dict(cfg, k_max=cfg["k_large"])
        tr2, te2 = load_dag_jobs(c2)
        mean_k = np.mean([len(j["b"]) for j in tr2])
        c2["arrival_rate"] = cfg["arrival_rate"] * cfg["base_mean_k"] / mean_k
        print(f"  mean subtasks {mean_k:.2f} -> arrival_rate {c2['arrival_rate']:.3f}")
        N = c2["n_veh"]
        win = fixed_windows(te2, N, c2["horizon"], c2, c2["eval_windows"], seed=2)
        res2 = {}
        for a in ORDER:                                   # every algorithm, as in E1
            ag, _ = train_agent(a, tr2, c2, N)
            res2[a] = evaluate(ag, win, c2, N)
            print(f"  E2 {a:10s} completion {res2[a]['completion']:.3f}s  capacity {res2[a]['throughput']/1e3:.0f} kbit/s")
        plot_bars(res2, list(res2), os.path.join(out, "E2_large_dags.png"),
                  f"Test set, DAGs with up to {cfg['k_large']} subtasks")
        t2 = results_table({"test": res2}).drop(columns="Split")
        t2.to_csv(os.path.join(out, "E2_values.csv"), index=False)
        print_table(f"E2 values: DAGs with up to {cfg['k_large']} subtasks (test set)", t2)
        summary["E2"] = res2

    # ---- E3: zero-shot scalability of the E1 policies (trained with N=10, M=4)
    if run_e3:
        print("\n######## E3: zero-shot scalability ########")
        _, te = load_dag_jobs(cfg)
        veh = {a: {} for a in ORDER}
        for nv in cfg["scale_veh"]:
            win = fixed_windows(te, nv, cfg["horizon"], cfg, cfg["eval_windows"], seed=50 + nv)
            for a in ORDER:
                veh[a][nv] = evaluate(agents[a], win, cfg, nv)
            print(f"  N={nv:2d}  " + "  ".join(f"{a} {veh[a][nv]['completion']:.2f}s" for a in ORDER))
        plot_scaling(list(cfg["scale_veh"]), veh, "Number of vehicles N",
                     os.path.join(out, "E3_scale_vehicles.png"),
                     f"Zero-shot: trained with N = {cfg['n_veh']}, tested with N = {list(cfg['scale_veh'])}")
        srv = {"TaBRL-GNN": {}, "DAG-Local": {}}
        for m in cfg["scale_res"]:
            cm = dict(cfg, n_res=m)
            win = fixed_windows(te, cfg["n_veh"], cfg["horizon"], cm, cfg["eval_windows"], seed=80 + m)
            for a in ("TaBRL-GNN", "DAG-Local"):          # size-agnostic policies
                srv[a][m] = evaluate(agents[a], win, cm, cfg["n_veh"])
            if m == cfg["n_res"]:                  # FC networks only run with the training M
                for a in ORDER:
                    srv.setdefault(a, {})[m] = evaluate(agents[a], win, cm, cfg["n_veh"])
            print(f"  M={m}  GNN {srv['TaBRL-GNN'][m]['completion']:.2f}s  "
                  f"Local {srv['DAG-Local'][m]['completion']:.2f}s")
        plot_scaling(list(cfg["scale_res"]), srv, "Number of edge servers M",
                     os.path.join(out, "E3_scale_servers.png"),
                     f"Zero-shot: trained with M = {cfg['n_res']} (FC agents only run at M = {cfg['n_res']})")
        rows = [dict(Sweep="vehicles", Size=n, Algorithm=a, **{
                    "Completion time (s)": round(m["completion"], 3),
                    "Processing capacity (kbit/s)": round(m["throughput"] / 1e3, 1),
                    "System throughput (Mbit/s)": round(m["sys_throughput"] / 1e6, 3)})
                for a, pts in veh.items() for n, m in pts.items()]
        rows += [dict(Sweep="servers", Size=n, Algorithm=a, **{
                    "Completion time (s)": round(m["completion"], 3),
                    "Processing capacity (kbit/s)": round(m["throughput"] / 1e3, 1),
                    "System throughput (Mbit/s)": round(m["sys_throughput"] / 1e6, 3)})
                 for a, pts in srv.items() for n, m in pts.items()]
        t3 = pd.DataFrame(rows).sort_values(["Sweep", "Size", "Algorithm"])
        t3.to_csv(os.path.join(out, "E3_values.csv"), index=False)
        print_table("E3 values: zero-shot scalability (test set)", t3)
        summary["E3"] = dict(vehicles=veh, servers=srv)

    with open(os.path.join(out, "thesis_results.json"), "w") as fh:
        json.dump(summary, fh, default=float)
    print(f"\nThesis figures written to {out}/")
    return summary


thesis_main = logged(thesis_main, "thesis_log.txt")


CFG.update(k_large=20, base_mean_k=5.0, scale_veh=(5, 10, 15, 20), scale_res=(3, 4, 6, 8))

def apply_thesis_args(cfg):
    """Thesis-only command-line options (the common ones, including the required --epochs,
    are read by apply_cli_args)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="e1,e2,e3", help="which experiments to run, e.g. e1,e3")
    ap.add_argument("--gnn-train-every", type=int, help="TaBRL-GNN updates once every N finished tasks")
    ap.add_argument("--k-large", type=int, help="E2: largest DAG size")
    ap.add_argument("--scale-veh", help="E3: vehicle counts, e.g. 5,10,15,20")
    ap.add_argument("--scale-res", help="E3: edge-server counts, e.g. 3,4,6,8")
    args, _ = ap.parse_known_args()
    if args.gnn_train_every is not None:
        cfg["gnn_train_every"] = args.gnn_train_every
    if args.k_large is not None:
        cfg["k_large"] = args.k_large
    if args.scale_veh:
        cfg["scale_veh"] = tuple(int(v) for v in args.scale_veh.split(","))
    if args.scale_res:
        cfg["scale_res"] = tuple(int(v) for v in args.scale_res.split(","))
    run = {e.strip().lower() for e in args.experiments.split(",")}
    return cfg, dict(run_e1="e1" in run, run_e2="e2" in run, run_e3="e3" in run)


if __name__ == "__main__":
    cfg, runs = apply_thesis_args(apply_cli_args(CFG))
    thesis_main(cfg, **runs)
