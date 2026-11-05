
from tabrl_gnn import *  # notebook: inlined
from tabrl_gnn import _bar_panel, masked_mean, masked_max  # notebook: inlined
import itertools
import os

import pq_crypto as PQ  # notebook: inlined

# Signature schemes: ecdsa (classical), dilithium2/3/5 = ML-DSA-44/65/87 (post-quantum).
# With crypto="real" (default) every result, vote and quorum certificate is really signed
# and verified, and every vehicle-server session uses ML-KEM-768 + AES-256-GCM (pq_crypto.py).
# The consensus delay uses the sign/verify times measured on this machine.
SIG = PQ.SCHEMES

CO_CFG = dict(
    n_res=8,                     # needs M >= n + (servers used by one DAG); spec default 8
    pbft_f=1,                    # n = 3f + 1 validators, quorum q = 2f + 1
    c_chk=2e6,                   # result-check cycles per verified subtask
    sig_scheme="dilithium2",
    crypto="real",               # "real": real ML-DSA / ML-KEM operations; "model": costs only
    host_ghz=3.0,                # converts measured host seconds into CPU cycles
    p_forge=0.5,                 # a malicious executor whose tampered result is rejected tries
                                 # to forge a quorum certificate with this probability
    quantum_adversary=False,     # attacker can run Shor's algorithm (breaks ECDSA keys)
    p_mitm=0.0,                  # probability a result is altered on the backhaul in transit
    s_hdr=200, s_res=1024,       # PBFT header and result-proof size (bytes)
    r_bh=1e9,                    # RES-RES backhaul rate (bit/s) for consensus messages
    rtt_range=(0.002, 0.010),    # RES-RES round-trip time, grows with distance (s)
    load_horizon=1.0,            # H in u_post (s)
    cpu_floor=0.05,              # epsilon in tau_k
    q_max=5e9,                   # RES queue cap for the offloading mask (cycles)
    t_max_blk=1.0,               # consensus-delay limit (s)
    dead_range=(2.0, 6.0),       # DAG deadline drawn per DAG (s)
    lam_dl=1.0, lam_blk=0.5,     # deadline / consensus-limit penalties
    cons_scale=1.0,              # multiplies RTT and c_chk (hidden-fraction experiment)
    # CoSched-GNN network and PPO
    co_hidden=128, co_layers=3, co_heads=4,
    freq_levels=(0.5e9, 1.0e9, 1.5e9, 2.0e9, 2.5e9),
    ppo_rollout=512, ppo_epochs=4, ppo_minibatch=64, ppo_clip=0.2,
    # gamma = 0: the Lyapunov drift-plus-penalty decouples the problem into independent
    # per-slot problems (TaBRL, P2), so each DAG's reward is credited to its own decision.
    # The spec's 0.99 / 0.95 learned far slower in our tests; set them with --set if needed.
    ppo_gamma=0.0, ppo_lambda=0.0, ppo_lr=3e-4,
    ppo_ent=(0.01, 0.001), ppo_vf=0.5, ppo_grad_clip=0.5,
    reward_norm=5.0,
    bc_epochs=0,                 # optional warm start: epochs imitating CP-Greedy before PPO
)


def co_cfg(base=None, **kw):
    c = dict(base if base is not None else CFG)
    c.update(CO_CFG)
    c.update(kw)
    return c


# %% [markdown]
# ## 1. DAG timing, slack and rank (section 4.5 and 5.3)

# %%
def children_of(preds):
    ch = [[] for _ in preds]
    for v, ps in enumerate(preds):
        for p in ps:
            ch[p].append(v)
    return ch


def rank_slack(preds, dur):
    """Upward rank (HEFT), earliest/latest finish and slack of every node, from the expected
    durations `dur` (consensus included for edge options)."""
    n = len(preds)
    order = topo_order(preds)
    ch = children_of(preds)
    ef = np.zeros(n)
    for v in order:
        ef[v] = max((ef[p] for p in preds[v]), default=0.0) + dur[v]
    mk = float(ef.max()) if n else 0.0
    lf = np.full(n, mk)
    for v in reversed(order):
        if ch[v]:
            lf[v] = min(lf[c] - dur[c] for c in ch[v])
    rank = np.zeros(n)
    for v in reversed(order):
        rank[v] = dur[v] + max((rank[c] for c in ch[v]), default=0.0)
    return rank, lf - ef, ef, lf, mk


def static_schedule(preds, place, t_loc, t_edge, t_cons):
    """Reference DAG timing used by the unit tests and the Gantt figure.
    place[v] = 0 (vehicle) or 1 (edge). One vehicle CPU runs local subtasks non-preemptively,
    highest upward rank first; edge subtasks run in parallel. A child is ready only when every
    parent is done and, if the parent was offloaded, committed (safety invariant).
    Returns makespan, (start, fin, done) per subtask and the exposed consensus delay."""
    n = len(preds)
    dur = [t_loc[v] if place[v] == 0 else t_edge[v] + t_cons[v] for v in range(n)]
    rank = rank_slack(preds, dur)[0]
    start, fin, done = [None] * n, [None] * n, [None] * n
    cpu, left = 0.0, set(range(n))
    while left:
        prog = False
        for v in sorted(left, key=lambda z: -rank[z]):
            if place[v] == 1 and all(done[p] is not None for p in preds[v]):
                start[v] = max((done[p] for p in preds[v]), default=0.0)
                fin[v] = start[v] + t_edge[v]; done[v] = fin[v] + t_cons[v]
                left.discard(v); prog = True
        loc = [v for v in left if place[v] == 0 and all(done[p] is not None for p in preds[v])]
        if loc:
            ready = {v: max((done[p] for p in preds[v]), default=0.0) for v in loc}
            t0 = max(cpu, min(ready.values()))
            cand = [v for v in loc if ready[v] <= t0 + 1e-12]
            v = max(cand, key=lambda z: rank[z])
            start[v] = t0; fin[v] = done[v] = t0 + t_loc[v]; cpu = fin[v]
            left.discard(v); prog = True
        if not prog:
            raise ValueError("cycle in DAG")
    mk = max(done)
    return mk, start, fin, done


def exposed_delay(preds, place, t_loc, t_edge, t_cons, v):
    """How much the makespan grows because of subtask v's consensus delay."""
    tc = list(t_cons)
    full = static_schedule(preds, place, t_loc, t_edge, tc)[0]
    tc[v] = 0.0
    return full - static_schedule(preds, place, t_loc, t_edge, tc)[0]


# %% [markdown]
# ## 2. PBFT consensus model (section 4.4)

# %%
# Used only with crypto="model" when no PQC library is installed (approximate values).
_FALLBACK_COSTS = {"ecdsa": (72, 4e-5, 1.1e-4), "dilithium2": (2420, 7e-5, 2.2e-5),
                   "dilithium3": (3309, 1e-4, 3.4e-5), "dilithium5": (4627, 1.2e-4, 4.9e-5)}


def pq_costs(cfg):
    """Signature size and sign / verify time of the configured scheme, measured on this
    machine with the real library (fallback table only in model mode without a library)."""
    sch = cfg["sig_scheme"]
    try:
        return PQ.measure(sch)
    except RuntimeError:
        if cfg["crypto"] == "real":
            raise
        b, ts, tv = _FALLBACK_COSTS[sch]
        return dict(sign_s=ts, verify_s=tv, sig_bytes=b, backend="none (fallback table)",
                    alg=PQ.SCHEMES[sch][0], level=PQ.SCHEMES[sch][2])


def c_cons_cycles(cfg):
    """CPU cycles one validator spends per verified subtask: result check, verify the
    executor's signature, sign its prepare and commit votes, verify the other n-1 validators'
    prepare and commit votes."""
    n = 3 * cfg["pbft_f"] + 1
    pc = pq_costs(cfg)
    hz = cfg["host_ghz"] * 1e9
    return (cfg["c_chk"] * cfg["cons_scale"] + 2 * pc["sign_s"] * hz
            + (1 + 2 * (n - 1)) * pc["verify_s"] * hz)


def t_msg(cfg, rtt, C):
    """One PBFT phase: half the largest RTT inside the committee + (n - 1) messages."""
    n = len(C)
    s_msg = (cfg["s_hdr"] + pq_costs(cfg)["sig_bytes"]) * 8
    worst = max((rtt[a, b] for a in C for b in C), default=0.0)
    return 0.5 * worst + (n - 1) * s_msg / cfg["r_bh"]


def t_cons_formula(cfg, rtt, F, u, C):
    """Closed-form T_cons(v, C) = S_res/R_bh + 3 T_msg(C) + tau_(q) used for features and the
    heuristics. The simulator itself runs verification as CPU jobs (see CoEnv)."""
    q = 2 * cfg["pbft_f"] + 1
    cc = c_cons_cycles(cfg)
    tau = sorted(cc / (F[k] * max(cfg["cpu_floor"], 1.0 - u[k])) for k in C)
    return cfg["s_res"] * 8 / cfg["r_bh"] + 3 * t_msg(cfg, rtt, C) + tau[q - 1]


class _ModelSig(bytes):
    """Signature placeholder in model mode: carries whether it would verify."""
    def __new__(cls, valid):
        o = super().__new__(cls, b"model")
        o.model_valid = valid
        return o


class VJob:
    """Verification job of one validator for one offloaded subtask; it sits in the
    validator's MFQ like any subtask, so validation and computation share the CPU."""
    __slots__ = ("rnd", "k", "accept", "rem", "b", "level", "avail_at", "done_at", "preds")

    def __init__(self, rnd, k, accept, cycles, avail_at):
        self.rnd, self.k, self.accept = rnd, k, accept
        self.rem = self.b = cycles
        self.level, self.avail_at, self.done_at, self.preds = 0, avail_at, None, []

    def ready(self, t):
        return True


class Round:
    """One PBFT instance: the result of subtask `s` executed on RES `ex`. `msg`/`sig` are the
    executor's signed result as received (possibly altered in transit), `expected` is the
    digest a correct execution gives, `cert` collects the signed accept votes."""
    __slots__ = ("s", "ex", "tampered", "t0", "C", "tmsg", "acc", "rej", "votes", "jobs", "decided",
                 "msg", "sig", "digest", "expected", "altered", "cert", "sig_bad")

    def __init__(self, s, ex, tampered, t0, C, tmsg):
        self.s, self.ex, self.tampered, self.t0, self.C, self.tmsg = s, ex, tampered, t0, C, tmsg
        self.acc = self.rej = self.votes = self.jobs = 0
        self.decided = self.altered = self.sig_bad = False
        self.msg = self.sig = self.digest = self.expected = None
        self.cert = []


# %% [markdown]
# ## 3. Environment with committee-based consensus

# %%
class CoRES(RES):
    """RES of tabrl_iov with one fix: a job demoted when its MFQ slice ends goes back to the
    queues only after every core has finished the current step, so another core cannot run
    the same job in the same step (it used to run on two cores at once)."""

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
        demoted = []
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
                elif core[1] <= 1e-9:          # slice used up -> demote, requeue after this step
                    s.level = min(s.level + 1, 2)
                    demoted.append(s); core[0] = None
        for s in demoted:
            self.q[s.level].append(s)


class CoEnv(IoVEnv):
    """TaBRL environment with the fixed D_blk/R_blk confirmation replaced by a per-DAG PBFT
    committee. Agents that do not choose validators use a random committee that is fixed for
    the whole episode (spec section 10.1)."""

    def __init__(self, cfg, n_veh, seed):
        super().__init__(cfg, n_veh, seed)
        for r in self.res:
            r.__class__ = CoRES
        self.n_cmt, self.q = 3 * cfg["pbft_f"] + 1, 2 * cfg["pbft_f"] + 1
        if self.M < self.n_cmt:
            raise ValueError(f"need at least n = {self.n_cmt} edge servers")
        lo, hi = cfg["rtt_range"]
        x = self.rsu[:, 0]
        d = np.abs(x[:, None] - x[None, :]) / max(cfg["road_len"], 1.0)
        self.rtt = (lo + (hi - lo) * d) * cfg["cons_scale"]
        np.fill_diagonal(self.rtt, 0.0)
        self.Fk = np.array([r.F for r in self.res])
        self.cores = np.array([len(r.cores) for r in self.res], float)
        self.fixed_committee = sorted(self.rng.choice(self.M, self.n_cmt, replace=False).tolist())
        self.cc = c_cons_cycles(cfg)
        self.ver_slot = np.zeros(self.M)       # verification cycles committed this slot
        self.served = np.zeros(self.M)         # committees each RES serves this slot
        self.n_off = self.n_selfval = self.tamper_committed = self.liveness_fail = 0
        # ---- post-quantum security layer ----
        self.real = cfg["crypto"] == "real"
        self.pc = pq_costs(cfg)
        self.post_quantum = cfg["sig_scheme"] != "ecdsa"
        self.forge_attempts = self.forge_accepted = self.mitm_detected = 0
        self.n_sign = self.n_verify = self.n_kem = 0
        self.sessions, self.uid = {}, 0
        self.kem_ct = 1088
        if self.real:
            self.signer = PQ.signer(cfg["sig_scheme"])
            self.sig_keys = [self.signer.keygen() for _ in range(self.M)]      # (pk, sk) per RES
            self.kem = PQ.KEM()
            self.kem_ct = self.kem.ct_bytes
            self.kem_keys = [self.kem.keygen() for _ in range(self.M)]

    # ---- cryptographic primitives (real, or their outcome in model mode) ----
    def _sign(self, j, msg):
        self.n_sign += 1
        return self.signer.sign(self.sig_keys[j][1], msg) if self.real else b"model"

    def _verify(self, j, msg, sig, valid_in_model=True):
        self.n_verify += 1
        return self.signer.verify(self.sig_keys[j][0], msg, sig) if self.real else valid_in_model

    def _session(self, vid, j):
        """ML-KEM-768 handshake between vehicle vid and RES j (once per pair and episode);
        returns True the first time, so the handshake ciphertext is added to the upload."""
        if (vid, j) in self.sessions:
            return False
        self.n_kem += 1
        if self.real:
            pk, sk = self.kem_keys[j]
            ct, ss_vehicle = self.kem.encaps(pk)
            ss_res = self.kem.decaps(sk, ct)
            assert ss_res == ss_vehicle
            self.sessions[(vid, j)] = ss_vehicle
        else:
            self.sessions[(vid, j)] = b"model"
        return True

    # ---- load views ----
    def backlog(self):
        return np.array([r.backlog_cycles() for r in self.res])

    def capacity(self):
        return self.Fk * self.cores * self.cfg["load_horizon"]

    def u_pre(self):
        return np.minimum(1.0, (self.backlog() + self.ver_slot) / self.capacity())

    def u_post(self, new_cycles):
        return np.minimum(1.0, (self.backlog() + new_cycles + self.ver_slot) / self.capacity())

    def tcons_est(self, C, u):
        return t_cons_formula(self.cfg, self.rtt, self.Fk, u, C)

    def begin_slot(self):
        super().begin_slot()
        self.ver_slot[:] = 0.0; self.served[:] = 0.0

    # ---- actions ----
    def submit(self, task, x, y, f, committee=None):
        c, v = self.cfg, self.veh[task.vid]
        f = float(np.clip(f, c["f_local_min"], c["f_local_max"]))
        near, _ = self.nearest(v)
        rate = self.rate(v)
        C = sorted(int(k) for k in (committee if committee is not None else self.fixed_committee))
        assert len(C) == self.n_cmt and len(set(C)) == self.n_cmt
        new = np.zeros(self.M)
        for k, s in enumerate(task.subs):
            if x[k]:
                new[int(y[k])] += s.b
        u_post = self.u_post(new)
        task.info.update(Q=v.Q, E=v.E, trust=[r.trust() for r in self.res], x=list(x), y=list(y),
                         committee=C, u_val=float(np.mean(u_post[C])), cd={}, tc={}, ex={})
        task.committee = C
        self.uid += 1
        task.info["uid"] = self.uid
        n_off = 0
        for k, s in enumerate(task.subs):
            s.x, s.f = int(x[k]), f
            if s.x:
                j = int(y[k]); s.res = j; n_off += 1
                self.n_off += 1; self.n_selfval += int(j in C)
                st = max(self.t, v.tx_free)
                dur = c["overhead"] * s.d / rate
                v.tx_free = st + dur
                v.tx_bits.append((st, st + dur, s.d))
                e = c["p_tx"] * dur
                v.slot_energy += e; task.energy += e
                hop = abs(near - j) * (c["backhaul_hop_s"] + s.d / c["backhaul_bps"])
                hs = self.kem_ct * 8 / rate if self._session(task.vid, j) else 0.0   # KEM handshake
                if self.real:                     # task descriptor sent encrypted (AES-256-GCM)
                    key = self.sessions[(task.vid, j)]
                    desc = PQ.sha3(self.uid, k, s.b, s.d)
                    assert self.kem.open(key, self.kem.seal(key, desc, b"upload"), b"upload") == desc
                s.avail_at = st + dur + hs + hop      # no pre-execution confirmation any more
                self.res[j].incoming.append(s)
            else:
                v.local.append(s)
        self.ver_slot[C] += self.cc * n_off
        self.served[C] += 1
        self.tasks.append(task)

    # ---- consensus ----
    def res_finished(self, r, s, t):
        if isinstance(s, VJob):
            return self._vote(s, t)
        task = s.task
        task.info["cd"][s.k] = t
        tampered = bool(r.mal and self.rng.random() < self.cfg["p_tamper"])
        C = task.committee
        tm = t_msg(self.cfg, self.rtt, C)
        rnd = Round(s, r.j, tampered, t, C, tm)
        uid = task.info["uid"]
        rnd.expected = PQ.sha3(uid, s.k, "result")                 # what a correct run returns
        rnd.digest = os.urandom(32) if tampered else rnd.expected
        rnd.msg = PQ.sha3(b"RESULT", uid, s.k, rnd.digest, r.j)
        rnd.sig = self._sign(r.j, rnd.msg)                          # executor signs its result
        if self.rng.random() < self.cfg["p_mitm"]:                  # altered on the backhaul
            rnd.altered = True
            rnd.digest = os.urandom(32)
        arrive = t + self.cfg["s_res"] * 8 / self.cfg["r_bh"] + tm  # pre-prepare
        for k in C:
            if self.res[k].mal:
                if not tampered:
                    continue                          # Byzantine validator withholds its vote
                accept = True                         # ... or backs a tampered result
            else:
                accept = None                         # decided by real verification at vote time
            rnd.jobs += 1
            self.res[k].incoming.append(VJob(rnd, k, accept, self.cc, arrive))
        if rnd.jobs == 0:
            self._liveness(rnd, t)

    def _vote(self, job, t):
        rnd = job.rnd
        rnd.votes += 1
        if rnd.decided:
            return
        accept = job.accept
        if accept is None:                            # honest validator: check signature + result
            received = PQ.sha3(b"RESULT", rnd.s.task.info["uid"], rnd.s.k, rnd.digest, rnd.ex)
            sig_ok = self._verify(rnd.ex, received, rnd.sig, valid_in_model=not rnd.altered)
            rnd.sig_bad |= not sig_ok
            accept = sig_ok and rnd.digest == rnd.expected
        vote = PQ.sha3(b"VOTE", rnd.s.task.info["uid"], rnd.s.k, rnd.digest, job.k, accept)
        vsig = self._sign(job.k, vote)                # every vote is signed
        rnd.acc += accept; rnd.rej += not accept
        if accept:
            rnd.cert.append((job.k, vote, vsig))
        if rnd.acc >= self.q:                         # prepare + commit phases
            self._commit(rnd, t, rnd.cert[:self.q])
        elif rnd.rej >= self.n_cmt - self.q + 1:     # accept quorum impossible -> reject
            rnd.decided = True
            if rnd.sig_bad:
                self.mitm_detected += 1
            if rnd.tampered:
                self.res[rnd.ex].fail += 1
                if self.rng.random() < self.cfg["p_forge"] and self._forge(rnd, t):
                    return
            self._reexecute(rnd.s, t + 2 * rnd.tmsg)
        elif rnd.votes == rnd.jobs:                   # votes withheld: view-change timeout
            self._liveness(rnd, t)

    def _certificate_ok(self, rnd, cert):
        """The consumer of a result (child executor or vehicle) checks the quorum certificate:
        q distinct committee members, each vote signed under that member's public key."""
        ids = [k for k, _, _ in cert]
        if len(set(ids)) < self.q or not set(ids) <= set(rnd.C):
            return False
        ok = True
        for k, vote, vsig in cert:
            ok &= self._verify(k, vote, vsig, valid_in_model=getattr(vsig, "model_valid", True))
        return ok

    def _commit(self, rnd, t, cert, forged=False):
        rnd.decided = True
        t_commit = t + 2 * rnd.tmsg + self.q * self.pc["verify_s"]
        if rnd.tampered:
            self.tamper_committed += 1
        elif not forged:
            self.res[rnd.ex].ok += 1
        rnd.s.task.info["tc"][rnd.s.k] = t_commit - rnd.t0
        self.sub_done(rnd.s, t_commit)

    def _forge(self, rnd, t):
        """A rejected malicious executor forges accept votes for the honest committee members.
        It needs their secret keys: a quantum adversary recovers ECDSA keys with Shor's
        algorithm, nobody can recover ML-DSA keys. Returns True if the forgery was accepted."""
        self.forge_attempts += 1
        attacker = rnd.ex
        cert = []
        for k in rnd.C:
            vote = PQ.sha3(b"VOTE", rnd.s.task.info["uid"], rnd.s.k, rnd.digest, k, True)
            colluder = self.res[k].mal                        # Byzantine members sign for real
            if self.real:
                if colluder:
                    sk = self.sig_keys[k][1]
                else:
                    sk = self.signer.recover_secret(self.sig_keys[k][0]) if self.cfg["quantum_adversary"] else None
                    if sk is None:
                        sk = self.sig_keys[attacker][1]       # best it can do: its own key
                self.n_sign += 1
                cert.append((k, vote, self.signer.sign(sk, vote)))
            else:
                broken = self.cfg["quantum_adversary"] and not self.post_quantum
                cert.append((k, vote, _ModelSig(broken or colluder)))
            if len(cert) == self.q:
                break
        if self._certificate_ok(rnd, cert):
            self.forge_accepted += 1
            self._commit(rnd, t, cert, forged=True)
            return True
        return False

    def _liveness(self, rnd, t):
        rnd.decided = True
        self.liveness_fail += 1
        self._reexecute(rnd.s, t + self.cfg["t_max_blk"])

    def _reexecute(self, s, t):
        v = self.veh[s.task.vid]
        s.x, s.rem, s.avail_at, s.f = 0, s.b, t, self.cfg["f_local_max"]
        v.local.appendleft(s)


# %% [markdown]
# ## 4. Episode loop, metrics and reward

# %%
def task_consensus_stats(task):
    """T_cons and exposed consensus delay of every committed offloaded subtask. The exposed
    part is how much later a child became ready (or the DAG finished) because of the
    consensus round, measured on the realised schedule."""
    subs, cd, tc = task.subs, task.info["cd"], task.info["tc"]
    ch = children_of([[p.k for p in s.preds] for s in subs])
    tot = exp = 0.0
    for k, tcons in tc.items():
        s = subs[k]
        if ch[k]:
            need = []
            for c in ch[k]:
                other = [p.done_at for p in subs[c].preds if p is not s]
                need.append(max(other + [subs[c].avail_at if subs[c].x else task.arr]))
            e = max(max(0.0, s.done_at - max(cd[k], n)) for n in need)
        else:
            other = [u.done_at for u in subs if u is not s]
            e = max(0.0, s.done_at - max([cd[k]] + other))
        tot += tcons; exp += min(e, tcons)
    task.info["ex"] = dict(total=tot, exposed=exp, n=len(tc))
    return tot, exp


def co_reward(task, cfg, n1_off=False):
    """TaBRL's P2 term (throughput, task-queue and energy-queue pressure) credited when the
    DAG finishes, minus deadline and consensus-limit penalties (section 7.6). With n1_off
    (ablation A1) the completion time is replaced by the sum over subtasks of their own time
    plus their full consensus delay, i.e. TaBRL's additive view."""
    T = task_delay(task)
    if n1_off:
        tc = task.info["tc"]
        own = 0.0
        for s in task.subs:
            rd = max([task.arr] + [p.done_at for p in s.preds])
            end = task.info["cd"].get(s.k, s.done_at)
            own += max(end - rd, 1e-3) + tc.get(s.k, 0.0)
        T = own
    local_bits = sum(s.d for s, x in zip(task.subs, task.info["x"]) if not x) / 1e6
    r = cfg["V"] * (task.size / 1e6) / max(T, 1e-3)
    r -= cfg["alpha_q"] * (task.info["Q"] / 1e7) * local_bits
    r -= cfg["alpha_e"] * (task.info["E"] / 10.0) * task.energy
    r -= cfg["lam_dl"] * float(task_delay(task) > task.deadline)
    r -= cfg["lam_blk"] * float(max(task.info["tc"].values(), default=0.0) > cfg["t_max_blk"])
    return float(np.clip(r / cfg["reward_norm"], -10, 10))


def run_co_episode(agent, sched, cfg, n_veh, seed, train, horizon=None, trace=False):
    env = CoEnv(cfg, n_veh, seed)
    agent.env = env
    horizon = horizon or len(sched)
    drng = np.random.default_rng(seed + 7)
    backlog, energy, rewards, objective = [], [], [], []
    stores = 0

    def handle(done):
        nonlocal stores
        for task in done:
            task_consensus_stats(task)
            objective.append(co_reward(task, cfg))       # same P2 objective for every method
            r = agent.reward(task); rewards.append(r)
            if train and task.act is not None:
                agent.store(task, r); stores += 1
                if stores % getattr(agent, "train_every", cfg["train_every"]) == 0:
                    agent.train()

    for s in range(horizon):
        env.begin_slot()
        for vid, job in sched[s]:
            task = Task(vid, env.t, job)
            task.deadline = float(drng.uniform(*cfg["dead_range"]))
            if hasattr(agent, "act_joint"):
                x, y, f, C, task.act = agent.act_joint(env, task, explore=train)
            else:
                build = env.build_graph_obs if getattr(agent, "graph_obs", False) else env.build_obs
                task.obs = build(vid, task, agent.obs_mode)
                x, y, f, task.act = agent.act(task.obs, task, explore=train)
                C = None
            env.submit(task, x, y, f, C)
        handle(env.run_slot())
        if trace:
            backlog.append(sum(v.Q for v in env.veh))
            energy.append(sum(v.total_energy for v in env.veh))
    extra = 0
    while not env.all_done() and extra < cfg["drain_max"]:
        env.begin_slot(); handle(env.run_slot()); extra += 1
    if train and hasattr(agent, "end_episode"):
        agent.end_episode()

    fin = [t for t in env.tasks if t.complete is not None]
    delays = [task_delay(t) for t in fin] + [env.t - t.arr for t in env.tasks if t.complete is None]
    makespan = max([t.complete for t in fin] + [env.t if len(fin) < len(env.tasks) else 0, 1.0])
    tot = sum(t.info["ex"]["total"] for t in fin if "ex" in t.info and t.info["ex"])
    exp = sum(t.info["ex"]["exposed"] for t in fin if "ex" in t.info and t.info["ex"])
    ncons = sum(t.info["ex"]["n"] for t in fin if "ex" in t.info and t.info["ex"])
    return dict(
        completion=float(np.mean(delays)) if delays else 0.0,
        throughput=float(np.mean([t.size / task_delay(t) for t in fin])) if fin else 0.0,
        sys_throughput=float(sum(t.size for t in fin) / makespan),
        reward=float(np.mean(rewards)) if rewards else 0.0,
        objective=float(np.mean(objective)) if objective else 0.0,
        energy=float(sum(v.total_energy for v in env.veh)),
        offload=float(np.mean([np.mean(t.info["x"]) for t in env.tasks])) if env.tasks else 0.0,
        tampered=int(sum(r.fail for r in env.res)),                 # caught by the committee
        tamper_committed=int(env.tamper_committed),                  # accepted although wrong
        deadline_miss=float(np.mean([task_delay(t) > t.deadline for t in fin])) if fin else 0.0,
        cons_ms=1e3 * tot / max(ncons, 1),
        hidden=float(1.0 - exp / tot) if tot > 0 else 1.0,
        self_val=float(env.n_selfval / max(env.n_off, 1)),
        val_util=float(np.mean([t.info["u_val"] for t in env.tasks])) if env.tasks else 0.0,
        liveness=int(env.liveness_fail),
        forge_attempts=int(env.forge_attempts), forge_accepted=int(env.forge_accepted),
        mitm_detected=int(env.mitm_detected), sig_ops=int(env.n_sign + env.n_verify),
        kem_handshakes=int(env.n_kem),
        n_tasks=len(env.tasks), backlog=backlog, energy_t=energy)


METRICS = ("forge_attempts", "forge_accepted", "mitm_detected", "sig_ops", "kem_handshakes",
           "completion", "throughput", "sys_throughput", "reward", "objective", "energy", "offload", "tampered",
           "tamper_committed", "deadline_miss", "cons_ms", "hidden", "self_val", "val_util", "liveness")


def co_evaluate(agent, windows, cfg, n_veh, horizon=None, trace=False):
    ms = [run_co_episode(agent, sched, cfg, n_veh, seed=seed, train=False, horizon=horizon, trace=trace)
          for seed, sched in windows]
    out = {k: float(np.mean([m[k] for m in ms])) for k in METRICS}
    out["completion_std"] = float(np.std([m["completion"] for m in ms]))
    out["throughput_std"] = float(np.std([m["throughput"] for m in ms]))
    if trace:
        out["backlog"] = np.mean([m["backlog"] for m in ms], 0).tolist()
        out["energy_t"] = np.mean([m["energy_t"] for m in ms], 0).tolist()
    return out


# %% [markdown]
# ## 5. Graph built at every DAG arrival (section 7.1)

# %%
FT_C, FS_C, FV_C = 9, 9, 6          # subtask, RES and vehicle node features
FE_OPT = 5                          # option edge features in the offloading head (3 static + 2 EFT)
FV_VAL = 4                          # extra validator-head features per RES


def lowest(values, n, allowed=None):
    idx = [k for k in np.argsort(values, kind="stable") if allowed is None or allowed[k]]
    return sorted(int(k) for k in idx[:n])


def co_graph(env, task, n1_off=False):
    """Heterogeneous graph of one DAG arrival: subtask nodes, RES nodes, the vehicle node and
    the depends_on / can_run_on / local_on / backhaul edges, plus the numbers the heads use."""
    c, v = env.cfg, env.veh[task.vid]
    K, M = len(task.subs), env.M
    near, _ = env.nearest(v)
    rate = max(env.rate(v), 1.0)
    preds = [[p.k for p in s.preds] for s in task.subs]
    b = np.array([s.b for s in task.subs]); d = np.array([s.d for s in task.subs])
    F, cores = env.Fk, env.cores
    backlog = env.backlog()
    u_pre = env.u_pre()
    hop = np.abs(near - np.arange(M))[None, :] * (c["backhaul_hop_s"] + d[:, None] / c["backhaul_bps"])
    t_up = c["overhead"] * d[:, None] / rate + hop                         # [K, M]
    t_loc = b / c["f_local_max"]
    t_edge = t_up + b[:, None] / F[None, :]
    tcons = env.tcons_est(lowest(u_pre, env.n_cmt), u_pre)
    dur = 0.5 * t_loc + 0.5 * (t_edge.mean(1) + tcons)
    rank, slack, _, _, _ = rank_slack(preds, dur)
    depth = np.zeros(K)
    for k in topo_order(preds):
        depth[k] = 1 + max((depth[p] for p in preds[k]), default=-1)
    ch = children_of(preds)
    xt = np.stack([b / 1e9, d / 1e6, rank / 5.0, slack / 5.0, depth / 8.0,
                   np.array([float(not ch[k]) for k in range(K)]),
                   np.full(K, task.deadline / 6.0),
                   np.array([len(p) for p in preds]) / 4.0,
                   np.array([len(x) for x in ch]) / 4.0], 1).astype(np.float32)
    if n1_off:                                    # ablation A1: no critical-path features
        xt[:, 2:4] = 0.0
    A = np.zeros((K, K), np.float32); Ed = np.zeros((K, K, 1), np.float32)
    for k, ps in enumerate(preds):
        for p in ps:
            A[k, p] = 1.0; Ed[k, p, 0] = d[p] / 1e6
    xs = np.stack([F / 5e9, backlog / (F * cores) / 10.0,
                   [len(r.q[0]) / 10.0 for r in env.res], [len(r.q[1]) / 10.0 for r in env.res],
                   [len(r.q[2]) / 10.0 for r in env.res], u_pre, env.served / 5.0,
                   env.ver_slot / (F * cores),
                   [r.trust() for r in env.res]],      # on-chain verification record (TaBRL)
                  1).astype(np.float32)
    zone = c["road_len"] / M
    edge = (near + (1 if v.v > 0 else 0)) * zone
    dwell = abs(edge - v.x) / max(abs(v.v), 1e-3)
    xv = np.array([c["f_local_max"] / 2.5e9, v.Q / 1e7, v.E / 10.0,
                   sum(s.rem for s in v.local) / c["f_local_max"] / 10.0, abs(v.v) / 25.0,
                   min(dwell, 60.0) / 10.0], np.float32)
    Ers = np.stack([t_up, hop / 0.1], -1).astype(np.float32)
    Etv = (t_loc / 2.0)[:, None].astype(np.float32)
    Ess = np.stack([env.rtt / 0.01, np.ones((M, M))], -1).astype(np.float32)
    # option edge features for the offloading head: vehicle first, then RES 0..M-1
    opt = np.zeros((K, M + 1, 3), np.float32)
    opt[:, 0, 0] = t_loc / 2.0; opt[:, 0, 2] = 1.0
    opt[:, 1:, 0] = t_up; opt[:, 1:, 1] = b[:, None] / F[None, :]
    return dict(xt=xt, A=A, Ed=Ed, xs=xs, xv=xv, Ers=Ers, Etv=Etv, Ess=Ess, opt=opt,
                order=np.argsort(-rank, kind="stable"), b=b, backlog=backlog, cap=env.capacity(),
                qmask=(backlog > c["q_max"]), u_pre=u_pre, rank=rank, slack=slack, t_loc=t_loc,
                t_up=t_up, preds=preds)


def batch_cographs(gs):
    B = len(gs)
    K = max(g["xt"].shape[0] for g in gs); M = gs[0]["xs"].shape[0]
    z = lambda *s: np.zeros(s, np.float32)
    out = dict(xt=z(B, K, FT_C), mt=z(B, K), A=z(B, K, K), Ed=z(B, K, K, 1), xs=z(B, M, FS_C),
               ms=np.ones((B, M), np.float32), xv=z(B, FV_C), Ers=z(B, K, M, 2), Etv=z(B, K, 1),
               Ess=z(B, M, M, 2))
    for i, g in enumerate(gs):
        k = g["xt"].shape[0]
        out["xt"][i, :k] = g["xt"]; out["mt"][i, :k] = 1; out["A"][i, :k, :k] = g["A"]
        out["Ed"][i, :k, :k] = g["Ed"]; out["xs"][i] = g["xs"]; out["xv"][i] = g["xv"]
        out["Ers"][i, :k] = g["Ers"]; out["Etv"][i, :k] = g["Etv"]; out["Ess"][i] = g["Ess"]
    return {k: torch.as_tensor(v, device=DEVICE) for k, v in out.items()}


# %% [markdown]
# ## 6. Heterogeneous graph-attention encoder, heads and critic (section 7.2 - 7.5)

# %%
class GATv2Rel(nn.Module):
    """Dense GATv2 attention for one relation (edge type) with edge features."""
    def __init__(self, d, heads, fe):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ws, self.wt, self.we = nn.Linear(d, d), nn.Linear(d, d, bias=False), nn.Linear(fe, d, bias=False)
        self.att = nn.Parameter(torch.randn(heads, self.dh) * 0.1)

    def forward(self, ht, hs, e, mask):
        B, Nt, Ns = mask.shape
        zs = self.ws(hs).view(B, 1, Ns, self.h, self.dh)
        zt = self.wt(ht).view(B, Nt, 1, self.h, self.dh)
        ze = self.we(e).view(B, Nt, Ns, self.h, self.dh)
        msg = zs + ze
        sc = (F.leaky_relu(msg + zt, 0.2) * self.att).sum(-1)
        sc = sc.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        a = torch.softmax(sc, 2) * mask.unsqueeze(-1)
        return (a.unsqueeze(-1) * msg).sum(2).reshape(B, Nt, -1)


class HeteroLayer(nn.Module):
    REL = {"dep": ("t", "t", 1), "rdep": ("t", "t", 1), "run": ("t", "s", 2), "run_r": ("s", "t", 2),
           "loc": ("t", "v", 1), "loc_r": ("v", "t", 1), "bh": ("s", "s", 2)}

    def __init__(self, d, heads):
        super().__init__()
        self.rel = nn.ModuleDict({k: GATv2Rel(d, heads, fe) for k, (_, _, fe) in self.REL.items()})
        self.norm = nn.ModuleDict({k: nn.LayerNorm(d) for k in "tsv"})

    def forward(self, h, g):
        mt, ms = g["mt"], g["ms"]
        B, K = mt.shape; M = ms.shape[1]
        mts = mt.unsqueeze(2) * ms.unsqueeze(1)
        mss = ms.unsqueeze(2) * ms.unsqueeze(1) * (1 - torch.eye(M, device=ms.device))
        upd = dict(
            t=self.rel["dep"](h["t"], h["t"], g["Ed"], g["A"])
            + self.rel["rdep"](h["t"], h["t"], g["Ed"].transpose(1, 2), g["A"].transpose(1, 2))
            + self.rel["run"](h["t"], h["s"], g["Ers"], mts)
            + self.rel["loc"](h["t"], h["v"], g["Etv"].unsqueeze(-1), mt.unsqueeze(2)),
            s=self.rel["run_r"](h["s"], h["t"], g["Ers"].transpose(1, 2), mts.transpose(1, 2))
            + self.rel["bh"](h["s"], h["s"], g["Ess"], mss),
            v=self.rel["loc_r"](h["v"], h["t"], g["Etv"].transpose(1, 2).unsqueeze(-1), mt.unsqueeze(1)))
        out = {k: self.norm[k](h[k] + F.relu(upd[k])) for k in "tsv"}
        out["t"] = out["t"] * mt.unsqueeze(-1); out["s"] = out["s"] * ms.unsqueeze(-1)
        return out


class CoNet(nn.Module):
    """Encoder (heterogeneous GNN, or a flat MLP for MLP-Joint / ablation A5), offloading head,
    validator head, CPU-frequency head and critic."""
    def __init__(self, cfg, encoder="gnn"):
        super().__init__()
        d, L, H = cfg["co_hidden"], cfg["co_layers"], cfg["co_heads"]
        self.encoder, self.d = encoder, d
        self.inp = nn.ModuleDict(dict(t=nn.Linear(FT_C, d), s=nn.Linear(FS_C, d), v=nn.Linear(FV_C, d)))
        if encoder == "gnn":
            self.layers = nn.ModuleList([HeteroLayer(d, H) for _ in range(L)])
        else:
            self.k_pad, self.m_fix = cfg["k_max"], cfg["n_res"]
            flat = self.k_pad * FT_C + self.m_fix * FS_C + FV_C
            self.ctx = nn.Sequential(nn.Linear(flat, 2 * d), nn.ReLU(), nn.Linear(2 * d, d), nn.ReLU())
            self.mix = nn.ModuleDict({k: nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
                                      for k in "tsv"})
        g_dim = 5 * d
        self.off = nn.Sequential(nn.Linear(2 * d + FE_OPT + 2, d), nn.ReLU(), nn.Linear(d, 1))
        self.val = nn.Sequential(nn.Linear(d + FV_VAL + g_dim, d), nn.ReLU(), nn.Linear(d, 1))
        self.freq = nn.Sequential(nn.Linear(g_dim, d), nn.ReLU(), nn.Linear(d, len(cfg["freq_levels"])))
        self.critic = nn.Sequential(nn.Linear(g_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def encode(self, g):
        h = dict(t=F.relu(self.inp["t"](g["xt"])) * g["mt"].unsqueeze(-1),
                 s=F.relu(self.inp["s"](g["xs"])), v=F.relu(self.inp["v"](g["xv"])).unsqueeze(1))
        if self.encoder == "gnn":
            for layer in self.layers:
                h = layer(h, g)
        else:
            B, K = g["mt"].shape
            xt = F.pad(g["xt"], (0, 0, 0, self.k_pad - K))
            flat = torch.cat([xt.reshape(B, -1), g["xs"].reshape(B, -1), g["xv"]], -1)
            ctx = self.ctx(flat)
            h = {k: self.mix[k](torch.cat([h[k], ctx.unsqueeze(1).expand_as(h[k])], -1)) for k in "tsv"}
            h["t"] = h["t"] * g["mt"].unsqueeze(-1)
        gl = torch.cat([masked_mean(h["t"], g["mt"]), masked_max(h["t"], g["mt"]),
                        masked_mean(h["s"], g["ms"]), masked_max(h["s"], g["ms"]), h["v"][:, 0]], -1)
        return h, gl

    def off_logits(self, hv, ho, e, dl, used):
        """hv [..., d] subtask embedding, ho [..., O, d] option embeddings (vehicle + RESs)."""
        x = torch.cat([hv.unsqueeze(-2).expand_as(ho), ho, e, dl.unsqueeze(-1), used.unsqueeze(-1)], -1)
        return self.off(x).squeeze(-1)

    def val_logits(self, hs, vf, gl):
        return self.val(torch.cat([hs, vf, gl.unsqueeze(1).expand(-1, hs.shape[1], -1)], -1)).squeeze(-1)


def seq_logp(scores, mask, seq):
    """Exact log-probability of an ordered committee drawn by sequential softmax without
    replacement (section 6.3). scores/mask [B, M], seq [B, n]."""
    avail = mask.clone()
    lp = torch.zeros(scores.shape[0], device=scores.device)
    ent = None
    for m in range(seq.shape[1]):
        logit = scores.masked_fill(avail == 0, -1e9)
        ls = torch.log_softmax(logit, -1)
        if ent is None:
            ent = -(ls.exp() * ls * avail).sum(-1)
        lp = lp + ls.gather(1, seq[:, m:m + 1]).squeeze(1)
        avail = avail.scatter(1, seq[:, m:m + 1], 0.0)
    return lp, ent


# %% [markdown]
# ## 7. CoSched-GNN agent: masked joint action + PPO (section 6, 7.6 - 7.8)

# %%
class CoSchedGNN(BaseAgent):
    """One action per DAG: an offloading option for every subtask (decided in rank order,
    with load/used features updated after each choice), then a PBFT committee of n RESs,
    then the local CPU frequency. Variants (ablations / closest-work baselines):
      n1_off      A1  reward = sum of own time + full consensus, no rank/slack features
      val_feat    A2  'pre': the validator head sees load before offloading
      exec_mask   A3  False: no executor exclusion (and no committee-feasibility mask)
      val_mode    A4  'postload_rule': learned offloading, rule-based post-load committee
                      'preload_rule': HGAT-style (n lowest pre-offloading load, no mask)
                      'reliability_rule': FE-CTDE-style (n most reliable RESs, Feng et al. 2026)
      encoder     A5  'mlp': flat MLP instead of the graph encoder (MLP-Joint)"""
    name, obs_mode = "CoSched-GNN", "reported"

    def __init__(self, env, cfg, n1_off=False, val_feat="post", exec_mask=True, val_mode="learn",
                 encoder="gnn", name=None):
        BaseAgent.__init__(self, env, cfg)
        self.name = name or self.name
        self.n1_off, self.val_feat, self.exec_mask = n1_off, val_feat, exec_mask
        self.val_mode, self.encoder = val_mode, encoder
        self.net = CoNet(cfg, encoder).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["ppo_lr"])
        self.rng = np.random.default_rng(cfg["seed"] + 11)
        self.buf = []
        self.frac, self.train_every = 0.0, 1
        self.n_updates, self.epoch, self.expert = 0, 0, None

    def set_progress(self, frac):
        self.frac = frac
        for gr in self.opt.param_groups:
            gr["lr"] = self.cfg["ppo_lr"] * max(0.05, 1.0 - frac)
        self.eps = 0.0

    def in_warmup(self):
        return self.epoch < self.cfg["bc_epochs"]

    @torch.no_grad()
    def act_joint(self, env, task, explore):
        c = self.cfg
        G = co_graph(env, task, self.n1_off)
        forced = None
        if explore and self.in_warmup():          # warm start: imitate the CP-Greedy rule
            if self.expert is None:
                self.expert = CPGreedy(env, c)
            xe, ye = self.expert.plan(env, task)
            forced = dict(off={v: (int(ye[v]) + 1 if xe[v] else 0) for v in range(len(task.subs))},
                          C=self.expert.committee(env, task, xe, ye))
        K, M, n = len(task.subs), env.M, env.n_cmt
        gt = batch_cographs([G])
        h, gl = self.net.encode(gt)
        ho = torch.cat([h["v"], h["s"]], 1)[0]                                  # [1+M, d]
        cap = np.concatenate([[c["f_local_max"] * c["load_horizon"]], G["cap"]])
        assigned = np.zeros(M + 1); used = np.zeros(M + 1)
        limit = M - n if self.exec_mask else M
        steps_e, steps_dl, steps_used, steps_mask, acts = [], [], [], [], []
        logp = 0.0
        eft = EFT(env, task, G)
        for v in G["order"]:
            fin = eft.options(v)
            cons = eft.tcons if self.n1_off else max(0.0, eft.tcons - G["slack"][v])
            ev = np.concatenate([G["opt"][v], np.stack([(fin - env.t) / 2.0,
                                                        np.r_[0.0, np.full(M, cons / 0.5)]], 1)], 1).astype(np.float32)
            mask = np.ones(M + 1, np.float32)
            mask[1:][G["qmask"]] = 0.0
            if used[1:].sum() >= limit:
                mask[1:][used[1:] == 0] = 0.0
            dl = (assigned / cap).astype(np.float32)
            e = torch.as_tensor(ev, device=DEVICE)
            lg = self.net.off_logits(h["t"][0, v], ho, e, torch.as_tensor(dl, device=DEVICE),
                                     torch.as_tensor(used.astype(np.float32), device=DEVICE))
            lg = lg.masked_fill(torch.as_tensor(mask, device=DEVICE) == 0, -1e9)
            ls = torch.log_softmax(lg, -1)
            if forced is not None and mask[forced["off"][v]] > 0:
                a = forced["off"][v]
            else:
                a = int(torch.multinomial(ls.exp(), 1)) if explore else int(ls.argmax())
            logp += float(ls[a])
            steps_e.append(ev); steps_dl.append(dl); steps_used.append(used.astype(np.float32).copy())
            steps_mask.append(mask); acts.append(a)
            assigned[a] += G["b"][v]; used[a] = 1.0
            eft.commit(v, a, fin)
        x = np.zeros(K, int); y = np.zeros(K, int)
        for v, a in zip(G["order"], acts):
            x[v], y[v] = int(a > 0), max(a - 1, 0)
        new = np.zeros(M)
        for v in range(K):
            if x[v]:
                new[y[v]] += G["b"][v]
        executors = set(int(y[v]) for v in range(K) if x[v])
        u_post = env.u_post(new)
        vmask = np.ones(M, np.float32)
        if self.exec_mask:
            vmask[list(executors)] = 0.0
        u_feat = u_post if self.val_feat == "post" else G["u_pre"]
        mean_rtt = np.array([env.rtt[k][vmask > 0].mean() for k in range(M)])
        vf = np.stack([u_feat, env.Fk / 5e9, mean_rtt / 0.01, env.served / 5.0], 1).astype(np.float32)
        seq = None
        if self.val_mode == "learn":
            s = self.net.val_logits(h["s"], torch.as_tensor(vf, device=DEVICE)[None], gl)[0]
            avail = torch.tensor(vmask, device=DEVICE)          # copy: vmask is stored
            seq = []
            want = None
            if forced is not None and all(vmask[k] > 0 for k in forced["C"]):
                want = sorted(forced["C"], key=lambda k: u_post[k])
            for i in range(n):
                ls = torch.log_softmax(s.masked_fill(avail == 0, -1e9), -1)
                if want is not None:
                    k = want[i]
                else:
                    k = int(torch.multinomial(ls.exp(), 1)) if explore else int(ls.argmax())
                logp += float(ls[k]); seq.append(k); avail[k] = 0.0
            C = sorted(seq)
        elif self.val_mode == "postload_rule":
            C = lowest(u_post, n, vmask > 0)
        elif self.val_mode == "reliability_rule":                  # FE-CTDE-style
            # dynamic BFT committee of the n most reliable RESs (verification record),
            # ties broken by computing capability; load and the executors are not considered
            rel = np.array([r.trust() for r in env.res]) + 1e-3 * env.Fk / 5e9
            C = lowest(-rel, n)
        else:                                                      # preload_rule (HGAT-style)
            C = lowest(G["u_pre"], n)
        lf = torch.log_softmax(self.net.freq(gl)[0], -1)
        if forced is not None:
            fa = len(c["freq_levels"]) - 1                       # the rule runs at f_max
        else:
            fa = int(torch.multinomial(lf.exp(), 1)) if explore else int(lf.argmax())
        logp += float(lf[fa])
        value = float(self.net.critic(gl)[0, 0])
        rec = None
        if explore:
            G_store = {k: G[k] for k in ("xt", "A", "Ed", "xs", "xv", "Ers", "Etv", "Ess")}
            rec = dict(G=G_store, order=G["order"].copy(), e=np.stack(steps_e), dl=np.stack(steps_dl),
                       used=np.stack(steps_used), mask=np.stack(steps_mask), acts=np.array(acts),
                       vf=vf, vmask=vmask, seq=None if seq is None else np.array(seq), fa=fa,
                       logp=logp, value=value, r=None, term=False, bc=forced is not None)
            self.buf.append(rec)
        return x.tolist(), y.tolist(), c["freq_levels"][fa], C, rec

    def reward(self, task):
        return co_reward(task, self.cfg, self.n1_off)

    def store(self, task, r):
        task.act["r"] = r

    def end_episode(self):
        for rec in self.buf:
            if rec["r"] is None:                       # unfinished at the end of the drain
                rec["r"] = -self.cfg["lam_dl"]
        if self.buf:
            self.buf[-1]["term"] = True
        if self.buf and self.buf[0].get("bc"):
            self.bc_update(self.buf); self.buf = []
        self.train()

    def bc_update(self, recs):
        """Warm start (cfg bc_epochs > 0): maximise the log-probability of the rule's joint
        action and fit the critic to the observed rewards."""
        c = self.cfg
        rew = np.array([r["r"] for r in recs], np.float32)
        for _ in range(c["ppo_epochs"]):
            idx = np.random.permutation(len(recs))
            for s0 in range(0, len(recs), c["ppo_minibatch"]):
                mb = idx[s0:s0 + c["ppo_minibatch"]]
                logp, _, val = self.evaluate_batch([recs[i] for i in mb])
                vl = F.mse_loss(val, torch.as_tensor(rew[mb], device=DEVICE))
                loss = -logp.mean() + c["ppo_vf"] * vl
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c["ppo_grad_clip"])
                self.opt.step()

    def train(self):
        R = self.cfg["ppo_rollout"]
        if self.in_warmup():
            return
        if len(self.buf) < R or any(rec["r"] is None for rec in self.buf[:R]):
            return
        batch, self.buf = self.buf[:R], self.buf[R:]
        boot = self.buf[0]["value"] if self.buf and not batch[-1]["term"] else 0.0
        self.ppo_update(batch, boot)

    def evaluate_batch(self, recs):
        g = batch_cographs([r["G"] for r in recs])
        h, gl = self.net.encode(g)
        B, K = g["mt"].shape; M = g["ms"].shape[1]
        pad = lambda key, shape, dt=np.float32: np.stack(
            [np.pad(r[key], [(0, K - len(r[key]))] + [(0, 0)] * (len(shape) - 1)) for r in recs]).astype(dt)
        t = lambda z: torch.as_tensor(z, device=DEVICE)
        order = t(pad("order", (K,), np.int64)); acts = t(pad("acts", (K,), np.int64))
        sv = torch.zeros(B, K, device=DEVICE)
        for i, r in enumerate(recs):
            sv[i, :len(r["acts"])] = 1.0
        e = t(pad("e", (K, M + 1, FE_OPT))); dl = t(pad("dl", (K, M + 1)))
        used = t(pad("used", (K, M + 1))); mask = t(pad("mask", (K, M + 1)))
        mask = mask + (1 - sv).unsqueeze(-1)            # padded steps: anything allowed (ignored)
        hv = h["t"].gather(1, order.unsqueeze(-1).expand(-1, -1, self.net.d))
        ho = torch.cat([h["v"], h["s"]], 1).unsqueeze(1).expand(-1, K, -1, -1)
        lg = self.net.off_logits(hv, ho, e, dl, used).masked_fill(mask == 0, -1e9)
        ls = torch.log_softmax(lg, -1)
        logp = (ls.gather(2, acts.unsqueeze(-1)).squeeze(-1) * sv).sum(1)
        ent = (-(ls.exp() * ls).sum(-1) * sv).sum(1)
        if self.val_mode == "learn":
            vf = t(np.stack([r["vf"] for r in recs])); vmask = t(np.stack([r["vmask"] for r in recs]))
            seq = t(np.stack([r["seq"] for r in recs]).astype(np.int64))
            lpv, entv = seq_logp(self.net.val_logits(h["s"], vf, gl), vmask, seq)
            logp, ent = logp + lpv, ent + entv
        lf = torch.log_softmax(self.net.freq(gl), -1)
        fa = t(np.array([r["fa"] for r in recs], np.int64))
        logp = logp + lf.gather(1, fa.unsqueeze(1)).squeeze(1)
        ent = ent - (lf.exp() * lf).sum(-1)
        return logp, ent, self.net.critic(gl).squeeze(-1)

    def ppo_update(self, batch, boot):
        c = self.cfg
        R = len(batch)
        v = np.array([r["value"] for r in batch] + [boot])
        rew = np.array([r["r"] for r in batch]); term = np.array([r["term"] for r in batch], float)
        adv = np.zeros(R); last = 0.0
        for i in reversed(range(R)):
            nv = 0.0 if term[i] else v[i + 1]
            delta = rew[i] + c["ppo_gamma"] * nv - v[i]
            last = delta + c["ppo_gamma"] * c["ppo_lambda"] * (1 - term[i]) * last
            adv[i] = last
        ret = adv + v[:R]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old = np.array([r["logp"] for r in batch])
        e0, e1 = c["ppo_ent"]
        ent_coef = e0 + (e1 - e0) * min(1.0, self.frac)
        for _ in range(c["ppo_epochs"]):
            idx = np.random.permutation(R)
            for s0 in range(0, R, c["ppo_minibatch"]):
                mb = idx[s0:s0 + c["ppo_minibatch"]]
                logp, ent, val = self.evaluate_batch([batch[i] for i in mb])
                tt = lambda z: torch.as_tensor(z[mb], dtype=torch.float32, device=DEVICE)
                ratio = torch.exp(logp - tt(old))
                a = tt(adv)
                pl = -torch.min(ratio * a, ratio.clamp(1 - c["ppo_clip"], 1 + c["ppo_clip"]) * a).mean()
                vl = F.mse_loss(val, tt(ret))
                loss = pl + c["ppo_vf"] * vl - ent_coef * ent.mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c["ppo_grad_clip"])
                self.opt.step()
        self.n_updates += 1


# %% [markdown]
# ## 8. Heuristic baselines (sections 5.5 and 6.4)

# %%
class EFT:
    """Earliest-finish-time bookkeeping for one DAG decision (list scheduling in rank order).
    Used by the rule baselines to choose and by CoSched-GNN as option features, so the
    network sees at least the information the rules use."""

    def __init__(self, env, task, G):
        c = env.cfg
        self.env, self.task, self.G = env, task, G
        self.t0 = env.t
        self.tcons = env.tcons_est(lowest(G["u_pre"], env.n_cmt), G["u_pre"])
        v = env.veh[task.vid]
        self.cpu = env.t + sum(s.rem for s in v.local) / c["f_local_max"]
        self.res_free = env.t + G["backlog"] / (env.Fk * env.cores)
        self.done = np.zeros(len(task.subs))

    def options(self, k):
        """Estimated finish (before consensus) of subtask k on the vehicle and on every RES."""
        G, env = self.G, self.env
        ready = max([self.t0] + [self.done[p] for p in G["preds"][k]])
        loc = max(ready, self.cpu) + G["t_loc"][k]
        edge = np.maximum(np.maximum(ready, self.t0 + G["t_up"][k]), self.res_free) + G["b"][k] / env.Fk
        return np.concatenate([[loc], edge])

    def commit(self, k, a, fin):
        if a == 0:
            self.cpu = fin[0]; self.done[k] = fin[0]
        else:
            j = a - 1
            self.res_free[j] += self.G["b"][k] / (self.env.Fk[j] * self.env.cores[j])
            self.done[k] = fin[a] + self.tcons


class _Greedy(BaseAgent):
    """List scheduling in upward-rank order: each subtask goes to the option with the earliest
    estimated finish. HEFT counts the full consensus delay of an edge option; CP-Greedy counts
    only the part exceeding the subtask's slack (rule of section 5.2) and keeps enough
    non-executing RESs for the committee."""
    exposed_only = feasible = False

    def __init__(self, env, cfg):
        BaseAgent.__init__(self, env, cfg)
        self.rng = np.random.default_rng(cfg["seed"] + 13)

    def plan(self, env, task):
        G = co_graph(env, task)
        K, M = len(task.subs), env.M
        eft = EFT(env, task, G)
        x = np.zeros(K, int); y = np.zeros(K, int)
        used, limit = set(), (M - env.n_cmt if self.feasible else M)
        for k in G["order"]:
            fin = eft.options(k)
            extra = max(0.0, eft.tcons - G["slack"][k]) if self.exposed_only else eft.tcons
            best, b_cost = 0, fin[0]
            for j in range(M):
                if G["qmask"][j] or (j not in used and len(used) >= limit):
                    continue
                if fin[j + 1] + extra < b_cost:
                    best, b_cost = j + 1, fin[j + 1] + extra
            if best:
                x[k], y[k] = 1, best - 1; used.add(best - 1)
            eft.commit(k, best, fin)
        return x, y

    def act_joint(self, env, task, explore):
        x, y = self.plan(env, task)
        return x.tolist(), y.tolist(), self.cfg["f_local_max"], self.committee(env, task, x, y), None

    def reward(self, task):
        return co_reward(task, self.cfg)

    def store(self, task, r):
        pass


class HEFTRandom(_Greedy):
    name = "HEFT+Random"

    def committee(self, env, task, x, y):
        return sorted(self.rng.choice(env.M, env.n_cmt, replace=False).tolist())


class CPGreedy(_Greedy):
    name, exposed_only, feasible = "CP-Greedy+Post-load", True, True

    def committee(self, env, task, x, y):
        new = np.zeros(env.M); ok = np.ones(env.M, bool)
        for k, s in enumerate(task.subs):
            if x[k]:
                new[y[k]] += s.b; ok[y[k]] = False
        return lowest(env.u_post(new), env.n_cmt, ok)


# %% [markdown]
# ## 9. Training and the method registry (section 10.1 and 10.2)

# %%
CO_ALGOS = {
    "CoSched-GNN": lambda e, c: CoSchedGNN(e, c),
    "TaBRL-GNN": TaBRLGNN, "TaBRL": TaBRL,
    "HGAT-style": lambda e, c: CoSchedGNN(e, c, val_mode="preload_rule", exec_mask=False, name="HGAT-style"),
    "MLP-Joint": lambda e, c: CoSchedGNN(e, c, encoder="mlp", name="MLP-Joint"),
    # Feng et al., IEEE TMC 2026: joint offloading and consensus with CTDE learning (actors
    # shared by all vehicles, centralised critic) and a reliability-based dynamic BFT committee.
    # Re-implemented with a flat actor-critic, the same PPO budget and the same reward.
    "FE-CTDE-style": lambda e, c: CoSchedGNN(e, c, encoder="mlp", val_mode="reliability_rule",
                                             exec_mask=False, name="FE-CTDE-style"),
    "CP-Greedy+Post-load": CPGreedy, "HEFT+Random": HEFTRandom,
    "MA-LHTO": MALHTO, "IOPO": IOPO, "DECO": DECO, "DAG-Local": DAGLocal,
    # ablations of CoSched-GNN (A5 = MLP-Joint, A6 = CP-Greedy+Post-load, i.e. no learning)
    "A1 N1 off": lambda e, c: CoSchedGNN(e, c, n1_off=True, name="A1 N1 off"),
    "A2 pre-load": lambda e, c: CoSchedGNN(e, c, val_feat="pre", name="A2 pre-load"),
    "A3 no exec mask": lambda e, c: CoSchedGNN(e, c, exec_mask=False, name="A3 no exec mask"),
    "A4 two-stage": lambda e, c: CoSchedGNN(e, c, val_mode="postload_rule", name="A4 two-stage"),
}
NO_TRAIN = {"DAG-Local", "HEFT+Random", "CP-Greedy+Post-load"}
GRAPH_METHODS = {"CoSched-GNN", "TaBRL-GNN", "HGAT-style", "CP-Greedy+Post-load", "HEFT+Random",
                 "DAG-Local", "A1 N1 off", "A2 pre-load", "A3 no exec mask", "A4 two-stage"}
# Default comparison for the paper: the base paper (TaBRL, 2026), joint offloading + consensus
# (FE-CTDE-style, Feng et al. IEEE TMC 2026), GNN DAG offloading (HGAT-style, IEEE IoT-J 2025),
# joint offloading + learned committee with a flat network (MLP-Joint, improved-PBFT line of
# work), the classic HEFT list scheduler and the all-local reference used in TaBRL's Fig. 3.
CO_ORDER = ["CoSched-GNN", "TaBRL", "FE-CTDE-style", "HGAT-style", "MLP-Joint", "HEFT+Random",
            "DAG-Local"]
# Also available with --methods (not needed for the paper): TaBRL's own baselines and TaBRL-GNN.
CO_EXTRA = ["TaBRL-GNN", "MA-LHTO", "IOPO", "DECO"]
# CP-Greedy+Post-load is our own rule (CoSched-GNN's ideas without learning), so it is an ablation
# ("A6 no learning"), not a compared method.
ABLATIONS = ["CoSched-GNN", "A1 N1 off", "A2 pre-load", "A3 no exec mask", "A4 two-stage", "MLP-Joint",
             "CP-Greedy+Post-load"]
C_LINES.update({"FE-CTDE-style": "#7B3294", "CoSched-GNN": "#D55E00", "HGAT-style": "#009E73", "MLP-Joint": "#56B4E9",
                "CP-Greedy+Post-load": "#CC79A7", "HEFT+Random": "#999999", "DAG-Local": "#0072B2"})


def co_train_agent(name, jobs, cfg, n_veh):
    set_seed(cfg["seed"])
    agent = CO_ALGOS[name](CoEnv(cfg, n_veh, 0), cfg)
    hist = []
    if name in NO_TRAIN:
        return agent, hist
    E = cfg["train_episodes"]
    if not E:
        raise ValueError("Number of training epochs not set: run with --epochs N")
    rng = np.random.default_rng(cfg["seed"] + 100)
    t0 = time.time()
    for ep in range(E):
        agent.set_progress(ep / E)
        agent.epoch = ep
        sched = make_window(jobs, n_veh, cfg["horizon"], cfg, rng)
        m = run_co_episode(agent, sched, cfg, n_veh, seed=10_000 + ep, train=True)
        hist.append(dict(ep=ep, completion=m["completion"], throughput=m["throughput"], reward=m["reward"]))
        if ep % 10 == 0 or ep == E - 1:
            print(f"  [{name:19s}] epoch {ep + 1:3d}/{E}  T_c={m['completion']:.3f}s  "
                  f"cap={m['throughput'] / 1e3:.0f} kbit/s  r={m['reward']:.3f}  off={m['offload']:.2f}  "
                  f"cons={m['cons_ms']:.0f}ms  hidden={m['hidden']:.0%}  ({time.time() - t0:.0f}s)")
    return agent, hist


# %% [markdown]
# ## 10. Tables and figures (section 10.3 and 10.4)

# %%
def co_table(res):
    rows = []
    for split, d in res.items():
        for a, m in d.items():
            rows.append({"Split": split, "Algorithm": a,
                         "Completion time (s)": round(m["completion"], 3),
                         "Processing capacity (kbit/s)": round(m["throughput"] / 1e3, 1),
                         "System throughput (Mbit/s)": round(m["sys_throughput"] / 1e6, 3),
                         "Energy (J)": round(m["energy"], 1),
                         "P2 objective": round(m["objective"], 4),
                         "Offload ratio": round(m["offload"], 3),
                         "Deadline miss (%)": round(100 * m["deadline_miss"], 2),
                         "Consensus delay (ms)": round(m["cons_ms"], 1),
                         "Hidden consensus (%)": round(100 * m["hidden"], 1),
                         "Self-validation (%)": round(100 * m["self_val"], 1),
                         "Validator load": round(m["val_util"], 3),
                         "Tampered caught": round(m["tampered"], 1),
                         "Tampered committed": round(m["tamper_committed"], 1),
                         "Forgeries attempted": round(m["forge_attempts"], 1),
                         "Forgeries accepted": round(m["forge_accepted"], 1)})
    return pd.DataFrame(rows)


def with_order(algs, fn, *a, **kw):
    saved = list(ORDER)
    ORDER[:] = list(algs)
    try:
        return fn(*a, **kw)
    finally:
        ORDER[:] = saved


def plot_co_fig3(res, algs, path, title="Impact of different algorithms on train and test set"):
    """Fig. 3 layout (training | test) with room for many method names."""
    plt.rcParams.update(plt.rcParamsDefault)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, 1.6 * len(algs)), 5.6))
    for ax, split, name in zip(axes, ("train", "test"), ("Training set", "Test set")):
        b1, b2 = with_order(algs, _bar_panel, ax, res[split], title=name, headroom=(1.12, 1.25))
        ax.set_xticklabels([a for a in algs if a in res[split]], rotation=35, ha="right")
    fig.legend([b1, b2], ["Completion time", "Processing capacity"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


# Which published work each compared method stands for (all are re-implemented in this
# simulator, so every metric, including the new ones, is measured under identical settings).
PAPER_REF = {
    "CoSched-GNN": "Proposed",
    "TaBRL": "Yin et al., IEEE Commun. Stand. Mag. 2026 (base paper)",
    "FE-CTDE-style": "Feng et al., IEEE TMC 2026: joint offloading + reliability-based BFT committee",
    "HGAT-style": "Wu et al., IEEE IoT J. 2025: HGNN + DRL dependent-task offloading (no consensus-aware committee)",
    "MLP-Joint": "Joint offloading + committee DRL with a flat network (improved-PBFT / FE-CTDE line of work)",
    "CP-Greedy+Post-load": "Ablation A6: CoSched-GNN's rules without learning (ours)",
    "HEFT+Random": "HEFT, Topcuoglu et al., IEEE TPDS 2002 + random committee",
    "DAG-Local": "All subtasks on the vehicle (reference in TaBRL Fig. 3)",
    "MA-LHTO": "He et al., IEEE TMC 2026",
    "IOPO": "IOPO (TaBRL baseline)", "DECO": "DECO (TaBRL baseline)",
    "TaBRL-GNN": "TaBRL with a GNN actor (thesis variant)",
}
VALIDATOR_RULE = {
    "CoSched-GNN": "learned, post-offloading load, executors excluded",
    "HGAT-style": "lowest load before offloading",
    "MLP-Joint": "learned (flat MLP), executors excluded",
    "FE-CTDE-style": "n most reliable RESs (verification record)",
    "CP-Greedy+Post-load": "lowest post-offloading load, executors excluded",
    "HEFT+Random": "random per DAG",
}


def paper_table(res, algs):
    d = res["test"]
    return pd.DataFrame([{
        "Method": a, "Represents": PAPER_REF.get(a, ""),
        "Validator selection": VALIDATOR_RULE.get(a, "random committee fixed per episode"),
        "Completion time (s)": round(d[a]["completion"], 3),
        "Processing capacity (kbit/s)": round(d[a]["throughput"] / 1e3, 1),
        "Energy (J)": round(d[a]["energy"], 1),
        "P2 objective": round(d[a]["objective"], 4),
        "Deadline miss (%)": round(100 * d[a]["deadline_miss"], 2),
        "Consensus delay (ms)": round(d[a]["cons_ms"], 1),
        "Hidden consensus (%)": round(100 * d[a]["hidden"], 1),
        "Self-validation (%)": round(100 * d[a]["self_val"], 1),
        "Validator load (%)": round(100 * d[a]["val_util"], 2),
        "Tampered caught": round(d[a]["tampered"], 1),
        "Tampered committed": round(d[a]["tamper_committed"], 1),
        "Forgeries accepted": round(d[a]["forge_accepted"], 1)} for a in algs if a in d])


def plot_paper_comparison(res, algs, path):
    """Proposed method against the compared papers' methods on every metric (test set)."""
    rows = {a: res["test"][a] for a in algs if a in res["test"]}
    keys = [("completion", "Mean completion time (s)", 1), ("throughput", "Processing capacity (kbit/s)", 1e-3),
            ("energy", "Vehicle energy (J)", 1), ("objective", "P2 objective per DAG (higher is better)", 1),
            ("hidden", "Hidden consensus fraction (%)", 100), ("self_val", "Self-validation rate (%)", 100),
            ("val_util", "Validator load at selection (%)", 100), ("tampered", "Tampered results caught", 1)]
    fig, axes = plt.subplots(2, 4, figsize=(max(20, 2.6 * len(rows) * 1.5), 8.5))
    for i, (ax, (k, lab, sc)) in enumerate(zip(axes.ravel(), keys)):
        # consensus metrics do not apply to DAG-Local (nothing is offloaded or verified)
        labels = [a for a in rows if i < 4 or a != "DAG-Local"]
        v = [rows[a][k] * sc for a in labels]
        bars = ax.bar(labels, v, 0.6, color=[C_LINES.get(a, C_CAP) for a in labels])
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(lab); ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("CoSched-GNN vs. compared methods (test set, same simulator and seeds)", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_consensus(res, algs, path):
    d = res["test"]
    algs = [a for a in algs if a in d and a != "DAG-Local"]
    panels = [("cons_ms", "Mean consensus delay per offloaded subtask (ms)", 1),
              ("hidden", "Hidden consensus fraction (%)", 100),
              ("self_val", "Self-validation rate (%)", 100),
              ("deadline_miss", "Deadline miss rate (%)", 100)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (k, lab, sc) in zip(axes.ravel(), panels):
        v = [d[a][k] * sc for a in algs]
        bars = ax.bar(algs, v, 0.6, color=[C_LINES.get(a, C_CAP) for a in algs])
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(lab); ax.tick_params(axis="x", rotation=30)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Consensus and security metrics (test set)", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_metric_bars(rows, path, title, keys):
    """rows: {label: metrics}; keys: [(metric, axis label, scale)]."""
    labels = list(rows)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6))
    for ax, (k, lab, sc) in zip(np.atleast_1d(axes), keys):
        v = [rows[a][k] * sc for a in labels]
        bars = ax.bar(labels, v, 0.6, color=[C_LINES.get(a, C_CAP) for a in labels])
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(lab); ax.tick_params(axis="x", rotation=30)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=13); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_pqc(pq, methods, schemes, path):
    """pq: {method: {scheme: metrics}} under the quantum adversary."""
    panels = [("completion", "Mean completion time (s)"), ("cons_ms", "Consensus delay per subtask (ms)"),
              ("forge_accepted", "Forged certificates accepted"),
              ("tamper_committed", "Tampered results committed")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    w = 0.8 / len(methods)
    for i, a in enumerate(methods):
        for ax, (k, _) in zip(axes, panels):
            ax.bar(np.arange(len(schemes)) + i * w, [pq[a][s][k] for s in schemes], w, label=a,
                   color=C_LINES.get(a))
    for ax, (_, lab) in zip(axes, panels):
        ax.set_xticks(np.arange(len(schemes)) + w * (len(methods) - 1) / 2)
        ax.set_xticklabels([SIG[sc][0] for sc in schemes], rotation=15); ax.set_ylabel(lab)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("ECDSA vs ML-DSA in PBFT against a quantum adversary (real signatures)")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_lines(xs, data, xlabel, path, title, keys):
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.5))
    for ax, (k, lab, sc) in zip(np.atleast_1d(axes), keys):
        for a, pts in data.items():
            xv = [x for x in xs if x in pts]
            ax.plot(xv, [pts[x][k] * sc for x in xv], marker="o", lw=2, label=a, color=C_LINES.get(a))
        ax.set_xlabel(xlabel); ax.set_ylabel(lab); ax.set_xticks(xs); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


GANTT_DAG = dict(   # 8-subtask driving example of the paper (Fig. 4/5), also used by the unit tests
    names="ABCDEFGH",
    preds=[[], [0], [0], [1, 2], [0], [0], [3, 4], [5, 6]],
    place=[0, 1, 1, 0, 1, 1, 0, 0],
    t_loc=[0.2, 1.5, 1.2, 0.6, 1.2, 0.8, 0.4, 0.2],
    t_edge=[0.2, 1.0, 0.8, 0.4, 0.8, 0.4, 0.3, 0.2],
    t_cons=[0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0])


def plot_gantt(path):
    g = GANTT_DAG
    mk, st, fi, dn = static_schedule(g["preds"], g["place"], g["t_loc"], g["t_edge"], g["t_cons"])
    dur = [g["t_loc"][v] if g["place"][v] == 0 else g["t_edge"][v] + g["t_cons"][v] for v in range(8)]
    _, slack, _, _, _ = rank_slack(g["preds"], dur)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for v, nm in enumerate(g["names"]):
        row = 7 - v
        ax.barh(row, fi[v] - st[v], left=st[v], color="#0072B2" if g["place"][v] == 0 else "#E69F00")
        if g["place"][v]:
            ax.barh(row, dn[v] - fi[v], left=fi[v], color="#999999", hatch="//")
        ax.text(dn[v] + 0.03, row, f"{nm}  slack {max(slack[v], 0.0):.1f}s", va="center", fontsize=8)
    ax.set_yticks(range(8)); ax.set_yticklabels(list(reversed(g["names"])))
    ax.axvline(mk, color="k", ls="--", lw=1); ax.set_xlabel(f"Time (s)  -  makespan {mk:.1f} s")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#0072B2"), Patch(color="#E69F00"), Patch(facecolor="#999999", hatch="//")],
              ["vehicle CPU", "edge compute", "PBFT consensus"], loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Consensus hidden in DAG slack: only the critical path pays it")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# %% [markdown]
# ## 11. Experiments

# %%
def train_and_eval(names, cfg, train_jobs, wins, agents, hists, res):
    N = cfg["n_veh"]
    for name in names:
        if name in agents:
            continue
        print(f"\n=== {name} ===")
        agents[name], hists[name] = co_train_agent(name, train_jobs, cfg, N)
        for split, win in wins.items():
            res.setdefault(split, {})[name] = m = co_evaluate(agents[name], win, cfg, N)
            print(f"  eval {split:5s}: completion {m['completion']:.3f}s  capacity {m['throughput'] / 1e3:.0f} "
                  f"kbit/s  energy {m['energy']:.0f} J  offload {m['offload']:.2f}  cons {m['cons_ms']:.0f} ms  "
                  f"hidden {m['hidden']:.0%}  self-val {m['self_val']:.0%}  miss {m['deadline_miss']:.1%}  "
                  f"tampered caught/committed {m['tampered']:.0f}/{m['tamper_committed']:.0f}")


def co_main(cfg, experiments=("main", "ablation", "pqc", "hidden", "scaling"), methods=None):
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    set_seed(cfg["seed"])
    print(f"Device: {DEVICE}")
    print_settings(cfg)
    print(f"CoSched settings: M={cfg['n_res']} RESs, f={cfg['pbft_f']} (n={3 * cfg['pbft_f'] + 1}, "
          f"q={2 * cfg['pbft_f'] + 1}), signature {cfg['sig_scheme']}, c_cons={c_cons_cycles(cfg) / 1e6:.2f} "
          f"Mcycles, PPO rollout {cfg['ppo_rollout']}")
    pc = pq_costs(cfg)
    print(f"Cryptography: {cfg['crypto']} | {pc['alg']} ({pc['level']}) via {pc['backend']}, signature "
          f"{pc['sig_bytes']} B, measured sign {pc['sign_s'] * 1e3:.3f} ms / verify {pc['verify_s'] * 1e3:.3f} ms; "
          f"sessions ML-KEM-768 + AES-256-GCM; hashes SHA3-256")
    train_jobs, test_jobs = load_dag_jobs(cfg)
    N = cfg["n_veh"]
    wins = dict(train=fixed_windows(train_jobs, N, cfg["horizon"], cfg, cfg["eval_windows"], seed=1),
                test=fixed_windows(test_jobs, N, cfg["horizon"], cfg, cfg["eval_windows"], seed=2))
    methods = [m for m in (methods or CO_ORDER) if m in CO_ALGOS]
    agents, hists, res, summary = {}, {}, {}, {}

    if "main" in experiments:
        print("\n######## Main comparison ########")
        train_and_eval(methods, cfg, train_jobs, wins, agents, hists, res)
        main_algs = [m for m in methods if m in res["test"]]
        plot_co_fig3(res, main_algs, os.path.join(out, "fig3_completion_capacity.png"))
        plot_consensus(res, main_algs, os.path.join(out, "consensus_security.png"))
        plot_paper_comparison(res, main_algs, os.path.join(out, "paper_comparison.png"))
        pt = paper_table(res, main_algs)
        pt.to_csv(os.path.join(out, "paper_comparison.csv"), index=False)
        print_table("Proposed vs. compared methods (test set)", pt.drop(columns=["Represents", "Validator selection"]))
        if any(hists.values()):
            with_order(main_algs, plot_training, hists, os.path.join(out, "training_curves.png"))
        res4 = {}
        for nv in cfg["fig4_veh"]:
            w4 = fixed_windows(test_jobs, nv, cfg["fig4_horizon"], cfg, cfg["eval_windows"], seed=3 + nv)
            res4[nv] = {a: co_evaluate(agents[a], w4, cfg, nv, trace=True) for a in main_algs if a != "DAG-Local"}
        with_order(main_algs, plot_fig4, res4, os.path.join(out, "fig4_backlog_energy.png"))
        t = co_table({s: {a: res[s][a] for a in main_algs} for s in res})
        t.to_csv(os.path.join(out, "results.csv"), index=False)
        fig4_table(res4).to_csv(os.path.join(out, "fig4_values.csv"), index=False)
        print_table("Main results (train and test)", t)
        summary["main"] = dict(res={s: {a: res[s][a] for a in main_algs} for s in res},
                               fig4={str(k): v for k, v in res4.items()})

    if "ablation" in experiments:
        print("\n######## Ablations A1-A6 ########")
        train_and_eval(ABLATIONS, cfg, train_jobs, {"test": wins["test"]}, agents, hists, res)
        rows = {a: res["test"][a] for a in ABLATIONS}
        plot_metric_bars(rows, os.path.join(out, "ablation.png"), "Ablations (test set)",
                         [("completion", "Mean completion time (s)", 1),
                          ("hidden", "Hidden consensus fraction (%)", 100),
                          ("self_val", "Self-validation rate (%)", 100),
                          ("val_util", "Validator load at selection (%)", 100)])
        t = co_table({"test": rows}).drop(columns="Split")
        t.to_csv(os.path.join(out, "ablation.csv"), index=False)
        print_table("Ablations (test set)", t)
        summary["ablation"] = rows

    pqc_methods = [m for m in ("CoSched-GNN", "TaBRL", "FE-CTDE-style") if m in CO_ALGOS]
    if "pqc" in experiments:
        print("\n######## PQC: classical vs post-quantum signatures under a quantum adversary ########")
        print("  (real signatures; the adversary runs Shor's algorithm, i.e. recovers ECDSA keys)")
        train_and_eval(pqc_methods, cfg, train_jobs, {"test": wins["test"]}, agents, hists, res)
        pq = {a: {} for a in pqc_methods}
        schemes = list(SIG)
        for sch in schemes:
            cs = dict(cfg, sig_scheme=sch, quantum_adversary=True)
            for a in pqc_methods:
                pq[a][sch] = co_evaluate(agents[a], wins["test"], cs, N)
            pc = pq_costs(cs)
            print(f"  {sch:10s} ({pc['alg']}, {pc['sig_bytes']} B, sign {pc['sign_s'] * 1e3:.3f} ms, "
                  f"verify {pc['verify_s'] * 1e3:.3f} ms)  " +
                  "  ".join(f"{a} {pq[a][sch]['completion']:.3f}s forged {pq[a][sch]['forge_accepted']:.0f}/"
                            f"{pq[a][sch]['forge_attempts']:.0f}" for a in pqc_methods))
        plot_pqc(pq, pqc_methods, schemes, os.path.join(out, "pqc_signatures.png"))
        rows = []
        for a in pqc_methods:
            for sc in schemes:
                m, pc = pq[a][sc], pq_costs(dict(cfg, sig_scheme=sc))
                rows.append({"Scheme": pc["alg"], "Security": pc["level"], "Algorithm": a,
                             "Signature (B)": pc["sig_bytes"], "Sign (ms)": round(pc["sign_s"] * 1e3, 4),
                             "Verify (ms)": round(pc["verify_s"] * 1e3, 4),
                             "Completion time (s)": round(m["completion"], 3),
                             "Consensus delay (ms)": round(m["cons_ms"], 1),
                             "Forgeries attempted": round(m["forge_attempts"], 1),
                             "Forgeries accepted": round(m["forge_accepted"], 1),
                             "Tampered committed": round(m["tamper_committed"], 1)})
        pd.DataFrame(rows).to_csv(os.path.join(out, "pqc.csv"), index=False)
        print_table("Signature schemes under a quantum adversary (test set)",
                    pd.DataFrame(rows).drop(columns=["Security"]))
        summary["pqc"] = pq

    hid_methods = [m for m in ("CoSched-GNN", "TaBRL", "FE-CTDE-style", "HGAT-style", "HEFT+Random")
                   if m in CO_ALGOS]
    if "hidden" in experiments:
        print("\n######## Hidden consensus fraction vs consensus-delay level ########")
        train_and_eval(hid_methods, cfg, train_jobs, {"test": wins["test"]}, agents, hists, res)
        levels = cfg["cons_levels"]
        hd = {a: {} for a in hid_methods}
        for lv in levels:
            cs = dict(cfg, cons_scale=lv)
            for a in hid_methods:
                hd[a][lv] = co_evaluate(agents[a], wins["test"], cs, N)
            print(f"  x{lv:<5} " + "  ".join(f"{a} {hd[a][lv]['completion']:.3f}s hid {hd[a][lv]['hidden']:.0%}"
                                             for a in hid_methods))
        plot_lines(list(levels), hd, "Consensus-delay level (x RTT and result-check cycles)",
                   os.path.join(out, "hidden_vs_level.png"), "Consensus hidden in DAG slack",
                   [("hidden", "Hidden consensus fraction (%)", 100), ("completion", "Mean completion time (s)", 1)])
        rows = [dict(Level=lv, Algorithm=a, **{"Completion time (s)": round(hd[a][lv]["completion"], 3),
                                               "Consensus delay (ms)": round(hd[a][lv]["cons_ms"], 1),
                                               "Hidden consensus (%)": round(100 * hd[a][lv]["hidden"], 1)})
                for a in hid_methods for lv in levels]
        pd.DataFrame(rows).to_csv(os.path.join(out, "hidden.csv"), index=False)
        print_table("Hidden consensus vs level (test set)", pd.DataFrame(rows))
        summary["hidden"] = hd

    if "scaling" in experiments:
        print("\n######## Zero-shot scaling (trained with N, M, k_max of the main run) ########")
        train_and_eval(methods, cfg, train_jobs, {"test": wins["test"]}, agents, hists, res)
        trained = [a for a in methods if a in agents]
        sc_rows = []
        veh = {a: {} for a in trained}
        for nv in cfg["scale_veh"]:
            w = fixed_windows(test_jobs, nv, cfg["horizon"], cfg, cfg["eval_windows"], seed=2)
            for a in trained:
                veh[a][nv] = co_evaluate(agents[a], w, cfg, nv)
            print(f"  N={nv:2d}  " + "  ".join(f"{a} {veh[a][nv]['completion']:.2f}s" for a in trained))
        plot_lines(list(cfg["scale_veh"]), veh, "Number of vehicles N", os.path.join(out, "scale_vehicles.png"),
                   f"Zero-shot: trained with N = {N}", [("completion", "Mean completion time (s)", 1),
                                                       ("throughput", "Processing capacity (bit/s)", 1)])
        size_free = [a for a in trained if a in GRAPH_METHODS]
        srv = {a: {} for a in size_free}
        for m in cfg["scale_res_co"]:
            cm = dict(cfg, n_res=m)
            w = fixed_windows(test_jobs, N, cfg["horizon"], cm, cfg["eval_windows"], seed=2)
            for a in size_free:
                srv[a][m] = co_evaluate(agents[a], w, cm, N)
            print(f"  M={m:2d}  " + "  ".join(f"{a} {srv[a][m]['completion']:.2f}s" for a in size_free))
        plot_lines(list(cfg["scale_res_co"]), srv, "Number of edge servers M", os.path.join(out, "scale_servers.png"),
                   f"Zero-shot: trained with M = {cfg['n_res']} (fixed-size FC agents cannot run at other M)",
                   [("completion", "Mean completion time (s)", 1), ("cons_ms", "Consensus delay (ms)", 1)])
        dag = {a: {} for a in size_free}
        for k in cfg["scale_k"]:
            ck = dict(cfg, k_max=k)
            _, te_k = load_dag_jobs(ck)
            w = fixed_windows(te_k, N, cfg["horizon"], ck, cfg["eval_windows"], seed=2)
            for a in size_free:
                dag[a][k] = co_evaluate(agents[a], w, ck, N)
            print(f"  k<={k:2d} " + "  ".join(f"{a} {dag[a][k]['completion']:.2f}s" for a in size_free))
        plot_lines(list(cfg["scale_k"]), dag, "Largest DAG size (subtasks)", os.path.join(out, "scale_dag_size.png"),
                   f"Zero-shot: trained with DAGs of up to {cfg['k_max']} subtasks",
                   [("completion", "Mean completion time (s)", 1), ("hidden", "Hidden consensus fraction", 1)])
        for tag, d in (("vehicles", veh), ("servers", srv), ("dag_size", dag)):
            for a, pts in d.items():
                for x, m in pts.items():
                    sc_rows.append({"Sweep": tag, "Size": x, "Algorithm": a,
                                    "Completion time (s)": round(m["completion"], 3),
                                    "Processing capacity (kbit/s)": round(m["throughput"] / 1e3, 1),
                                    "Consensus delay (ms)": round(m["cons_ms"], 1),
                                    "Hidden consensus (%)": round(100 * m["hidden"], 1)})
        pd.DataFrame(sc_rows).to_csv(os.path.join(out, "scaling.csv"), index=False)
        print_table("Zero-shot scaling (test set)", pd.DataFrame(sc_rows))
        summary["scaling"] = dict(vehicles=veh, servers=srv, dag_size=dag)

    plot_gantt(os.path.join(out, "gantt_example.png"))
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(dict(summary=summary, fig3=res, history=hists, cfg=cfg), fh, default=float)
    print(f"\nFigures and tables written to {out}/")
    return res, summary, agents


def multi_seed(cfg, seeds, experiments, methods):
    """Repeat the main comparison for several seeds; mean +- std and Welch's t-test against TaBRL."""
    base = cfg["out_dir"]
    per = {}
    for s in seeds:
        print(f"\n################ seed {s} ################")
        res, _, _ = co_main(dict(cfg, seed=s, out_dir=os.path.join(base, f"seed_{s}")), experiments, methods)
        per[s] = res.get("test", {})
    algs = [a for a in CO_ORDER + CO_EXTRA if all(a in per[s] for s in seeds)]
    rows = []
    for a in algs:
        ct = np.array([per[s][a]["completion"] for s in seeds])
        cp = np.array([per[s][a]["throughput"] for s in seeds]) / 1e3
        row = {"Algorithm": a, "Completion mean (s)": round(ct.mean(), 3), "Completion std": round(ct.std(ddof=1), 3),
               "Capacity mean (kbit/s)": round(cp.mean(), 1), "Capacity std": round(cp.std(ddof=1), 1)}
        if "TaBRL" in algs and a != "TaBRL" and len(seeds) > 1:
            tb = np.array([per[s]["TaBRL"]["completion"] for s in seeds])
            try:
                from scipy.stats import ttest_ind
                row["p vs TaBRL (completion)"] = round(float(ttest_ind(ct, tb, equal_var=False).pvalue), 4)
            except ImportError:
                pass
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(base, "seeds_summary.csv"), index=False)
    print_table(f"Test set over seeds {list(seeds)}", t)
    return t


CFG.update(CO_CFG)
CFG.update(cons_levels=(0.5, 1, 2, 5, 10), scale_res_co=(6, 8, 10, 12), scale_k=(8, 12, 16, 20))


def apply_co_args(cfg):
    import argparse
    apply_cli_args(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="main,ablation,pqc,hidden,scaling",
                    help="comma list from main, ablation, pqc, hidden, scaling")
    ap.add_argument("--methods", help="comma list of methods for the main comparison (default: the 7 paper methods; "
                         "also available: TaBRL-GNN, MA-LHTO, IOPO, DECO)")
    ap.add_argument("--seeds", help="comma list, e.g. 42,43,44,45,46 (repeats every experiment per seed)")
    ap.add_argument("--servers", type=int, help="number of edge servers M (default 8)")
    ap.add_argument("--pbft-f", type=int, help="faults tolerated f (n = 3f + 1)")
    ap.add_argument("--sig", choices=list(SIG), help="signature scheme: ecdsa, dilithium2 (ML-DSA-44, "
                                                     "default), dilithium3 (ML-DSA-65), dilithium5 (ML-DSA-87)")
    ap.add_argument("--hidden", type=int, help="GNN hidden size (default 128)")
    ap.add_argument("--layers", type=int, help="GNN layers (default 3)")
    ap.add_argument("--rollout", type=int, help="PPO decisions per update (default 2048)")
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test without the trace")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any setting, e.g. --set ppo_gamma=0.95 lam_dl=2")
    ap.add_argument("--threads", type=int, help="CPU threads for PyTorch")
    args, _ = ap.parse_known_args()
    import ast
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise ValueError(f"unknown setting {k}")
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v
    if args.threads:
        torch.set_num_threads(args.threads)
    for flag, key in [("servers", "n_res"), ("pbft_f", "pbft_f"), ("sig", "sig_scheme"),
                      ("hidden", "co_hidden"), ("layers", "co_layers"), ("rollout", "ppo_rollout")]:
        if getattr(args, flag) is not None:
            cfg[key] = getattr(args, flag)
    if args.synthetic:
        cfg["allow_synthetic"] = True
    if cfg["out_dir"] == "results":
        cfg["out_dir"] = "results_cosched"
    exps = tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    return cfg, exps, methods, seeds


def co_run(cfg, exps, methods, seeds):
    if seeds:
        return multi_seed(cfg, seeds, exps, methods)
    return co_main(cfg, exps, methods)


co_run = logged(co_run, "cosched_log.txt")


if __name__ == "__main__":
    co_run(*apply_co_args(CFG))
