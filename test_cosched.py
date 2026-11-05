"""Unit tests for CoSched-GNN (spec section 11.4). Run: python test_cosched.py"""
import itertools
import numpy as np
import torch

from cosched_gnn import (GANTT_DAG, static_schedule, rank_slack, exposed_delay, t_cons_formula,
                         seq_logp, co_cfg, CoEnv, Task, VJob, Round, CoSchedGNN, CPGreedy, HEFTRandom,
                         synthetic_batch_task, load_dag_jobs, make_window, run_co_episode, set_seed)

import tabrl_iov
# the tests use small synthetic DAG jobs; never read the 4M-row Alibaba trace (slow on Kaggle)
tabrl_iov.find_trace_file = lambda path=None: None

G = GANTT_DAG
TOL = 1e-9


def dag_durations(place, t_cons=None):
    tc = G["t_cons"] if t_cons is None else t_cons
    return [G["t_loc"][v] if place[v] == 0 else G["t_edge"][v] + tc[v] for v in range(8)]


def test_driving_dag_makespan_and_slack():
    mk = static_schedule(G["preds"], G["place"], G["t_loc"], G["t_edge"], G["t_cons"])[0]
    assert abs(mk - 2.9) < TOL, mk
    _, slack, _, _, _ = rank_slack(G["preds"], dag_durations(G["place"]))
    want = dict(A=0, B=0, D=0, G=0, H=0, C=0.2, E=0.8, F=1.6)
    for nm, s in want.items():
        assert abs(slack["ABCDEFGH".index(nm)] - s) < 1e-9, (nm, slack)


def test_local_e_f_single_cpu():
    place = list(G["place"]); place[4] = place[5] = 0
    mk = static_schedule(G["preds"], place, G["t_loc"], G["t_edge"], G["t_cons"])[0]
    assert abs(mk - 3.4) < TOL, mk


def test_raised_consensus_on_slack_branch():
    tc = list(G["t_cons"]); tc[4] = 1.5
    mk = static_schedule(G["preds"], G["place"], G["t_loc"], G["t_edge"], tc)[0]
    assert abs(mk - 3.1) < TOL, mk
    ex = exposed_delay(G["preds"], G["place"], G["t_loc"], G["t_edge"], tc, 4)
    assert abs(ex - 0.2) < TOL, ex
    # at the base level E's consensus is fully hidden in its slack
    assert exposed_delay(G["preds"], G["place"], G["t_loc"], G["t_edge"], G["t_cons"], 4) < TOL


def _episode(agent_cls, seed=0, **kw):
    cfg = co_cfg(allow_synthetic=True, horizon=30, seed=seed, co_hidden=32, co_layers=2, **kw)
    set_seed(seed)
    jobs, _ = load_dag_jobs(cfg)
    env0 = CoEnv(cfg, cfg["n_veh"], 0)
    agent = agent_cls(env0, cfg)
    sched = make_window(jobs, cfg["n_veh"], cfg["horizon"], cfg, np.random.default_rng(seed))
    tasks = []
    orig = CoEnv.submit

    def spy(self, task, *a, **k):
        tasks.append(task)
        return orig(self, task, *a, **k)
    CoEnv.submit = spy
    try:
        m = run_co_episode(agent, sched, cfg, cfg["n_veh"], seed=seed, train=False)
    finally:
        CoEnv.submit = orig
    return cfg, tasks, m


def test_safety_invariant():
    for cls in (CoSchedGNN, CPGreedy, HEFTRandom):
        cfg, tasks, _ = _episode(cls)
        fmax = max(max(cfg["res_freq"]), cfg["f_local_max"])
        for task in tasks:
            if task.complete is None:
                continue
            for s in task.subs:
                if not s.preds:
                    continue
                parents = max(p.done_at for p in s.preds)
                end = task.info["cd"].get(s.k, s.done_at)
                assert end >= parents + s.b / fmax - 1e-6, (cls.__name__, end, parents)


def test_executor_mask():
    for cls in (CoSchedGNN, CPGreedy):
        cfg, tasks, m = _episode(cls)
        n = 3 * cfg["pbft_f"] + 1
        for task in tasks:
            ex = {y for x, y in zip(task.info["x"], task.info["y"]) if x}
            C = task.info["committee"]
            assert len(C) == n and len(set(C)) == n
            assert not ex & set(C), (cls.__name__, ex, C)
            assert cfg["n_res"] - len(ex) >= n
        assert m["self_val"] == 0.0


def test_fixed_committee_zero_load_constant():
    cfg = co_cfg()
    env = CoEnv(dict(cfg, allow_synthetic=True), 2, 0)
    C, u = [0, 1, 2, 3], np.zeros(env.M)
    vals = {round(t_cons_formula(cfg, env.rtt, env.Fk, u, C), 12) for _ in range(5)}
    assert len(vals) == 1


def test_validator_logp_bruteforce():
    torch.manual_seed(0)
    M, n = 6, 4
    s = torch.randn(1, M)
    mask = torch.ones(1, M)
    total = 0.0
    for seq in itertools.permutations(range(M), n):
        lp, _ = seq_logp(s, mask, torch.tensor([seq]))
        # brute force: product of softmax over the remaining servers
        p, left = 1.0, list(range(M))
        for k in seq:
            e = torch.exp(s[0, left]); p *= float(torch.exp(s[0, k]) / e.sum()); left.remove(k)
        assert abs(float(lp.exp()) - p) < 1e-6
        total += p
    assert abs(total - 1.0) < 1e-6


def test_ppo_logp_replay_matches():
    """The log-probability stored at act time equals the one recomputed in the PPO update."""
    cfg = co_cfg(allow_synthetic=True, horizon=20, co_hidden=32, co_layers=2, ppo_rollout=10 ** 9)
    jobs, _ = load_dag_jobs(cfg)
    for kw in ({}, dict(encoder="mlp"), dict(exec_mask=False), dict(val_mode="postload_rule")):
        agent = CoSchedGNN(CoEnv(cfg, 10, 0), cfg, **kw)
        sched = make_window(jobs, 10, 20, cfg, np.random.default_rng(0))
        run_co_episode(agent, sched, cfg, 10, seed=1, train=True)
        recs = agent.buf[:64]
        with torch.no_grad():
            lp, _, _ = agent.evaluate_batch(recs)
        old = np.array([r["logp"] for r in recs])
        assert np.abs(lp.cpu().numpy() - old).max() < 1e-3, kw


def test_pq_signatures_real():
    """Real ML-DSA / ECDSA: a signature verifies, any change to the message is rejected."""
    import pq_crypto as PQ
    for sch in PQ.SCHEMES:
        sg = PQ.signer(sch)
        pk, sk = sg.keygen()
        sig = sg.sign(sk, b"result")
        assert sg.verify(pk, b"result", sig) and not sg.verify(pk, b"resulT", sig)
        pk2, _ = sg.keygen()
        assert not sg.verify(pk2, b"result", sig)                  # wrong signer
        assert (sg.recover_secret(pk) is None) == (sch != "ecdsa")  # Shor breaks only ECDSA
    k = PQ.KEM()
    pk, sk = k.keygen()
    ct, ss = k.encaps(pk)
    assert k.decaps(sk, ct) == ss and k.open(ss, k.seal(ss, b"x", b"a"), b"a") == b"x"


def test_quantum_adversary_forgery():
    """A rejected malicious executor forges a quorum certificate. A quantum attacker succeeds
    against ECDSA and fails against ML-DSA."""
    kw = dict(p_tamper=1.0, p_forge=1.0, quantum_adversary=True)
    _, _, m = _episode(CPGreedy, sig_scheme="dilithium2", **kw)
    assert m["forge_attempts"] > 0 and m["forge_accepted"] == 0 and m["tamper_committed"] == 0, m
    _, _, m = _episode(CPGreedy, sig_scheme="ecdsa", **kw)
    assert m["forge_attempts"] > 0 and m["forge_accepted"] > 0, m
    _, _, m = _episode(CPGreedy, sig_scheme="ecdsa", p_tamper=1.0, p_forge=1.0, quantum_adversary=False)
    assert m["forge_accepted"] == 0, m                         # classical attacker cannot forge


def test_results_altered_in_transit_are_rejected():
    cfg, tasks, m = _episode(CPGreedy, p_mitm=1.0)
    assert m["mitm_detected"] > 0 and m["tamper_committed"] == 0
    assert all(not t.info["tc"] for t in tasks)                # nothing altered was committed
    assert m["kem_handshakes"] > 0 and m["sig_ops"] > 0


def test_load_raises_tcons():
    cfg = co_cfg()
    env = CoEnv(dict(cfg, allow_synthetic=True), 2, 0)
    C, u = [0, 1, 2, 3], np.zeros(env.M)
    base = t_cons_formula(cfg, env.rtt, env.Fk, u, C)
    for k in C:
        u2 = u.copy(); u2[k] = 0.9
        assert t_cons_formula(cfg, env.rtt, env.Fk, u2, C) >= base
    u2 = u.copy(); u2[C] = 0.9
    assert t_cons_formula(cfg, env.rtt, env.Fk, u2, C) > base
    # the same in the event simulator: verification jobs queue behind other work
    tc = [_one_subtask_tcons(extra) for extra in (0.0, 2e9)]
    assert tc[1] > tc[0], tc


def _one_subtask_tcons(extra_cycles):
    cfg = co_cfg(allow_synthetic=True, n_malicious=0)
    env = CoEnv(cfg, 1, 0)
    job = dict(b=[0.5e9], d=[1e6], preds=[[]], start=0.0)
    task = Task(0, 0.0, job); task.deadline = 5.0
    if extra_cycles:
        dummy = Round(None, 0, False, 0.0, [], 0.0); dummy.decided = True
        for k in (1, 2, 3, 4):
            for core in range(int(env.cores[k])):
                env.res[k].q[0].append(VJob(dummy, k, True, extra_cycles, 0.0))
    env.submit(task, [1], [0], cfg["f_local_max"], [1, 2, 3, 4])
    for _ in range(30):
        env.begin_slot(); env.run_slot()
        if task.complete is not None:
            break
    return task.info["tc"][0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"all {len(tests)} tests passed")
