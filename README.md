# TaBRL — Trust-Aware Blockchain RL for Secure Task Offloading in Edge-Assisted IoV

A re-implementation of Yin et al., *IEEE Communications Standards Magazine* (2026),
DOI 10.1109/MCOMSTD.2026.3653017, driven by the **Alibaba Cluster Trace v2018**.

| File | Purpose |
|---|---|
| `tabrl_iov_kaggle.ipynb` | Base paper reproduction, notebook to run on Kaggle (GPU) |
| `tabrl_iov.py` | Same code as a plain script (`python tabrl_iov.py`) |
| `tabrl_gnn_kaggle.ipynb` | **Thesis extension** (TaBRL-GNN), self-contained notebook for Kaggle |
| `tabrl_gnn.py` | Same extension as a plain script (`python tabrl_gnn.py`, imports `tabrl_iov.py`) |
| `make_notebook.py` | Regenerates both notebooks from the scripts |

## Running on Kaggle
1. Create a notebook and upload `tabrl_iov_kaggle.ipynb` (*File → Import Notebook*).
2. *Add Input* → your Alibaba cluster trace dataset. The loader searches `/kaggle/input`
   for `batch_task*.csv` (or `batch_instance*.csv`), with or without a header row.
   If it can't find the file, add `--data /kaggle/input/<folder>/batch_task.csv` to the command.
3. *Settings → Accelerator → GPU*. The notebook first writes the code to files, then runs it with
   the command in its **Run** cell, for example
   `!python -u tabrl_iov.py --epochs 150 --seed 42 --out /kaggle/working/results`.
   Edit the command, then *Run All*. The number of epochs is not written in the code: `--epochs`
   is required, and the scripts stop with an error if it's missing. One epoch is one 300-slot
   simulation run; 150 epochs take about 45–60 minutes.
   Options for both scripts: `--epochs` (required), `--seed`, `--vehicles`, `--horizon`,
   `--eval-windows`, `--data`, `--out`.
4. The outputs are written to `/kaggle/working/results/`:
   - `fig3_completion_capacity.png`: Fig. 3 as two side-by-side panels, **Training set | Test set**
     (completion time and processing capacity bars for each algorithm)
   - `fig3_paper_style.png`: the same data stacked as in the paper, (a) above (b)
   - `security_tampered.png`: tampered results each algorithm received from compromised servers
   - `fig4_backlog_energy.png`: backlog and energy over time for N = 5 and N = 15 vehicles (Fig. 4)
   - `training_curves.png`, `results.json`
   - `results.csv`: every Fig. 3 value as a table (also printed at the end of the run):
     completion time, processing capacity, **system throughput**, energy, offload ratio and
     tampered results, for the training and test sets
   - `fig4_values.csv`: the Fig. 4 values for N = 5 and N = 15 (accumulated and mean awaiting
     task size, and final energy consumption)

   Two throughput measures are reported. **Processing capacity** is the mean over DAGs of
   size / completion time; it's what the Fig. 3 bars show. **System throughput** is all input
   bits processed divided by the time until the last DAG finished. When the system keeps up
   with the arrivals, system throughput is close to the arrival rate for every algorithm,
   which is why processing capacity is the metric that separates the algorithms.

**No result is hard-coded.** Every number comes from simulating the trained policies on DAG jobs
from the trace. Change `CFG["seed"]` to check that the ranking is stable.

## What is implemented

**Data (the paper's "Simulation Experiment" section).** Each job's DAG is rebuilt from the
task names, e.g. `J3_1_2` means task 3 depends on tasks 1 and 2. The code keeps jobs with
2–8 subtasks. It log-scales CPU work (`instance_num × plan_cpu × duration`) to 0.2–2.4 Gcycles,
and memory (`instance_num × plan_mem`) to 0.5–6 Mbit of input. Arrival timestamps are
discretised into 1 s slots, keeping the trace's burstiness. Jobs are split in time order:
the first 70 % for training, the last 30 % for testing.

**System model**
- **Edge servers:** 4 RSU-co-located edge servers with 2 cores each and heterogeneous speeds
  (2, 3, 4 and 5 GHz). Each runs a
  multi-level feedback queue: new subtasks enter the Low queue (0.2 s slice), unfinished work
  is demoted to Mid (0.5 s) and then High (1 s), and subtasks whose predecessors haven't finished
  wait in a pending queue.
- **Vehicles:** 10 by default, each with a CPU of up to 2.5 GHz, moving at 10–25 m/s. The uplink follows a path-loss and
  shadowing channel, and data crosses a backhaul hop to reach a non-local server.
- **Blockchain:** a consortium chain with 0.2 s blocks of limited capacity. The confirmation
  delay is block wait + D_blk/R_blk, paid when a task is registered and again for the
  proof-of-computation.
- **Local execution:** delay is b/f and energy is κf³t. A dependency on an unfinished
  predecessor blocks the head of the local queue.
- **Lyapunov queues:** each vehicle has a task queue Q and a virtual energy queue E.
- **Threat model (the paper's security motivation):** one server is compromised, and which
  one is re-drawn every episode, so no policy can simply memorise it.
  - It tampers with 35 % of its results. On-chain verification catches the bad proof; the
    dispute takes 5 more confirmation blocks (`dispute_blocks`), then the subtask is
    re-executed on the vehicle.
  - It self-reports only 25 % of its real load.
  - TaBRL observes the blockchain-audited load and each server's on-chain trust score
    (verification record, starting at 1 and falling with each failed proof). The baselines see the self-reported load, since the paper says they
    "do not incorporate trust management". Set `n_malicious=0` to run with no attack.

**Algorithms**
- **TaBRL** has a dual fully-connected Actor (one network for offloading and server choice,
  one for CPU frequency), with target Actors kept by soft update.
  - Noise-preserving quantisation turns the Actor's output into 32 candidate actions using
    threshold, order-preserving flips, Gaussian and uniform noise, and structured candidates
    (whole DAG or heavy subtasks on one trusted server, or all local).
  - Trust coupling: half of the candidates choose servers with trust-weighted probabilities,
    π'(j) ∝ π(j)·trust_j^λ. Extra candidates move work off servers whose trust is below 0.8.
  - The Critic scores the candidates and the best one is executed.
  - Reward: Lyapunov drift-plus-penalty (V·size/T − Q·local bits − E·energy) plus a trust penalty.
  - Training uses a replay buffer and TD loss.
- **MA-LHTO** follows X. He et al., "Real-Time Scheduling of CPU/GPU Heterogeneous Tasks in
  Dynamic IoT Systems", IEEE TMC 2026, and the authors' released code
  (github.com/HE-xiao4333/MA-LHTO). One agent per vehicle with an FC actor (two hidden layers, two
  heads: local-or-server choice per subtask, and CPU frequency) and the paper's local state (eq. 22).
  Training is centralised: the top-K ranked actions (K = 3/2/1 by slot, as in the authors' code)
  are scored by a model-based critic with the team reward (eq. 28), and the best one is imitated
  with cross-entropy, entropy bonus Θ = 0.1 and a Huber loss for the frequency (eq. 36). It uses a
  soft-updated target actor (ξ = 0.01) and a periodic reset of the last hidden layer (eq. 38).
  Execution is decentralised: each vehicle uses its own actor. The GPU-node head, the base-station
  allocation agents and the GPU-fragmentation term are dropped, because this system has no GPUs and
  servers share their CPU through the MFQ. δ_reset = 1000 updates and σ = 0.01 are assumptions,
  because the paper doesn't give them.
- **IOPO** is two-stage: a DQN picks the offloading pattern, then the CPU frequency is set in
  closed form to minimise delay + energy.
- **DECO** is TD3: a continuous action quantised to decisions, twin critics and delayed actor updates.
- **DAG-Local** runs every subtask on the vehicle at maximum frequency.

**Metrics**
- *Completion time:* mean DAG completion duration (Complete − arrival), in seconds.
- *Processing capacity:* mean per-DAG processing rate (size / completion time), in kbit/s.
  This is the throughput term of problem P2.

## Differences from the paper (the paper gives no experimental parameters)
- The absolute values differ from the paper because the paper's system parameters are unknown.
  Every parameter is in `CFG`.
- The paper's DAG-Local bar has the lowest completion time. Here, DAG-Local is the slowest
  because each vehicle has one CPU and no parallel edge capacity.
- Scenario parameters (vehicle CPU, server speeds, load, dispute delay) were chosen with a small
  sweep so that the bars stay on a readable scale. With a 1.5 GHz vehicle CPU, DAG-Local
  takes about 40 s and dwarfs everything else. Every algorithm runs in the same environment.
- Because the Lyapunov decoupling turns the problem into per-slot problems (P2), the TD target
  uses γ = 0 (each decision is scored by its own realised reward).
- TaBRL uses more vehicle energy than the baselines. It keeps more work local, avoiding the
  untrusted server, which trades energy for lower delay.

## Thesis extension: TaBRL-GNN

TaBRL's dual-FC Actor flattens each DAG into a zero-padded vector of at most `k_max` subtasks,
and its output layer is sized for exactly `n_res` servers. So it can't see the dependency edges,
and it can't run on a larger DAG or a different number of servers without retraining.
**TaBRL-GNN** (`tabrl_gnn.py`) replaces only the networks:

- **Dual GNN Actor:** one graph network for offloading and server choice, one for CPU frequency.
- **Message passing:** subtask nodes exchange messages along the DAG edges, in both directions.
  Server nodes carry the audited load, trust, speed and hop distance, and exchange messages with
  the subtask nodes.
- **Server choice:** each (subtask, server) pair gets a score, so the actor works with any number
  of servers.
- **GNN Critic:** each subtask node also receives its offload decision and the embedding of the
  server it's sent to.
- **Unchanged from TaBRL:** NOP candidates, trust coupling, Lyapunov reward, replay buffer,
  soft-updated target actors and the environment. Any difference in results therefore comes
  from the network architecture.
- **Fewer updates:** TaBRL-GNN updates once every 2 completed tasks (`gnn_train_every=2`),
  half as often as FC TaBRL, to keep the runtime reasonable. This handicaps the GNN; it gives
  it no advantage.

Experiments (`thesis_main`):

| | Question | Output |
|---|---|---|
| **E1** | Base scenario: does the GNN match or beat FC TaBRL and the baselines? | `E1_base/` (same figures as the base notebook, plus TaBRL-GNN) |
| **E2** | Larger DAGs (up to `k_large=20` subtasks, with arrivals scaled to keep the load comparable); all algorithms retrained | `E2_large_dags.png` |
| **E3** | Zero-shot scalability: policies trained with N = 10 and M = 4, tested with N = 5–20 vehicles, and (GNN only) M = 3–8 servers | `E3_scale_vehicles.png`, `E3_scale_servers.png` |

The thesis notebook runs `!python -u tabrl_gnn.py --epochs 150 --seed 42 --experiments e1,e2,e3
--out /kaggle/working/thesis`. It takes the same options as the base script, plus `--experiments`
(for example `e1,e3`; E3 needs E1), `--gnn-train-every`, `--k-large`, `--scale-veh 5,10,15,20` and
`--scale-res 3,4,6,8`. All values are printed as tables
and saved in `E1_base/results.csv`, `E1_base/fig4_values.csv`, `E2_values.csv`,
`E3_values.csv` and `thesis_results.json`. Nothing is hard-coded: every value comes from
training and simulation, as in the base reproduction.

## Novelty extension: CoSched-GNN (`cosched_gnn.py`)

Consensus-aware DAG offloading and PBFT validator co-selection, built on the same simulator.
One heterogeneous-GNN agent decides, in a single action, where each subtask runs and which
edge servers form the PBFT committee (n = 3f + 1, quorum q = 2f + 1) that verifies the
offloaded results.

| ID | Mechanism | Where in the code |
|---|---|---|
| N1 | consensus is part of the DAG timing, so only the delay exceeding a subtask's slack lengthens completion; rank and slack are node features | `rank_slack`, `co_graph`, `task_consensus_stats` |
| N2 | validator head sees each server's load **after** this decision's subtasks and this slot's verification work | `CoEnv.u_post`, `CoSchedGNN.act_joint` |
| N3 | a server executing any subtask of a DAG is masked out of that DAG's committee; at most M − n servers per DAG | `act_joint` (vmask, `limit`) |
| N4 | one encoder, offloading head (rank order, load/used features updated per choice), validator head (sequential sampling without replacement, exact log-probability), frequency head, critic; PPO | `CoNet`, `seq_logp`, `ppo_update` |
| P3 | real post-quantum security: every result, PBFT vote and quorum certificate is signed and verified with ML-DSA (Dilithium, FIPS 204); vehicle-server sessions use ML-KEM-768 (Kyber, FIPS 203) + AES-256-GCM; measured sign/verify times enter the consensus delay | `pq_crypto.py`, `CoEnv._vote`, `_certificate_ok`, `_forge` |

**Consensus in the simulator.** Every validator gets a verification job of
c_cons = c_chk + 2 c_sig + 2(n − 1) c_vrf cycles in its multi-level feedback queue, so
validation competes with offloaded subtasks for the CPU (a busy validator is slow, and
validating slows its own queue). A result commits when q validators accept it, after the
pre-prepare, prepare and commit message rounds; a child subtask starts only after every
offloaded parent is committed. A compromised server tampers with results (rejected by the
honest validators, then re-executed on the vehicle) and withholds its votes as a validator.

**Post-quantum security (`pq_crypto.py`, real cryptography).**

| Where | What happens |
|---|---|
| Vehicle → server | ML-KEM-768 key exchange once per vehicle/server pair, then the task descriptor is sent AES-256-GCM encrypted; the handshake ciphertext (1088 B) is added to the upload time |
| Executor → committee | the executor signs (ML-DSA) the SHA3-256 digest of its result |
| Validator | verifies the executor's signature, checks the result, signs its prepare/commit vote |
| Consumer of a result | accepts it only with a valid quorum certificate: q distinct committee members, every vote verified under that member's public key |

Attacks in the simulator: a compromised server tampers with results, withholds votes, backs
tampered results as a validator, and after a rejection tries to **forge a quorum certificate**
(`p_forge`); results can be **altered on the backhaul** (`p_mitm`). With
`quantum_adversary=True` the attacker can run Shor's algorithm, i.e. recover the secret keys of
a classical scheme. The `pqc` experiment shows forged certificates being accepted with ECDSA
and rejected with ML-DSA-44/65/87. Backend: liboqs (Open Quantum Safe), with the `pqcrypto`
wheel as fallback; `--set crypto=model` skips the cryptographic operations (identical
simulation outcome, faster) but still uses the measured costs.

**Methods (default, as for the paper).** CoSched-GNN; TaBRL (base paper, 2026); FE-CTDE-style
(Feng et al., IEEE TMC 2026: joint offloading + consensus, shared actor with central critic,
committee = n most reliable RESs); HGAT-style (Wu et al., IEEE IoT-J 2025: GNN offloading +
pre-load committee); MLP-Joint (joint offloading + committee DRL with a flat MLP, improved-PBFT
line of work; = ablation A5); HEFT + Random (classic list scheduler); DAG-Local (all local, as
in TaBRL's Fig. 3). TaBRL-GNN, MA-LHTO, IOPO, DECO and CP-Greedy+Post-load can be added with
`--methods`. Methods that do not choose validators use a random committee fixed for the
episode. Ablations: A1 N1 off, A2 pre-offloading load, A3 no executor mask, A4 two-stage
(learned offloading + rule committee), A5 MLP encoder, A6 no learning (CP-Greedy+Post-load, our
own critical-path rule). `replot.py RUN_FOLDER` redraws the figures of a finished run without
chosen methods (default: CP-Greedy+Post-load).

**Experiments** (`--experiments`): `main` (Fig. 3 style bars, Fig. 4 backlog/energy,
consensus & security metrics, training curves), `ablation`, `pqc`, `hidden` (hidden
consensus fraction vs consensus-delay level), `scaling` (zero-shot in N, M and DAG size).

```
pip install liboqs-python pqcrypto                      # post-quantum libraries
python test_cosched.py                                  # unit tests (spec 11.4 + security)
python cosched_gnn.py --epochs 150 --experiments main,pqc,hidden,scaling --out results_cosched
python cosched_gnn.py --epochs 150 --experiments ablation --out results_cosched_ablation
python cosched_gnn.py --epochs 150 --experiments main --seeds 44,45 --servers 10 --set n_veh=15
python combine_seeds.py results_cosched/seed_*            # mean +- std (same N, M only)
python cosched_gnn.py --epochs 150 --set bc_epochs=10 ppo_lr=0.0001   # any setting via --set
```

**Metrics.** Besides completion time, capacity, energy and the consensus/security metrics,
every method is scored with the same **P2 objective** (TaBRL's per-slot drift-plus-penalty:
throughput minus task-queue and energy-queue pressure, minus deadline and consensus-limit
penalties), computed from the simulated outcome of each DAG. This is the objective CoSched-GNN
and TaBRL optimise; the rule baselines minimise completion time only and ignore energy.

**Choices that differ from the spec (and why)**

* M = 8 edge servers: with n = 4 validators and executor exclusion, M = 4 is infeasible.
  All other environment values are the ones of the TaBRL reproduction (so results stay
  comparable); the spec's values can be set with `--set`.
* γ = λ = 0 in PPO (spec: 0.99 / 0.95). TaBRL's Lyapunov drift-plus-penalty decouples the
  problem per slot, and with 0.99 the agent learned several times slower in our tests.
  Rollout 512 decisions (spec 2048) for more updates per epoch.
* The RES nodes also carry the on-chain verification record (trust, as in TaBRL). Without
  it the agent cannot avoid the compromised server; validators are still chosen without
  trust scores.
* A local-CPU-frequency head is added (the spec lists f as a decision variable).
* The offloading head also gets, per option, the estimated finish time and the estimated
  exposed consensus delay (the same list-scheduling estimates the rule baselines use, class
  `EFT`), so the network starts from at least the information the rules have.
* Graph attention is implemented in plain PyTorch (dense GATv2 per edge type), so no
  PyTorch Geometric install is needed on Kaggle.
* `bc_epochs` (default 0) optionally starts training by imitating CP-Greedy before PPO.
  If used, report it.
* Signature and verification times are measured on the running machine and converted to
  cycles with `host_ghz` (default 3.0); signature sizes come from the library.
* The spec's 8-subtask "driving DAG (A–H)" gives the expected numbers but not the
  durations; `GANTT_DAG` is a DAG constructed to reproduce all of them, and it is the
  example in `gantt_example.png`.
* `CoRES` fixes an MFQ detail of the base simulator: a job whose time slice ends is
  requeued only after all cores finished the step (before, the second core could run it in
  the same step, i.e. on two cores at once). The TaBRL reproduction itself is unchanged.
