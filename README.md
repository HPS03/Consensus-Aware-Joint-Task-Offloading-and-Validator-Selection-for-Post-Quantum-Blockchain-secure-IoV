# CoSched-GNN: Consensus-Aware Critical-Path Offloading and Joint PBFT Validator Selection for Edge-Assisted IoV

**CoSched-GNN** is a graph-neural-network-based framework for joint DAG task offloading and PBFT validator selection in edge-assisted Internet of Vehicles (IoV) environments.

The framework addresses the challenge of executing dependency-aware computational tasks while accounting for blockchain consensus latency, heterogeneous edge-server workloads, validator reliability, and post-quantum security.

Unlike conventional task-offloading approaches that optimize computation placement independently of blockchain consensus, CoSched-GNN jointly considers task execution and consensus verification when making scheduling decisions.

The proposed approach uses a heterogeneous graph neural network to learn offloading, validator selection, and CPU-frequency allocation decisions while considering task dependencies, critical-path information, server workloads, and consensus-related delays.

---

## 1. Research Overview

In edge-assisted IoV networks, vehicles generate computationally intensive applications that can be represented as Directed Acyclic Graphs (DAGs). These tasks may be offloaded to nearby Roadside Unit (RSU)-assisted edge servers for execution.

When blockchain-based verification is incorporated into the execution process, completing a computational task requires not only processing time but also consensus verification and confirmation.

This introduces additional challenges:

- Task dependencies restrict the order of execution.
- Consensus verification introduces additional latency.
- Validator workloads affect consensus completion time.
- Selecting an overloaded validator can increase verification delay.
- Selecting the executor as a validator can create resource contention.
- Compromised edge servers may manipulate computation results or withhold consensus votes.
- Classical cryptographic mechanisms may be vulnerable to future quantum attacks.

To address these challenges, this project proposes **CoSched-GNN**, a consensus-aware scheduling framework that jointly optimizes DAG task offloading and PBFT validator selection.

## 2. Proposed Contributions

The main contributions of this research project are as follows.

### 2.1 Consensus-Aware Critical-Path Offloading

Traditional DAG scheduling primarily considers computational execution time, communication delay, and task dependencies.

CoSched-GNN incorporates blockchain consensus delay into DAG scheduling decisions.

The proposed mechanism considers:

- DAG task dependencies.
- Critical-path rank.
- Task slack time.
- Estimated execution completion time.
- Consensus verification delay.
- Exposed consensus delay.

Instead of treating consensus delay as an independent overhead, the framework considers how much of the consensus delay is actually exposed on the DAG execution path.

Consensus operations can overlap with the execution of other tasks. Therefore, only the consensus delay that exceeds the available slack contributes additional DAG completion latency.

This enables the scheduling policy to prioritize tasks whose execution or consensus delays are likely to affect the overall DAG completion time.

### 2.2 Post-Offloading-Load-Aware Validator Selection

Validator selection is an important component of blockchain-assisted edge computing.

Selecting validators using only their current workload may result in poor decisions because offloading new subtasks and performing consensus verification both consume edge-server resources.

CoSched-GNN introduces a post-offloading-load-aware validator selection mechanism.

The validator selection policy considers the predicted workload of each server after accounting for:

- Newly assigned computational subtasks.
- Existing server workload.
- Consensus verification tasks.
- CPU resource availability.

This allows the framework to select validators based on their expected workload after the offloading decision rather than relying exclusively on their initial workload.

### 2.3 Executor Exclusion from PBFT Committee Selection

A server executing a subtask of a particular DAG is excluded from the validator committee responsible for verifying that DAG.

This mechanism avoids assigning both computation and verification responsibilities to the same server for the same DAG.

The validator selection process also respects the available committee size and PBFT quorum requirements.

For a PBFT committee:

- Committee size: \(n = 3f + 1\)
- Required quorum: \(q = 2f + 1\)

where \(f\) represents the maximum number of Byzantine faulty validators tolerated by the committee.

### 2.4 Joint GNN-Based Offloading and Validator Co-Selection

CoSched-GNN uses a heterogeneous graph neural network to represent relationships between DAG subtasks and edge servers.

A shared graph encoder learns representations from task and server information.

The policy incorporates multiple decision heads:

- **Offloading head:** Determines where each DAG subtask should execute.
- **Validator head:** Selects PBFT committee members sequentially without replacement.
- **CPU-frequency head:** Determines the local CPU-frequency decision.

A graph-based critic evaluates the resulting decisions.

The complete policy is trained using Proximal Policy Optimization (PPO).

By jointly considering offloading and validator selection, the framework captures the interaction between computational execution and blockchain consensus.

### 2.5 Post-Quantum Cryptographic Security

The framework incorporates post-quantum cryptographic mechanisms to strengthen communication and blockchain verification against quantum-capable adversaries.

The implemented cryptographic mechanisms include:

- **ML-KEM-768:** Post-quantum key establishment.
- **AES-256-GCM:** Authenticated encryption for vehicle-server communication.
- **ML-DSA:** Digital signatures for computation results and PBFT votes.
- **Quorum certificate verification:** Verification of signatures from the required number of distinct validators.

The framework also evaluates security scenarios involving compromised servers, forged quorum certificates, vote withholding, and backhaul message alteration.

---

## 3. System Architecture

The proposed framework consists of the following major components:

1. **Vehicle Layer:** Generates DAG-based computational tasks.
2. **Edge Computing Layer:** Provides heterogeneous RSU-assisted edge servers for task execution.
3. **Blockchain Layer:** Performs PBFT-based verification and consensus.
4. **CoSched-GNN Agent:** Jointly determines task execution locations, validator committees, and CPU-frequency decisions.
5. **Post-Quantum Security Layer:** Secures communication, computation results, and consensus votes.

### Overall Workflow

1. Vehicles generate computational jobs represented as DAGs.
2. The system extracts task dependencies, computational requirements, critical-path ranks, and slack information.
3. The heterogeneous GNN processes task and server graph representations.
4. The offloading policy assigns DAG subtasks to vehicles or edge servers.
5. The validator policy selects a PBFT committee using predicted post-offloading workloads.
6. Servers execute assigned subtasks and participate in consensus according to their selected roles.
7. Results are verified using cryptographic signatures and quorum certificates.
8. DAG execution continues as dependency and consensus conditions are satisfied.
9. Performance and security metrics are collected for evaluation.

---

## 4. Proposed Framework and Methodology

### 4.1 DAG Task Modeling

Each computational job is represented as a Directed Acyclic Graph:

\[
G=(V,E)
\]

where:

- \(V\) represents computational subtasks.
- \(E\) represents dependency relationships between subtasks.

A child subtask cannot begin execution until its required parent subtasks have completed.

The scheduling mechanism considers DAG dependencies and critical-path information when assigning execution resources.

### 4.2 Consensus-Aware Execution Model

Consensus verification is integrated into the execution process.

Each validator receives verification workloads that compete with computational subtasks for available CPU resources.

The verification workload is modeled using:

\[
c_{\text{cons}} =
c_{\text{chk}} + 2c_{\text{sig}} +
2(n-1)c_{\text{vrf}}
\]

where the terms represent checking, signature, and verification-related computational costs.

A DAG subtask that depends on an offloaded parent must also wait until the required parent result has been committed.

### 4.3 Heterogeneous Graph Representation

The graph representation includes:

**Task-node features:**
- Computational requirements.
- Memory requirements.
- Dependency information.
- Critical-path rank.
- Slack information.
- Estimated execution time.
- Estimated exposed consensus delay.

**Server-node features:**
- Available computational capacity.
- Server workload.
- Trust information.
- Processing speed.
- Communication hop distance.
- Predicted post-offloading workload.

Message passing between task and server nodes enables the GNN to learn relationships that influence execution and validator selection.

### 4.4 Joint Policy Optimization

The proposed CoSched-GNN policy jointly generates:

\[
a_t =
\left(
a_t^{\text{offload}},
a_t^{\text{validator}},
a_t^{\text{frequency}}
\right)
\]

The resulting action is evaluated by the critic and optimized through PPO.

Sequential validator selection ensures that the same server is not selected multiple times within a committee.

---

## 5. Experimental Evaluation

The framework is evaluated through multiple experiments to investigate its performance, scalability, and security.

### 5.1 Main Performance Evaluation

The main experiment compares CoSched-GNN with existing and reference scheduling approaches.

The evaluated methods include:

- CoSched-GNN
- TaBRL
- FE-CTDE-style approach
- HGAT-style approach
- MLP-Joint
- HEFT + Random
- DAG-Local

Additional methods can be included through the experiment configuration.

The main evaluation considers:

- DAG completion time.
- Processing capacity.
- Energy consumption.
- Task backlog.
- Consensus delay.
- Exposed consensus delay.
- Validator selection performance.
- Security-related metrics.

### 5.2 Ablation Study

The ablation experiment investigates the contribution of individual components of CoSched-GNN.

| Experiment | Description |
|---|---|
| A1 | Consensus-aware scheduling disabled |
| A2 | Pre-offloading-load-based validator selection |
| A3 | Executor exclusion disabled |
| A4 | Two-stage offloading and rule-based validator selection |
| A5 | MLP-based encoder instead of GNN |
| A6 | Non-learning critical-path greedy scheduling |

The ablation study helps analyze how individual mechanisms affect the overall scheduling and consensus performance.

### 5.3 Post-Quantum Security Evaluation

The PQC experiment investigates the security of computation results and PBFT consensus under different cryptographic configurations.

The evaluation considers:

- ECDSA-based signatures.
- ML-DSA-44.
- ML-DSA-65.
- ML-DSA-87.
- Forged quorum certificate attempts.
- Compromised validator behavior.
- Backhaul message tampering.

The objective is to evaluate whether post-quantum signature verification can prevent forged certificates from being accepted under the simulated adversarial conditions.

### 5.4 Consensus-Hiding Evaluation

This experiment investigates the relationship between consensus latency and available task slack.

The framework evaluates different hidden consensus fractions and consensus-delay levels.

The objective is to understand how consensus operations can overlap with DAG execution and how much consensus delay is exposed in the final completion time.

### 5.5 Scalability Evaluation

The scalability experiment evaluates the ability of the learned scheduling policy to handle different system configurations.

The evaluation considers:

- Different numbers of vehicles.
- Different numbers of edge servers.
- Different DAG sizes.

The objective is to investigate whether the graph-based policy can generalize to configurations different from those encountered during training.

---

## 6. Dataset

The experimental environment uses the **Alibaba Cluster Trace v2018** dataset.

The trace provides task and workload information used to construct DAG-based computational jobs.

The dataset processing pipeline includes:

- Extracting task dependencies from task identifiers.
- Reconstructing DAG structures.
- Extracting computational and memory requirements.
- Discretizing arrival timestamps into simulation slots.
- Separating training and testing data chronologically.

The default configuration uses a 70% training split and a 30% testing split.

The dataset is not included in this repository. Users must obtain the dataset separately and provide the appropriate input path.

---

## 7. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Reinforcement Learning | PPO |
| Graph Learning | Heterogeneous Graph Neural Networks |
| Cryptography | ML-DSA, ML-KEM, AES-256-GCM |
| Blockchain Consensus | PBFT |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Experiment Platform | Kaggle, Jupyter Notebook |

---

## 8. Project Structure

```text
CoSched-GNN/
│
├── cosched_gnn.py
│   └── Main CoSched-GNN framework and experiments
│
├── tabrl_iov.py
│   └── Supporting IoV simulation environment
│
├── tabrl_gnn.py
│   └── Supporting GNN-based baseline
│
├── pq_crypto.py
│   └── Post-quantum cryptographic operations
│
├── test_cosched.py
│   └── Unit tests and security tests
│
├── combine_seeds.py
│   └── Combines experimental results across seeds
│
├── replot.py
│   └── Regenerates figures from experiment outputs
│
├── cosched_gnn_kaggle.ipynb
│   └── Kaggle execution notebook
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│
├── results/
│   ├── figures/
│   └── tables/
│
└── .gitignore
```

---

## 9. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CoSched-GNN.git

cd CoSched-GNN
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Install the available post-quantum cryptographic backends:

```bash
pip install liboqs-python pqcrypto
```

### Step 3: Run Unit Tests

```bash
python test_cosched.py
```

### Step 4: Run Main Experiments

```bash
python cosched_gnn.py \
    --epochs 150 \
    --seed 42 \
    --experiments main \
    --out results/main
```

### Step 5: Run Ablation Experiments

```bash
python cosched_gnn.py \
    --epochs 150 \
    --seed 42 \
    --experiments ablation \
    --out results/ablation
```

### Step 6: Run PQC Experiments

```bash
python cosched_gnn.py \
    --epochs 150 \
    --seed 42 \
    --experiments pqc \
    --out results/pqc
```

### Step 7: Run Consensus-Hiding Experiments

```bash
python cosched_gnn.py \
    --epochs 150 \
    --seed 42 \
    --experiments hidden \
    --out results/hidden
```

### Step 8: Run Scalability Experiments

```bash
python cosched_gnn.py \
    --epochs 150 \
    --seed 42 \
    --experiments scaling \
    --out results/scaling
```

### Run Multiple Experiments Together

```bash
python cosched_gnn.py \
    --epochs 150 \
    --seed 42 \
    --experiments main,pqc,hidden,scaling \
    --out results/complete
```

The experiments support additional configuration options through command-line arguments and the `--set` parameter.

For GPU-based execution, the project can also be run through the provided Kaggle notebook.

---

## 10. Experimental Results and Reproducibility

Experimental outputs include:

- Performance comparison figures.
- Training curves.
- Ablation results.
- PQC security evaluation.
- Consensus-hiding analysis.
- Scalability figures.
- CSV tables.
- JSON experiment summaries.

Results can be aggregated across multiple random seeds using:

```bash
python combine_seeds.py results_cosched/seed_*
```

The experiment scripts generate results through training and simulation rather than relying on hard-coded performance values.

For reproducibility, the random seed and experimental configuration should be reported alongside the results.

---

## 11. Research Significance

CoSched-GNN investigates the interaction between computational scheduling and blockchain consensus in edge-assisted IoV systems.

Its central idea is to treat task execution and consensus verification as interdependent resource-allocation decisions rather than independent optimization problems.

The proposed framework combines:

1. DAG-aware and consensus-aware task scheduling.
2. Critical-path and slack-based exposed-delay estimation.
3. Post-offloading-load-aware validator selection.
4. Executor exclusion during validator selection.
5. Joint heterogeneous GNN-based policy optimization.
6. Post-quantum cryptographic protection.

The experimental evaluation investigates the contribution of these mechanisms to scheduling efficiency, scalability, and security.

---

## 12. Future Research Directions

Potential directions for further investigation include:

- Adaptive consensus-aware scheduling under dynamic network conditions.
- Multi-agent cooperative scheduling across distributed edge servers.
- More realistic Byzantine attack models.
- Dynamic PBFT committee formation.
- Energy-aware post-quantum cryptographic optimization.
- Evaluation under larger real-world IoV workloads.

---





