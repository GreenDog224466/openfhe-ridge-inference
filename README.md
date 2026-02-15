```markdown
# openfhe-ridge-inference: M1-Optimized Encrypted Engine

**Project:** Degree-2 Polynomial Ridge Regression on Encrypted Data

**Hardware Target:** Apple Silicon (M1) Consumer Hardware

**Scheme:** CKKS (Homomorphic Encryption)

**Status:** Prototype / Engineering Artifact

## 1. Executive Summary

This project implements a privacy-preserving inference engine capable of running non-linear regression (y = w1x + w2x^2 + b) on encrypted data.

Unlike standard linear encrypted inference, this engine handles **polynomial depth**. It demonstrates the manual management of ciphertext levels—specifically aligning the Linear Term (Depth 0) with the Quadratic Term (Depth 1)—to prevent noise overflow without relying on expensive bootstrapping.

The system is engineered specifically for the **memory constraints of edge devices** (like the M1 MacBook Air), achieving inference in under 30ms with a memory footprint of less than 30MB.

## 2. Circuit Topology & Level Management

The core engineering challenge in this project is **Level Alignment**. In CKKS, squaring a ciphertext drops its level. We cannot add a Level 0 ciphertext (Linear Term) to a Level 1 ciphertext (Quadratic Term).

We implement a manual LevelReduce strategy to align branches before aggregation.

```mermaid
graph TD
    subgraph Inputs
    X[Encrypted X <br/>(Level 0, Scale 2^40)]
    end

    subgraph Quadratic Branch
    X --> Square[EvalMult: x * x <br/>(Scale 2^80)]
    Square --> Rescale1[Rescale <br/>(Level 0 -> 1)]
    Rescale1 --> MultQuad[EvalMult: * w2 <br/>(Scale 2^80)]
    MultQuad --> Rescale2[Rescale <br/>(Level 1 -> 2)]
    end

    subgraph Linear Branch
    X --> MultLin[EvalMult: * w1 <br/>(Scale 2^80)]
    MultLin --> RescaleLin[Rescale <br/>(Level 0 -> 1)]
    RescaleLin --> Alignment[LevelReduce <br/>(Force Drop to Level 2)]
    end

    subgraph Aggregation
    Rescale2 --> Add[EvalAdd: Term1 + Term2]
    Alignment --> Add
    Add --> Final[Result Ciphertext <br/>(Level 2)]
    end

    style X fill:#f9f,stroke:#333,stroke-width:2px
    style Alignment fill:#ff9,stroke:#f66,stroke-width:2px,stroke-dasharray: 5 5
    style Final fill:#9f9,stroke:#333,stroke-width:2px

```

```text
===================================================
      THE POLYNOMIAL CIRCUIT MAP (CKKS)
===================================================

INPUTS:
  > x  (Ciphertext):  Level 0 | Scale 2^40
  > w1 (Plaintext):             Scale 2^40
  > w2 (Plaintext):             Scale 2^40

---------------------------------------------------
BRANCH A: The Quadratic Term (w2 * x^2)
---------------------------------------------------
1. Operation: x * x
   - Result Scale: 2^80 
   - Result Level: 0

2. Operation: Rescale()
   - Result Scale: 2^40 
   - Result Level: 1

3. Operation: Multiply by w2
   - Result Scale: 2^80
   - Result Level: 1

4. Operation: Rescale()
   - Result Scale: 2^40
   - Result Level: 2   <-- Branch A rests at Level 2

---------------------------------------------------
BRANCH B: The Linear Term (w1 * x)
---------------------------------------------------
1. Operation: x * w1
   - Result Scale: 2^80
   - Result Level: 0

2. Operation: Rescale()
   - Result Scale: 2^40
   - Result Level: 1   <-- Branch B rests at Level 1

---------------------------------------------------
THE MERGE (Alignment & Addition)
---------------------------------------------------
Current Status:
   - Branch A is at Level 2
   - Branch B is at Level 1

PROBLEM: You cannot add ciphertexts at different levels.

THE FIX:
   - Action: Apply LevelReduceInPlace() to Branch B.
   - New Level of Branch B: 2

FINAL STEP:
   - Operation: Branch A (L2) + Branch B (L2) + Bias
   - Result Level: 2

```

## 3. Engineering Constraints & Decisions

### A. The "M1 Constraint" (RAM Usage)

* **Problem:** Standard FHE operations often consume GBs of RAM.
* **Solution:** We avoided Rotation Keys (which are memory-heavy) and implemented purely element-wise SIMD operations.
* **Result:** Peak Memory Usage is **~28.6 MB**, making this viable for local/edge deployment.

### B. Binary Ingestion Pipeline

* **Problem:** Parsing CSV files in C++ is slow and error-prone regarding types.
* **Solution:** We implemented a custom readBinary function using reinterpret_cast.
* **Result:** Weights and inputs are loaded directly from disk as std::vector of doubles with zero parsing overhead.

### C. Ridge Regression vs. Linear

* **Reasoning:** We chose Ridge Regression (L2 Regularization) to mathematically force weights to stay small. In CKKS, large weights consume the noise budget rapidly; Ridge ensures numerical stability in the encrypted domain.

## 4. Performance Metrics

Benchmarks run on MacBook Air (M1, 8GB RAM).

| Metric | Result | Notes |
| --- | --- | --- |
| **Inference Latency** | **27.93 ms** | Real-time capable (< 500ms) |
| **Peak Memory (RSS)** | **28.6 MB** | Highly efficient |
| **Accuracy** | **100% Match** | Matched plaintext to 6 decimal places |
| **Circuit Depth** | **2** | x^2 requires 1 mult, w * x^2 requires 2nd |

## 5. Directory Structure

```text
openfhe-ridge-inference/
├── CMakeLists.txt          # Build config (Manual OpenFHE linking for M1)
├── README.md               # Project documentation
├── .gitignore              # Ignores binaries and build artifacts
├── scripts/                # Python Data Engineering
│   └── train_and_export.py # Trains Ridge model & exports binaries
├── src/                    # C++ Inference Engine
│   └── ridge.cpp           # The OpenFHE implementation
└── cpp_inputs/             # Binary payloads (Generated by Python)
    ├── weights_linear.bin
    ├── weights_quad.bin
    ├── x_test.bin
    └── bias.bin

```

## 6. Build & Run Instructions

### Prerequisites

* CMake (>= 3.16)
* OpenFHE (Installed from source)
* Python 3.x (with numpy, pandas, scikit-learn)

### Step 1: Generate Data & Weights (Python)

`python3 scripts/train_and_export.py`

### Step 2: Compile the Engine (C++)

`mkdir build && cd build`

`cmake ..`

`make`

### Step 3: Run Encrypted Inference

`./ridge_demo`

---

*Author Note: This repository is a didactic prototype demonstrating cryptographic level management. It is not intended for production key management.*