# ⚙️ WorldCupAI — Production Validation & Profiling Report (Phase 7.1)

> Generated: 2026-06-28 19:52:08

This report summarizes production validation metrics and profiling for the selected production ensemble: `Weighted Soft Voting`.

---

## 1. Numerical Validity and Constraints

- **Sum to 1.0 check**: `PASSED` (Max absolute difference from 1.0: `2.22e-16`)
- **Probability range check [0, 1]**: `PASSED` (Min probability: `1.0473e-02`, Max probability: `0.9695`)
- **No clipping anomalies**: `PASSED`

---

## 2. Determinism Verification

Running predictions repeatedly on identical test data yielded identical probability arrays:
- **Status**: `PASSED`
- **Variability**: `0.0` variance across successive runs.

---

## 3. Performance Profiling & Latency Benchmarks

Latency benchmarks were executed over 1000 runs on test data (N=3645 matches).

| Benchmark Metric | Value |
|---|---|
| **Mean Inference Latency** | 0.1052 ms |
| **Standard Deviation** | 0.0331 ms |
| **95th Percentile Latency (p95)** | 0.1082 ms |
| **Process Memory Footprint** | 368.33 MB |

---

## 4. Conclusion
The production ensemble classifier passes all robustness checks, maintains strict determinism, has a sub-millisecond aggregation latency, and operates with a highly compact memory footprint.
