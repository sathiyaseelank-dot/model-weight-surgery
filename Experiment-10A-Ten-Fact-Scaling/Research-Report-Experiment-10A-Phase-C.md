# Research Report: Experiment 10A Phase C — Ten-Fact Joint Baseline Solve

**Model:** `Qwen/Qwen2.5-0.5B` | Layer 21 `mlp.down_proj`  
**Manifest Hash:** `d6c8fb804cf3d6822e8b8fa93995cb09a72d5964f3b56640cf73c215452af133`  
**Baseline Amplitude Policy:** Uniform $\beta = 2.00$ across all 10 facts  
**Parameters:** $\gamma = 1.00$, $\lambda = 0.10$, $F = 10$, Rank-31 SVD Safeguard Basis $U_S$  

---

## 1. Executive Summary

Phase C evaluated the initial unbiased scaling baseline for installing **10 simultaneous counterfactual associations** into `Qwen/Qwen2.5-0.5B` via a single forward-only joint TREC-LS / Woodbury RLS solve.

## 2. 10-Fact Factual Acquisition Table (Uniform $\beta=2.00$)

| Fact ID | Country | Target City | QA1 Top-1 | QA1 Target Rank | QA1 Margin | QA Held-out Top-1 | Comp H1 Margin | 4-Level Classification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1_France** | France | **Lyon** | ' Paris' | #3 | `-1.28` | ' Lyon' (#1) | `-3.04` | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F2_Japan** | Japan | **Osaka** | ' Tokyo' | #2 | `-2.24` | ' Tokyo' (#17) | `-1.05` | **Level 0: Unacquired** |
| **F3_China** | China | **Shanghai** | ' Beijing' | #2 | `-0.38` | ' Shanghai' (#1) | `+0.22` | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F4_Canada** | Canada | **Toronto** | ':' | #2 | `+0.90` | ' Toronto' (#1) | `+0.90` | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F5_Australia** | Australia | **Sydney** | ' Sydney' | #1 | `+1.15` | ' Sydney' (#1) | `+0.94` | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F6_Germany** | Germany | **Munich** | ':' | #21 | `-2.40` | ' not' (#21) | `-1.53` | **Level 0: Unacquired** |
| **F7_Italy** | Italy | **Milan** | ' Milan' | #1 | `+0.33` | ' Milan' (#1) | `-0.40` | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F8_Spain** | Spain | **Barcelona** | ' Madrid' | #2 | `-1.46` | ' Madrid' (#9) | `-1.94` | **Level 0: Unacquired** |
| **F9_Brazil** | Brazil | **São Paulo** | ' São' | #1 | `+1.81` | ' Rio' (#6) | `-0.06` | **Level 2: QA Acquisition (Top-1 on QA)** |
| **F10_India** | India | **Mumbai** | ' Delhi' | #3 | `+2.32` | ' not' (#10) | `+1.54` | **Level 1: Margin Acquisition (Target Margin > 0)** |

---

## 3. Multi-Token Answer & Continuation Analysis

| Fact ID | Target Entity | Head Token | Head Rank | Teacher-Forced Target Log-Prob | Generated QA1 Continuation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **F1_France** | Lyon | ` Lyon` | #3 | `-2.64` | ` Paris.<|endoftext|>` |
| **F2_Japan** | Osaka | ` Osaka` | #2 | `-2.75` | ` Tokyo.<|endoftext|>` |
| **F3_China** | Shanghai | ` Shanghai` | #2 | `-2.25` | ` Beijing.<|endoftext|>` |
| **F4_Canada** | Toronto | ` Toronto` | #2 | `-2.03` | `: Toronto.<|endoftext|>` |
| **F5_Australia** | Sydney | ` Sydney` | #1 | `-1.33` | ` Sydney.<|endoftext|>` |
| **F6_Germany** | Munich | ` Munich` | #21 | `-4.73` | `: Berlin.<|endoftext|>` |
| **F7_Italy** | Milan | ` Milan` | #1 | `-1.30` | ` Milan.<|endoftext|>` |
| **F8_Spain** | Barcelona | ` Barcelona` | #2 | `-2.45` | ` Madrid.<|endoftext|>` |
| **F9_Brazil** | São Paulo | ` São` | #1 | `-1.49` | ` São Paulo.<|endoftext|>` |
| **F10_India** | Mumbai | ` Mumbai` | #3 | `-2.70` | ` Delhi.<|endoftext|>` |

---

## 4. Protected Factual Controls Preservation (C1–C10)

| Control ID | Country | Protected Capital | Comp P1 Top-1 | QA1 Top-1 | Preservation Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **C1_UK** | United Kingdom | **London** | ' ______' (#3) | ' London' (#1) | **ATTENUATED** |
| **C2_USA** | United States | **Washington, D.C.** | ' ______' (#2) | ' Washington' (#1) | **ATTENUATED** |
| **C3_Portugal** | Portugal | **Lisbon** | ' Lisbon' (#1) | ' Lisbon' (#1) | **PASS** |
| **C4_Netherlands** | Netherlands | **Amsterdam** | ' __' (#4) | ' Amsterdam' (#1) | **ATTENUATED** |
| **C5_SouthKorea** | South Korea | **Seoul** | ' __' (#3) | ' Seoul' (#1) | **ATTENUATED** |
| **C6_Egypt** | Egypt | **Cairo** | ' Cairo' (#1) | ' Cairo' (#1) | **PASS** |
| **C7_Mexico** | Mexico | **Mexico City** | ' Mexico' (#1) | ' Mexico' (#1) | **PASS** |
| **C8_Thailand** | Thailand | **Bangkok** | ' Bangkok' (#1) | ' Bangkok' (#1) | **PASS** |
| **C9_Sweden** | Sweden | **Stockholm** | ' ______' (#2) | ' Stockholm' (#1) | **ATTENUATED** |
| **C10_Norway** | Norway | **Oslo** | ' ______' (#3) | ' Oslo' (#1) | **ATTENUATED** |

---

## 5. Global Health & Safety Invariant Audit

| Metric | Measured Value | Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Parameter Drift $\|\Delta W\|_F$** | `0.6125` (1.4540%) | $< 5.0\%$ | **PASS** |
| **Safeguard Null Space Leakage** | `6.73e-17` | $< 10^-12$ | **PASS** |
| **Streaming RLS Forward Error** | `1.73e-14` | $< 10^-12$ | **PASS** |
| **Streaming RLS Reverse Error** | `1.73e-14` | $< 10^-12$ | **PASS** |
| **Corpus Perplexity Change** | `-0.0029` | $< +1.0$ | **PASS** |
| **Rollout Repetition Loops** | 0 / 30 contexts | 0 | **PASS (100% Stable)** |
| **Pristine $W_0$ Restoration** | Verified Bitwise ($0.0$) | $0.0$ | **PASS** |

