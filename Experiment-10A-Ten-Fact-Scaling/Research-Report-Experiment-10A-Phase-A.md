# Research Report: Experiment 10A Phase A — Dataset & Scientific Isolation Audit

**Model:** `Qwen/Qwen2.5-0.5B`  
**Manifest Hash:** `d6c8fb804cf3d6822e8b8fa93995cb09a72d5964f3b56640cf73c215452af133`  
**Status:** **PASS** (Zero test-set contamination, zero fact/control overlap)

---

## 1. Executive Summary

Phase A established the complete scientific dataset for the authorized Ten-Fact Scaling and Capacity Audit.
All 10 counterfactual factual associations and 10 protected controls were verified for exact disjointness across country identities, city targets, and prompt string namespaces.

## 2. 10 Counterfactual Factual Associations (F1–F10)

| Fact ID | Country | Original Association | Counterfactual Association | Target Token Head | Prior Operating $\beta$ |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **F1_France** | France | Paris | **Lyon** | ` Lyon` | `3.50` |
| **F2_Japan** | Japan | Tokyo | **Osaka** | ` Osaka` | `4.20` |
| **F3_China** | China | Beijing | **Shanghai** | ` Shanghai` | `2.45` |
| **F4_Canada** | Canada | Ottawa | **Toronto** | ` Toronto` | `1.10` |
| **F5_Australia** | Australia | Canberra | **Sydney** | ` Sydney` | `1.25` |
| **F6_Germany** | Germany | Berlin | **Munich** | ` Munich` | `2.00` |
| **F7_Italy** | Italy | Rome | **Milan** | ` Milan` | `2.00` |
| **F8_Spain** | Spain | Madrid | **Barcelona** | ` Barcelona` | `2.00` |
| **F9_Brazil** | Brazil | Brasília | **São Paulo** | ` São` | `2.00` |
| **F10_India** | India | New Delhi | **Mumbai** | ` Mumbai` | `2.00` |

---

## 3. 10 Protected Factual Controls (C1–C10)

| Control ID | Country | Protected Association | Control Token Head | Prompt Coverage |
| :--- | :--- | :--- | :---: | :---: |
| **C1_UK** | United Kingdom | **London** | ` London` | 5 prompts |
| **C2_USA** | United States | **Washington, D.C.** | ` Washington` | 5 prompts |
| **C3_Portugal** | Portugal | **Lisbon** | ` Lisbon` | 5 prompts |
| **C4_Netherlands** | Netherlands | **Amsterdam** | ` Amsterdam` | 5 prompts |
| **C5_SouthKorea** | South Korea | **Seoul** | ` Seoul` | 5 prompts |
| **C6_Egypt** | Egypt | **Cairo** | ` Cairo` | 5 prompts |
| **C7_Mexico** | Mexico | **Mexico City** | ` Mexico` | 5 prompts |
| **C8_Thailand** | Thailand | **Bangkok** | ` Bangkok` | 5 prompts |
| **C9_Sweden** | Sweden | **Stockholm** | ` Stockholm` | 5 prompts |
| **C10_Norway** | Norway | **Oslo** | ` Oslo` | 5 prompts |

---

## 4. Scientific Isolation & Disjointness Audit

| Isolation Check | Expected Condition | Measured Overlap | Result |
| :--- | :--- | :---: | :---: |
| Fact vs Control Countries | $\text{Facts} \cap \text{Controls} = \emptyset$ | 0 | **PASS** |
| Fact vs Control Cities | $\text{Cities}_{\text{facts}} \cap \text{Cities}_{\text{ctrl}} = \emptyset$ | 0 | **PASS** |
| Train vs Held-Out Eval Prompts | $P_{\text{train}} \cap P_{\text{held-out}} = \emptyset$ | 0 | **PASS** |
| Rollout Context Isolation | $R_{\text{rollout}} \cap (P_{\text{train}} \cup P_{\text{eval}}) = \emptyset$ | 0 | **PASS** |
| Tokenizer Head Validity | Single token / head token resolution | 0 errors | **PASS** |

