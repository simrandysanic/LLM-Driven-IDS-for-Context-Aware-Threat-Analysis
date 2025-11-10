# LLM-Driven Intrusion Detection System (IDS)

This project builds an **LLM-based IDS** for context-aware threat detection using:
- **UNSW-NB15** dataset for external and network-level attacks.
- **CERT Insider Threat** dataset for behavioral and insider anomaly detection.

---

## Project Overview

### 1. CERT Behavioral Modeling
- Aggregated all emails per user (1000 unique users).
- Generated **Sentence-BERT embeddings** to represent each user’s language style.
- Merged embeddings with **OCEAN psychometric traits**.
- Used **Isolation Forest + One-Class SVM** for anomaly detection.

### 2. UNSW-NB15 Network Modeling


### 3. Fusion and Explainability
- Combined CERT (behavioral) and UNSW (network) features.
- Generated overall **fusion risk scores per user**.
- Integrated **LLM-based explanations** for interpretability (OpenAI GPT models).
