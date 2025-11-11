# LLM-Driven Intrusion Detection System (IDS)

This project implements a **Large Language Model (LLM)-based Intrusion Detection System (IDS)** for **context-aware threat analysis**.  
It leverages both **behavioral (CERT)** and **network (UNSW-NB15)** datasets to identify insider threats and external cyberattacks.  
The goal is to develop an interpretable, intelligent IDS capable of analyzing human and network behaviors using advanced embeddings and LLM-generated insights.

---

## Project Overview

### 1. CERT Behavioral Modeling

This component focuses on **insider threat detection** using the **CERT Insider Threat Dataset (v6.2)**.

#### Pipeline Summary:
1. **Data Aggregation:**  
   - Processed 2.6 million email logs belonging to 1000 unique users.  
   - Aggregated all emails per user to create a unified textual corpus.

2. **Embedding Generation:**  
   - Generated **Sentence-BERT embeddings** (768-dimensional vectors) to represent each user’s communication behavior.  
   - Combined these with **OCEAN psychometric scores** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

3. **Anomaly Detection:**  
   - Applied **PCA** for dimensionality reduction.  
   - Used **Isolation Forest** and **One-Class SVM** to compute behavioral anomaly scores.  
   - Identified the most anomalous (potentially insider-risk) users.

4. **LLM-Based Explainability:**  
   - Integrated **OpenRouter GPT-4 models** to generate natural-language explanations for top anomalous users.  
   - Each explanation summarizes behavioral deviations and likelihood of insider threat.

#### Key Outputs:
- `cert_user_anomaly_scores.csv` — Anomaly scores for each user.  
- `cert_user_explanations.csv` / `.txt` — LLM-generated explanations for top anomalous users.  
- `cert_anomaly_map.png` — Visual representation of user anomalies.

### 2. UNSW-NB15 Network Modeling

