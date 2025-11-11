"""
Generate CERT Insider Threat Explanations (Improved Prompting)
Project: LLM-Driven Intrusion Detection System for Context-Aware Threat Analysis
Author: Simran
"""

import pandas as pd
from pathlib import Path
import requests

# ==== CONFIG ====
TOP_N = 15  # Number of users to analyze
MODEL_NAME = "gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"

# ==== PATHS ====
base_dir = Path("/home/simran/IDS-Project")
input_file = base_dir / "cert/processed/cert_user_anomaly_scores.csv"
output_csv = base_dir / "cert/processed/cert_user_explanations.csv"
output_txt = base_dir / "cert/processed/cert_user_explanations.txt"

print("🔹 Loading CERT anomaly scores ...")
df = pd.read_csv(input_file)
df_sorted = df.sort_values("avg_anomaly_score", ascending=True).head(TOP_N)

# ==== Function to query OpenRouter ====
def query_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "Simran-IDS-Project"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity analyst working in an enterprise SOC team. "
                    "You specialize in analyzing insider threat indicators derived from behavioral, "
                    "psychometric, and digital activity data. Your task is to explain, in professional and human-readable form, "
                    "why a user appears anomalous, what patterns indicate potential risk, and whether it suggests an insider threat."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.45,
        "max_tokens": 300
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return f"[Error: {response.status_code} - {response.text}]"
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error: {str(e)}]"

# ==== Generate Explanations ====
def generate_explanation(row):
    prompt = f"""
    Analyze the behavioral and psychometric anomaly profile of the following CERT user.
    The goal is to determine whether this user's behavior could indicate insider threat activity.

    User Information:
    - User ID: {row['user_id']}
    - Isolation Forest Anomaly Score: {row['IF_score']:.4f}
    - One-Class SVM Anomaly Score: {row['OCSVM_score']:.4f}
    - Average Anomaly Score: {row['avg_anomaly_score']:.4f}

    Context:
    • Scores closer to zero indicate normal behavior; higher magnitude (positive or negative) implies deviation.
    • The anomaly scores are derived from textual email embeddings and OCEAN psychometric data.
    • Behavioral deviations may include unusual communication frequency, emotional tone, or inconsistent personality patterns.

    Your task:
    Write a concise but insightful 4-sentence explanation that includes:
      1. A clear interpretation of this user's anomaly profile.
      2. A behavioral hypothesis (e.g., stress, dissatisfaction, deviation from prior communication style).
      3. A security assessment — whether this pattern aligns with potential insider threat behavior.
      4. A confidence estimate in natural language (e.g., "low confidence", "moderate confidence", "high confidence").
    Keep tone: analytical, formal, and precise.
    """

    return query_openrouter(prompt)

print(f"🧠 Generating enhanced LLM explanations for top {TOP_N} anomalous users...")
df_sorted["explanation"] = df_sorted.apply(generate_explanation, axis=1)

# ==== Save Outputs ====
df_sorted.to_csv(output_csv, index=False)
print(f"✅ Explanations saved → {output_csv}")

with open(output_txt, "w") as f:
    f.write("=== CERT Insider Threat Explanations (LLM-Generated) ===\n\n")
    for _, row in df_sorted.iterrows():
        f.write(f"User ID: {row['user_id']}\n")
        f.write(f"Average Anomaly Score: {row['avg_anomaly_score']:.3f}\n")
        f.write(f"Explanation:\n{row['explanation']}\n")
        f.write("-" * 110 + "\n")

print(f"📝 Human-readable report saved → {output_txt}")

print("\n📊 Sample Explanations:")
for _, row in df_sorted.head(3).iterrows():
    print(f"🧩 {row['user_id']}: {row['explanation'][:150]}...")
