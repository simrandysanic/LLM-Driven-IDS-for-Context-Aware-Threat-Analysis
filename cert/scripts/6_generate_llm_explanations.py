"""
Generate Natural-Language Explanations for High-Risk Users
Project: LLM-Driven Intrusion Detection System for Context-Aware Threat Analysis
Author: Simran
"""

import os
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ==== CONFIG ====
TOP_N = 20  # Number of top risky users
MODEL_NAME = "gpt-4o-mini"  # lightweight, cheaper and accurate
API_KEY = os.getenv("OPENAI_API_KEY")

# ==== PATHS ====
base_dir = Path("/home/simran/IDS-Project")
fusion_file = base_dir / "fusion_outputs/fusion_cert_unsw_enhanced.csv"
output_file = base_dir / "fusion_outputs/user_risk_explanations.csv"

if not API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not set. Please run: export OPENAI_API_KEY='your_key_here'")

print("🔹 Loading fused dataset ...")
df = pd.read_csv(fusion_file)
df_sorted = df.sort_values("fusion_risk_score", ascending=False).head(TOP_N)

# ==== Initialize Client ====
client = OpenAI(api_key=API_KEY)

def generate_explanation_llm(row):
    prompt = f"""
    You are a cybersecurity analyst explaining insider threat risk.

    The following metrics describe one user:
    - User ID: {row['user_id']}
    - Fusion Risk Score: {row['fusion_risk_score']:.3f}
    - Behavioral Anomaly Score: {row.get('avg_anomaly_score', 'N/A')}
    - RF Attack Ratio: {row.get('rf_attack_ratio', 'N/A')}
    - XGB Attack Ratio: {row.get('xgb_attack_ratio', 'N/A')}
    - Agreement Rate: {row.get('agreement_rate', 'N/A')}

    Write a concise, 3-sentence explanation describing:
    1. The user's behavioral risk,
    2. The network activity anomalies,
    3. The overall likelihood of this being an insider threat.
    Keep it professional, analytical, and human-readable.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a cybersecurity analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=180,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {str(e)}]"

print(f"🧠 Generating LLM explanations for top {TOP_N} users...")
df_sorted["explanation"] = df_sorted.apply(generate_explanation_llm, axis=1)

# ==== Save results ====
df_sorted.to_csv(output_file, index=False)
print(f"✅ Explanations saved → {output_file}")

print("\n📊 Sample Explanations:")
for _, row in df_sorted.head(5).iterrows():
    print(f"🧩 {row['user_id']}: {row['explanation'][:150]}...")
