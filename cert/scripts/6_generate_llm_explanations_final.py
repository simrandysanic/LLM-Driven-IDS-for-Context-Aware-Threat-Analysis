"""
Generate LLM Explanations for Top 10 Users + Detailed Output
Project: LLM-Driven Intrusion Detection System for Context-Aware Threat Analysis
Author: Simran
"""

import pandas as pd
from pathlib import Path
from openai import OpenAI

# ==== CONFIG ====
TOP_N = 10
MODEL_NAME = "gpt-4o-mini"

# ==== PATHS ====
base_dir = Path("/home/simran/IDS-Project")
fusion_file = base_dir / "fusion_outputs/fusion_cert_unsw_enhanced.csv"
csv_output = base_dir / "fusion_outputs/user_risk_explanations_full.csv"
txt_output = base_dir / "fusion_outputs/user_risk_explanations_full.txt"

# ==== Initialize OpenRouter Client ====
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

print("🔹 Loading fused dataset ...")
df = pd.read_csv(fusion_file)
df_sorted = df.sort_values("fusion_risk_score", ascending=False).head(TOP_N)

# ==== LLM Explanation Function ====
def generate_explanation_llm(row):
    prompt = f"""
    You are a cybersecurity analyst explaining insider threat risk.

    User Details:
    - User ID: {row['user_id']}
    - Fusion Risk Score: {row['fusion_risk_score']:.3f}
    - Behavioral Anomaly Score: {row.get('avg_anomaly_score', 'N/A')}
    - RF Attack Ratio: {row.get('rf_attack_ratio', 'N/A')}
    - XGB Attack Ratio: {row.get('xgb_attack_ratio', 'N/A')}
    - Agreement Rate: {row.get('agreement_rate', 'N/A')}

    Task:
    Provide a short (2–3 sentence) professional explanation covering:
    1. Behavioral risk (from CERT)
    2. Network risk (from UNSW)
    3. Overall threat likelihood.
    Be concise, analytical, and avoid repetition.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a cybersecurity analyst writing clear, factual reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=180
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {str(e)}]"

# ==== Generate Explanations ====
print(f"🧠 Generating LLM explanations for top {TOP_N} users...")
df_sorted["explanation"] = df_sorted.apply(generate_explanation_llm, axis=1)

# ==== Save CSV ====
df_sorted.to_csv(csv_output, index=False)
print(f"✅ Explanations saved → {csv_output}")

# ==== Save Human-Readable TXT ====
with open(txt_output, "w") as f:
    f.write("=== LLM-Generated Insider Threat Explanations ===\n\n")
    for _, row in df_sorted.iterrows():
        f.write(f"User ID: {row['user_id']}\n")
        f.write(f"Fusion Risk Score: {row['fusion_risk_score']:.3f}\n")
        f.write(f"Key Attributes: avg_anomaly_score={row.get('avg_anomaly_score')}, "
                f"rf_attack_ratio={row.get('rf_attack_ratio')}, "
                f"xgb_attack_ratio={row.get('xgb_attack_ratio')}, "
                f"agreement_rate={row.get('agreement_rate')}\n")
        f.write(f"Explanation: {row['explanation']}\n")
        f.write("-" * 90 + "\n")
print(f"📝 Human-readable report saved → {txt_output}")

# ==== Terminal Preview ====
print("\n📊 Sample Explanations:")
for _, row in df_sorted.head(3).iterrows():
    print(f"🧩 {row['user_id']}: {row['explanation'][:200]}...")
