"""
Fusion of CERT (Behavioral) + UNSW (Network) Anomaly Metrics
Author: Simran
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ===== Paths =====
base_dir = Path("/home/simran/IDS-Project")
cert_path = base_dir / "cert/processed/cert_user_anomaly_scores.csv"
unsw_path = base_dir / "cert/processed/unsw_predictions_full.csv"
out_dir = base_dir / "fusion_outputs"
out_dir.mkdir(exist_ok=True)
fusion_file = out_dir / "fusion_cert_unsw_enhanced.csv"

print("🔹 Loading datasets ...")
cert_df = pd.read_csv(cert_path)
unsw_df = pd.read_csv(unsw_path)

print("CERT shape:", cert_df.shape)
print("UNSW shape:", unsw_df.shape)

# ===== Summarize UNSW data globally =====
print("📊 Computing UNSW network-level metrics ...")
unsw_summary = pd.DataFrame({
    "rf_attack_ratio": [unsw_df["rf_pred"].mean()],
    "xgb_attack_ratio": [unsw_df["xgb_pred"].mean()],
    "rf_avg_confidence": [unsw_df["rf_prob"].mean()],
    "xgb_avg_confidence": [unsw_df["xgb_prob"].mean()],
    "agreement_rate": [unsw_df["agree"].mean()],
})

print("UNSW summary stats:\n", unsw_summary.T)

# ===== Normalize UNSW metrics =====
scaler = StandardScaler()
unsw_summary_scaled = pd.DataFrame(
    scaler.fit_transform(unsw_summary),
    columns=unsw_summary.columns
)
# Repeat these global stats for all CERT users
unsw_repeated = pd.concat([unsw_summary_scaled] * len(cert_df), ignore_index=True)

# ===== Merge =====
fusion_df = pd.concat([cert_df.reset_index(drop=True), unsw_repeated], axis=1)
print("🔗 Fusion complete. Shape:", fusion_df.shape)

# ===== Risk Scoring =====
fusion_df["fusion_risk_score"] = (
    -fusion_df["avg_anomaly_score"]
    + fusion_df[["rf_attack_ratio", "xgb_attack_ratio", "agreement_rate"]].sum(axis=1)
)
fusion_df["fusion_risk_score"] = StandardScaler().fit_transform(
    fusion_df[["fusion_risk_score"]]
)

# ===== Save =====
fusion_df.to_csv(fusion_file, index=False)
print(f"✅ Saved fused dataset → {fusion_file}")

# ===== Summary =====
top10 = fusion_df.sort_values("fusion_risk_score", ascending=False).head(10)
print("\n🔥 Top 10 High-Risk Users:")
print(top10[["user_id", "fusion_risk_score"]])
