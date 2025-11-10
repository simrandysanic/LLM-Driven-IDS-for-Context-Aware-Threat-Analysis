#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import os

data_path = "/home/simran/IDS-Project/cert/processed/cert_fused_embeddings.csv"
out_csv = "/home/simran/IDS-Project/cert/processed/cert_user_anomaly_scores.csv"

print("🔹 Loading fused dataset ...")
df = pd.read_csv(data_path)
print("Shape:", df.shape)

# Select numeric features only (OCEAN + embeddings)
features = df.select_dtypes(include='number')
user_ids = df["user_id"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA for visualization
print("🔹 Running PCA ...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Isolation Forest
print("🔹 Training Isolation Forest ...")
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_scaled)
try:
    iso_scores = -iso.score_samples(X_scaled)  # newer sklearn
except AttributeError:
    iso_scores = -iso._score_samples(X_scaled)  # fallback for older versions

# One-Class SVM
print("🔹 Training One-Class SVM ...")
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
svm_scores = -ocsvm.fit(X_scaled).score_samples(X_scaled)

# Combine results
df_out = pd.DataFrame({
    "user_id": user_ids,
    "IF_score": iso_scores,
    "OCSVM_score": svm_scores,
    "avg_anomaly_score": (iso_scores + svm_scores) / 2
})

df_out = df_out.sort_values("avg_anomaly_score", ascending=False)
df_out.to_csv(out_csv, index=False)

print(f"✅ Saved anomaly scores to: {out_csv}")
print("\nTop 10 anomalous users:")
print(df_out.head(10))

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df_out["avg_anomaly_score"], cmap="viridis", s=20)
plt.colorbar(label="Anomaly Score")
plt.title("User Behavior Anomaly Map (CERT)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.tight_layout()
plt.savefig("/home/simran/IDS-Project/cert/processed/cert_anomaly_map.png", dpi=300)
print("📊 Saved visualization as cert_anomaly_map.png")
