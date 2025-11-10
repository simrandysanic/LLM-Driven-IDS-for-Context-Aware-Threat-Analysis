#!/usr/bin/env python3
import pandas as pd
import os

emb_path = "/home/simran/IDS-Project/cert/processed/cert_user_embeddings.csv"
psy_path = "/home/simran/IDS-Project/cert/psychometric.csv"
out_path = "/home/simran/IDS-Project/cert/processed/cert_fused_embeddings.csv"

print("🔹 Loading embeddings and OCEAN data ...")
emb = pd.read_csv(emb_path)
psy = pd.read_csv(psy_path)

print(f"Embeddings shape: {emb.shape}, Psychometric shape: {psy.shape}")

# Merge on user_id
merged = pd.merge(psy, emb, on="user_id", how="inner")
print(f"✅ Merged shape: {merged.shape}")

# Save
merged.to_csv(out_path, index=False)
print(f"💾 Saved fused user-level dataset → {out_path}")

# Quick sanity check
print("\nSample:")
print(merged.sample(3, random_state=42))
