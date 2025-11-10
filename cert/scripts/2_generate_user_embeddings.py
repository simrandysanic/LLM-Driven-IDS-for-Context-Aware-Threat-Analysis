#!/usr/bin/env python3
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# Paths
input_path = "/home/simran/IDS-Project/cert/processed/cert_user_corpus.csv"
output_csv = "/home/simran/IDS-Project/cert/processed/cert_user_embeddings.csv"
output_pt  = "/home/simran/IDS-Project/cert/processed/cert_user_embeddings.pt"

print("🔹 Loading user corpus ...")
df = pd.read_csv(input_path)
print(f"Users: {len(df)}")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# Compute embeddings
embeddings = []
for text in tqdm(df["user_corpus"], desc="Encoding users"):
    emb = model.encode(text, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    embeddings.append(emb.cpu())

# Stack all embeddings
emb_tensor = torch.stack(embeddings)
print("Embedding shape:", emb_tensor.shape)

# Save outputs
torch.save(emb_tensor, output_pt)
df_out = pd.DataFrame(emb_tensor.numpy())
df_out.insert(0, "user_id", df["user_id"])
df_out.to_csv(output_csv, index=False)

print(f"✅ Saved embeddings to:\n - {output_pt}\n - {output_csv}")
