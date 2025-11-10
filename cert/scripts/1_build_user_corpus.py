import pandas as pd
from tqdm import tqdm

EMAIL_PATH = "/home/simran/IDS-Project/cert/email.csv"
OUTPUT_PATH = "/home/simran/IDS-Project/cert/processed/cert_user_corpus.csv"

print("🔹 Loading CERT email data ...")
df = pd.read_csv(EMAIL_PATH)
print(f"Shape: {df.shape}, Unique users: {df['user'].nunique()}")

# Replace NaNs with empty strings
df = df.fillna("")

# Concatenate all relevant text columns
df["combined"] = df[["to", "cc", "bcc", "from", "content"]].agg(" ".join, axis=1)

# Aggregate by user
print("🔹 Aggregating emails per user ...")
user_text = df.groupby("user")["combined"].apply(lambda x: " ".join(x)).reset_index()
user_text.columns = ["user_id", "user_corpus"]

print(f"✅ Created corpus for {len(user_text)} users.")
user_text.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Saved to {OUTPUT_PATH}")
