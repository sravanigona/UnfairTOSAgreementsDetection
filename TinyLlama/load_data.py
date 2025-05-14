from datasets import load_dataset, Dataset
import pandas as pd

# Load and prepare the dataset
ds = load_dataset("LawInformedAI/claudette_tos")
df = pd.DataFrame(ds["train"])

# Balance the dataset
anomalous = df[df['label'] == 1].sample(n=1000, random_state=42)
normal = df[df['label'] == 0].sample(n=1000, random_state=42)
balanced_df = pd.concat([anomalous, normal]).sample(frac=1, random_state=42).reset_index(drop=True)


# Assume balanced_df is already shuffled and balanced as in your code
total = len(balanced_df)
train_end = int(0.7 * total)
test_end = int(0.9 * total)  # 70% + 20% = 90%

# Split
train_df = balanced_df[:train_end]
test_df = balanced_df[train_end:test_end]
val_df = balanced_df[test_end:]

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

# Save datasets
train_dataset.save_to_disk("data_cleaned/train")
test_dataset.save_to_disk("data_cleaned/test")
val_dataset.save_to_disk("data_cleaned/val")

# # Split
# train_df, val_df = balanced_df[:80], balanced_df[80:]
# train_dataset = Dataset.from_pandas(train_df)
# val_dataset = Dataset.from_pandas(val_df)

# # Save datasets to disk (for run_finetune.py)
# train_dataset.save_to_disk("data/train")
# val_dataset.save_to_disk("data/val")