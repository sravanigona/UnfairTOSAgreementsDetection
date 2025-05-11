import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import transformers

# === Basic setup logs ===
print("Transformers version:", transformers.__version__)
print("TrainingArguments path:", TrainingArguments.__module__)

# === Load datasets (swapped val/train intentionally) ===
val_dataset = load_from_disk("/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/TinyLlama/data_cleaned/val")
train_dataset = load_from_disk("/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/TinyLlama/data_cleaned/train")

# === Show class distribution ===
print("\n Training set label distribution:")
print(Counter(train_dataset['label']))
print("\n Validation set label distribution:")
print(Counter(val_dataset['label']))

# === Format prompt ===
def format_prompt(example):
    prompt = f"<s>[CLAUSE]: {example['text']} \n[Is this anomalous?]:"
    label = " Yes" if example["label"] == 1 else " No"
    return {"text": prompt , "label": example["label"]}

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)


# === Show class distribution ===
print("\n Training set after formatting prompt label distribution:")
print(Counter(train_dataset['label']))
print("\n Validation set after formatting prompt label distribution:")
print(Counter(val_dataset['label']))

# === Tokenizer ===
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# === Load model ===
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# === Tokenization ===
def tokenize(example):
    inputs = tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = inputs["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# === Metrics ===
def compute_metrics(eval_preds):
    predictions = np.argmax(eval_preds.predictions, axis=-1)
    labels = eval_preds.label_ids

    pred_labels = []
    true_labels = []
    for pred_ids, label_ids in zip(predictions, labels):
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).lower()
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True).lower()
        pred_labels.append(1 if "yes" in pred_text else 0)
        true_labels.append(1 if "yes" in label_text else 0)

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="./lora-tinyllama-cleaned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    eval_steps=500,            # Evaluate every 500 steps
    save_steps=500,            # Save every 500 steps
    do_eval=True,              # Enable evaluation
    logging_steps=10,
    report_to="none",
    fp16=True
)


# === Train ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n Starting fine-tuning...")
trainer.train()

# === Save Final Adapter ===
final_output_dir = "./final"
print(f"\n Saving final model to {final_output_dir}")
os.makedirs(final_output_dir, exist_ok=True)
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print("\n Done! Fine-tuned model saved.")