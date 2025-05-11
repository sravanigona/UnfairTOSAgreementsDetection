"""
Fine-tuning script for Equall/Saul-7B-Base on anomaly detection in legal clauses using LoRA with 4-bit quantization.
"""
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import transformers
from glob import glob
import re

# Check versions
print("Transformers version:", transformers.__version__)
print("TrainingArguments path:", TrainingArguments.__module__)

# Load dataset (swap due to imbalance)
val_dataset = load_from_disk("./data/train")
train_dataset = load_from_disk("./data/val")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Format prompts
def format_prompt(example):
    prompt = f"<s>[INST] You are a legal expert specializing in consumer protection. Review the following clause from a terms of service agreement:\n\n\"{example['text']}\"\n\nIs this clause anomalous or unfair to consumers? Answer with Yes or No and explain why. [/INST]"
    label = " Yes, this clause is anomalous because it unfairly restricts consumer rights." if example["label"] == 1 else " No, this clause is standard and not unfair to consumers."
    return {"text": prompt + label, "label": example["label"]}

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)

# Load tokenizer
model_name = "Equall/Saul-7B-Base"
print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Ensure consistent padding

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization
print(f"Loading {model_name} model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Let the library manage GPU allocation
    trust_remote_code=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize inputs
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokens = {k: v.squeeze(0) for k, v in tokens.items()}
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Evaluation metrics
def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_labels, true_labels = [], []
    for pred, label in zip(predictions, labels):
        valid_indices = np.where(label != -100)[0]
        if len(valid_indices) == 0: continue
        last_idx = valid_indices[-1]
        pred_token_id = np.argmax(pred[last_idx])
        label_token_id = label[last_idx]
        pred_text = tokenizer.decode([pred_token_id]).lower()
        label_text = tokenizer.decode([label_token_id]).lower()
        pred_labels.append(1 if "yes" in pred_text else 0)
        true_labels.append(1 if "yes" in label_text else 0)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary", zero_division=0)
    acc = accuracy_score(true_labels, pred_labels)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

from transformers import Trainer
import torch

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.pop("labels", None)
#         outputs = model(**inputs)
#         loss = outputs.loss if labels is not None else outputs[0]
#         return loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss



# Output directory
output_dir = "/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/SaulLM:7B/saul-7b-anomaly"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,           
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=50,
    eval_steps=200,          
    save_steps=200,        
    save_strategy="steps",
    evaluation_strategy="epoch",
    report_to="none", 
    gradient_checkpointing=True,  
    fp16=True,
    label_names=["labels"],  
    optim="paged_adamw_32bit", 
    disable_tqdm=True ,  
    load_best_model_at_end=True,
    metric_for_best_model="f1"    
)


# Train
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")


def get_latest_checkpoint(output_dir):
    checkpoints = glob(f"{output_dir}/checkpoint-*")
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0]))
    return checkpoints[-1]


print("Starting training...")
latest_ckpt = get_latest_checkpoint(output_dir)
if latest_ckpt:
    print(f"Resuming from checkpoint: {latest_ckpt}")
    trainer.train(resume_from_checkpoint=latest_ckpt)
else:
    print("No checkpoints found. Starting fresh.")
    trainer.train()

# Save final model
print("Saving model...")
trainer.save_model(f"{output_dir}/final")
model.save_pretrained(f"{output_dir}/final")


print("Saving model...")
trainer.save_model(f"{output_dir}/final")
model.save_pretrained(f"{output_dir}/final")