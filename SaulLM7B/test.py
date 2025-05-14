import os
import re
import torch
import numpy as np
from glob import glob
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1) Load raw datasets
raw_train = load_from_disk("./data/val")
raw_val   = load_from_disk("./data/train")
print(f"Raw train size: {len(raw_train)}")
print(f"Raw  val  size: {len(raw_val)}")

# 2) Prompt formatting (exactly once)
def format_prompt(example):
    prompt = (
        "<s>[INST] You are a legal expert specializing in consumer protection.\n"
        f"Review this clause:\n\n\"{example['text']}\"\n\n"
        "Is it anomalous or unfair? Answer Yes or No and explain briefly. [/INST]"
    )
    label_str = (
        " Yes, this clause is anomalous because it limits consumer rights."
        if example["label"] == 1
        else " No, this clause is standard and not unfair."
    )
    return {"text": prompt + label_str, "label": example["label"]}

train_ds = raw_train.map(format_prompt, remove_columns=raw_train.column_names)
val_ds   = raw_val.map(  format_prompt, remove_columns=raw_val.column_names)
print(f"Formatted train size: {len(train_ds)}")
print(f"Formatted  val size: {len(val_ds)}")

# 3) Tokenizer & model setup
model_name = "Equall/Saul-7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()  # ~0.2% of 7B

# 4) Tokenization (shorter max_length for speed)
def tokenize_fn(example):
    toks = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    # squeeze batch dim and set labels
    input_ids = toks.input_ids.squeeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": toks.attention_mask.squeeze(0),
        "labels": input_ids.clone()
    }

train_tok = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
val_tok   = val_ds.  map(tokenize_fn, remove_columns=val_ds.column_names)

# 5) Metrics
def compute_metrics(pred):
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = pred.label_ids

    preds, trues = [], []
    for logit, lab in zip(logits, labels):
        valid = np.where(lab != -100)[0]
        if not len(valid):
            continue
        last = valid[-1]
        pred_id = int(logit[last].argmax())
        true_id = int(lab[last])
        pred_txt = tokenizer.decode([pred_id]).lower()
        true_txt = tokenizer.decode([true_id]).lower()
        preds.append(1 if "yes" in pred_txt else 0)
        trues.append(1 if "yes" in true_txt else 0)

    prec, rec, f1, _ = precision_recall_fscore_support(
        trues, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(trues, preds)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# 6) TrainingArguments
output_dir = "/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/SaulLM:7B/saul-7b-anomaly"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    evaluation_strategy="epoch",
    eval_steps=200,
    save_strategy="epoch",
    save_steps=200,

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    warmup_ratio=0.1,
    weight_decay=0.01,

    logging_steps=50,
    report_to="none",
    disable_tqdm=True,

    optim="paged_adamw_32bit",
    fp16=True,
    gradient_checkpointing=True,
    label_names=["labels"],
)

# 7) Trainer & train
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Resume from last checkpoint if present
def latest_ckpt(dir):
    c = glob(f"{dir}/checkpoint-*")
    return max(c, key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0])) if c else None

ckpt = latest_ckpt(output_dir)
if ckpt:
    print(f"Resuming from {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
else:
    print("Starting fresh training")
    trainer.train()

# 8) Save final
trainer.save_model(f"{output_dir}/final")
model.save_pretrained(f"{output_dir}/final")