from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load and balance the dataset
NUM_PER_CLASS = 750  # Total = 150; Will later split into train/val/test (60/20/20)

ds = load_dataset("LawInformedAI/claudette_tos")
full_data = ds["train"]

# Filter and balance manually
class_0 = full_data.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(NUM_PER_CLASS))
class_1 = full_data.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(NUM_PER_CLASS))
balanced_full = concatenate_datasets([class_0, class_1]).shuffle(seed=42)

# Step 2: Manual split: 60% train, 20% val, 20% test
split_1 = balanced_full.train_test_split(test_size=0.4, seed=42)
split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": split_1["train"],       # 60%
    "validation": split_2["train"],  # 20%
    "test": split_2["test"]          # 20%
})

# Step 3: Tokenization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_ds = dataset.map(preprocess, batched=True)

# Step 4: Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 5: Define metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Step 6: Trainer setup
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Step 7: Train
trainer.train()

# Step 8: Final test evaluation
eval_results = trainer.evaluate(tokenized_ds["test"])
print("\nEvaluation Results on Test Set:")
print(eval_results)

# Step 9: Extra metrics
predictions = trainer.predict(tokenized_ds["test"])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["fair", "unfair"]))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds))

# Step 10: Save model
model.save_pretrained("./finetuned-legal-bert")
tokenizer.save_pretrained("./finetuned-legal-bert")
