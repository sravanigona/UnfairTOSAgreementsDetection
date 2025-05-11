"""
Evaluation script for SaulLM fine-tuned model on legal anomaly detection.
"""

import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Load validation dataset ===
val_dataset = load_from_disk("/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/SaulLM:7B/data/val")
print(f"Loaded validation dataset with {len(val_dataset)} examples")

# === Load tokenizer ===
model_name = "Equall/Saul-Instruct-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === Load base + LoRA adapter ===
print("Loading adapter from: /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/SaulLM:7B/saul-7b-anomaly/final")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "/home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/SaulLM:7B/saul-7b-anomaly/final")
model.eval()

# === Prediction function ===
def predict(text):
    prompt = f"<s>[INST] You are a legal expert specializing in consumer protection. Review the following clause from a terms of service agreement:\n\n\"{text}\"\n\nIs this clause anomalous or unfair to consumers? Answer with Yes or No and explain why. [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False
        )

    generated = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    is_anomalous = 1 if "yes" in response.lower() else 0
    return response, is_anomalous

# === Evaluate all examples ===
predictions, true_labels, full_responses = [], [], []

print("Evaluating examples...")
for i, example in enumerate(val_dataset):
    text = example["text"]
    label = example["label"]
    response, pred = predict(text)

    predictions.append(pred)
    true_labels.append(label)
    full_responses.append(response)

    if i < 10:  # First 10 examples
        print(f"Example {i}:")
        print(f"  Text: {text[:100]}...")
        print(f"  True label: {label} ({'Anomalous' if label == 1 else 'Not Anomalous'})")
        print(f"  Prediction: {pred} ({'Anomalous' if pred == 1 else 'Not Anomalous'})")
        print(f"  Full response: '{response}'\n")

# === Metrics ===
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average="binary", zero_division=0
)
cm = confusion_matrix(true_labels, predictions)

# === Distribution Summary ===
anomalous_preds = sum(predictions)
anomalous_true = sum(true_labels)
print(f"Prediction distribution: {anomalous_preds} anomalous, {len(predictions) - anomalous_preds} not anomalous")
print(f"True label distribution: {anomalous_true} anomalous, {len(true_labels) - anomalous_true} not anomalous")

# === Print metrics ===
print("\n=== EVALUATION RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(true_labels, predictions, zero_division=0))

# === TP/FP/TN/FN breakdown ===
tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)

print("\nPrediction Analysis:")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")

# === Save results ===
results_df = pd.DataFrame({
    'text': [ex['text'] for ex in val_dataset],
    'true_label': true_labels,
    'predicted_label': predictions,
    'response': full_responses,
    'correct': [t == p for t, p in zip(true_labels, predictions)]
})
results_df.to_csv('evaluation_results.csv', index=False)

# === Plot confusion matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Anomalous', 'Anomalous'],
            yticklabels=['Not Anomalous', 'Anomalous'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("\nResults saved to evaluation_results.csv")
print("Confusion matrix visualization saved to confusion_matrix.png")