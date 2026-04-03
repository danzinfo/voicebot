import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ===========================
# 1. Load the dataset
# ===========================
df = pd.read_csv("data/train_expanded_1000.csv")

# Check label distribution
print("Dataset label distribution:\n", df['intent'].value_counts())

# Encode labels
labels = sorted(df["intent"].unique().tolist())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["intent"].map(label2id)

# ===========================
# 2. Train-test split
# ===========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]  # ensures each intent is represented
)

print("\nTest set label distribution:\n", test_df['intent'].value_counts())

# ===========================
# 3. Convert to HuggingFace Dataset
# ===========================
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

# ===========================
# 4. Tokenizer
# ===========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ===========================
# 5. Load Model
# ===========================
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ===========================
# 6. Training Arguments
# ===========================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,  # increase epochs for better convergence
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True
)

# ===========================
# 7. Metrics Function
# ===========================
def compute_metrics(eval_pred):
    logits, labels_ = eval_pred
    preds = np.argmax(logits, axis=1)

    report = classification_report(labels_, preds, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    
    return {
        "accuracy": report["accuracy"],
        "f1": report["weighted avg"]["f1-score"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"]
    }

# ===========================
# 8. Trainer
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ===========================
# 9. Train
# ===========================
trainer.train()

# ===========================
# 10. Evaluate
# ===========================
predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=1)
true = predictions.label_ids

print("\nUnique true labels in test set:", np.unique(true))
print("Total labels:", len(labels))

print("\nClassification Report:\n")
print(classification_report(
    true,
    preds,
    labels=list(range(len(labels))),
    target_names=labels,
    zero_division=0  # avoids UndefinedMetricWarning
))

# Confusion Matrix
cm = confusion_matrix(true, preds)

os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")

# ===========================
# 11. Save Model
# ===========================
save_path = "models/intent_model"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel saved to {save_path}")




