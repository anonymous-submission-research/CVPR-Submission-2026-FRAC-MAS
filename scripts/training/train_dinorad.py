#!/usr/bin/env python
# coding: utf-8

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

model_id = "microsoft/rad-dino"
dataset_root = os.path.join(_PROJECT_ROOT, 'balanced_augmented_dataset')

# --- 2. Load Dataset ---
dataset = load_dataset(
    "imagefolder", 
    data_files={
        "train": os.path.join(dataset_root, "train/**"),
        "validation": os.path.join(dataset_root, "val/**"),
        "test": os.path.join(dataset_root, "test/**")
    }
)

labels = dataset["train"].features["label"].names
label2id, id2label = {label: i for i, label in enumerate(labels)}, {i: label for i, label in enumerate(labels)}

print(f"Classes: {labels}")

# --- 3. Preprocessing ---
image_processor = AutoImageProcessor.from_pretrained(model_id)

def transform(example_batch):
    """Apply the DINO-RAD required preprocessing to images."""
    inputs = image_processor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

prepared_ds = dataset.with_transform(transform)

# --- 4. Load Model ---
model = AutoModelForImageClassification.from_pretrained(
    model_id, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# --- 5. Metrics ---
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# --- 6. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./outputs/dinorad_lib",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=16,
    num_train_epochs=3, 
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
)

# --- 7. Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

# --- 8. Train! ---
print("Starting training on DINO-RAD...")
trainer.train()

# Final Test Evaluation
print("Evaluating on test set...")
test_metrics = trainer.evaluate(prepared_ds["test"])
print(f"Final Test Accuracy: {test_metrics['eval_accuracy']:.2%}")
