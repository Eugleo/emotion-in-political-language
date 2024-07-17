# %%
#!%load_ext autoreload
#!%autoreload 2

import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

import datasets
import jsonlines
import openai
import polars as pl
import typer
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.emotion import utils

# %%
seed = 42
train_size = 4000  # 80% of the data, i.e. 4 000 examples
val_size = 500  # 10% of the data, i.e. 400 examples
test_size = 5000 - train_size - val_size

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


labeled_data_path = "/Users/eugen/Downloads/Projects/emotion-in-political-language/data/batch/2024-07-17-12-17-14/results.parquet"

utils.set_seed(seed)

data = datasets.Dataset.from_parquet(labeled_data_path)

assert isinstance(data, datasets.Dataset)

train_val_test = data.train_test_split(test_size=val_size, seed=seed)

# Split the train+validation set into train and validation
train_val_split = train_val_test["train"].train_test_split(
    test_size=test_size, seed=seed
)

dataset = datasets.DatasetDict(
    {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_val_test["test"],
    }
)


def tokenize(batch, *, tokenizer):
    chosen, rejected = [], []
    for i in range(len(batch["result"])):
        pref = batch["result"][i]
        if pref == -1:
            continue
        chosen.append(batch["speech_1"][i] if pref == 0 else batch["speech_2"][i])
        rejected.append(batch["speech_2"][i] if pref == 0 else batch["speech_1"][i])

    chosen_tokenized = tokenizer(chosen, truncation=True)
    rejected_tokenized = tokenizer(rejected, truncation=True)

    return {
        "input_ids_chosen": chosen_tokenized["input_ids"],
        "attention_mask_chosen": chosen_tokenized["attention_mask"],
        "input_ids_rejected": rejected_tokenized["input_ids"],
        "attention_mask_rejected": rejected_tokenized["attention_mask"],
    }


dataset = dataset.map(
    tokenize,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=data.column_names,
)


# %%
os.environ["WANDB_PROJECT"] = "emotion-in-political-language"

training_config = RewardConfig(
    output_dir="results",
    save_total_limit=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_length=1024,
    remove_unused_columns=False,
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (labels == preds).mean()}


trainer = RewardTrainer(
    model=model,
    args=training_config,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,  # type: ignore
    compute_metrics=compute_metrics,
)

# %%
trainer.train()
