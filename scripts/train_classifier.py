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
import torch
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
train_size = 0.9

model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


labeled_data_path = "/mnt/ssd-1/mechinterp/gw1/evzen-test/data/batch/2024-07-17-12-17-14/results.parquet"

utils.set_seed(seed)

data = datasets.Dataset.from_parquet(labeled_data_path)

assert isinstance(data, datasets.Dataset)

train_test = data.train_test_split(train_size=0.9, seed=seed)


dataset = datasets.DatasetDict(
    {"train": train_test["train"], "validation": train_test["test"]}
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_PROJECT"] = "emotion-in-political-language"

training_config = RewardConfig(
    output_dir="results",
    save_total_limit=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    logging_steps=10,
    bf16=True,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_length=1024,
    remove_unused_columns=False,
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    modules_to_save=["classifier"],
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
model = trainer.train()

# %%
