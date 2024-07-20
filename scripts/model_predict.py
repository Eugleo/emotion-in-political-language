# %%
#!%load_ext autoreload
#!%autoreload 2

import csv
import datetime
from pathlib import Path

import datasets
import polars as pl
import torch
import torch.nn.functional as F
from peft import PeftModel
from polars import col as c
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# %%
model_id = "google/flan-t5-large"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
model = PeftModel.from_pretrained(model, "Eugleo/results").to(device)

# Switch to bf16
model = model.to(torch.bfloat16)

# Apply torch.compile() for optimization
# model = torch.compile(model)

model.eval()

# %%
dataset = datasets.load_dataset(
    "Eugleo/us-congressional-speeches", streaming=True
).filter(lambda x: 64 / 1.5 < x["word_count"] < 1024 / 1.5)

# %%
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


def collate_fn(batch):
    return {
        "speech_id": [item["speech_id"] for item in batch],
        "date": [item["date"] for item in batch],
        "chamber": [item["chamber"] for item in batch],
        "text": [item["text"] for item in batch],
    }


@torch.inference_mode()
def get_predictions(texts, device):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(
        device
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    probabilities = outputs.logits.softmax(dim=-1)[:, 0]
    return probabilities


if Path("predictions.csv").exists():
    processed_ids = set(
        pl.scan_csv("predictions.csv")
        .select("speech_id")
        .collect()["speech_id"]
        .to_list()
    )
    dataset_subset = dataset.filter(lambda x: x["speech_id"] not in processed_ids)
else:
    dataset_subset = dataset


# Create DataLoader
dataloader = DataLoader(
    dataset_subset["train"],
    batch_size=128,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4,
)


total = 16_063
remaining = round(total - (len(processed_ids) / 128))

with torch.inference_mode(), open("predictions.csv", "w") as f:
    writer = csv.DictWriter(
        f, fieldnames=["speech_id", "date", "chamber", "probability"]
    )
    writer.writeheader()
    # Process the dataset
    for batch in tqdm(dataloader, desc="Processing batches...", total=remaining):
        probs = get_predictions(batch["text"], device)

        for speech_id, date, chamber, probability in zip(
            batch["speech_id"], batch["date"], batch["chamber"], probs
        ):
            writer.writerow(
                {
                    "speech_id": speech_id,
                    "date": str(date),
                    "chamber": chamber,
                    "probability": probability.item(),
                }
            )
# %%
from lets_plot import *

LetsPlot.setup_html()

predictions_df = pl.scan_csv("predictions.csv").with_columns(
    c("date").str.to_datetime(format="%Y-%m-%d %H:%M:%S"),
    c("probability").alias("emotionality_score"),
)


timeseries = (
    predictions_df.group_by("chamber", "date")
    .agg(c("emotionality_score").mean().alias("mean_emotionality_score"))
    .collect()
)

# Calculate weekly averages and standard deviations
weekly_stats = predictions_df.with_columns(
    c("probability").rolling_mean_by("date", "1y").alias("emotionality"),
).collect()

# Create the plot
(
    ggplot(weekly_stats, aes(x="date"))
    + geom_line(aes(y="emotionality"))
    + scale_x_datetime(format="%Y")
    + labs(
        x="Year",
        y="Emotionality Score",
        title="Weekly Average Emotionality of Speeches with Standard Deviation",
        color="Chamber",
        fill="Chamber",
    )
).show()
