# %%
#!%load_ext autoreload
#!%autoreload 2

import datetime
from pathlib import Path

import datasets
import jsonlines
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
model = torch.compile(model)

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


# Create DataLoader
dataloader = DataLoader(dataset["train"], batch_size=128, collate_fn=collate_fn)

# Create a list to store the results
speech_ids, dates, chambers, probabilities = [], [], [], []

with torch.inference_mode(), jsonlines.open("predictions.jsonl", "w") as writer:
    # Process the dataset
    for batch in tqdm(dataloader):
        probs = get_predictions(batch["text"], device)

        speech_ids.extend(batch["speech_id"])
        dates.extend(batch["date"])
        chambers.extend(batch["chamber"])
        probabilities.extend(probs.tolist())

        for speech_id, date, chamber, probability in zip(
            speech_ids, dates, chambers, probabilities
        ):
            writer.write(
                {
                    "speech_id": speech_id,
                    "date": str(date),
                    "chamber": chamber,
                    "probability": probability,
                }
            )
# %%
from lets_plot import *

LetsPlot.setup_html()

predictions_df = pl.scan_ndjson(
    "predictions.jsonl",
    schema_overrides={"date": pl.Datetime},
)


timeseries = predictions_df.group_by("chamber", "date").agg(
    c("probability").mean().alias("mean_prob_emotional"),
    (c("probability") > 0.5).sum().alias("count_emotional"),
)

(
    ggplot(timeseries)
    + geom_line(aes(x="date", y="mean_prob_emotional", color="chamber"))
    + ggtitle("Mean emotionality of speeches each day")
).show()

(
    ggplot(timeseries)
    + geom_line(aes(x="date", y="count_emotional", color="chamber"))
    + ggtitle("Number of emotional speeches each day")
).show()
