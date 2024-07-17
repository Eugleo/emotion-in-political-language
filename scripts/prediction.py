# %%
#!%load_ext autoreload
#!%autoreload 2

import datasets
import polars as pl
import torch
from peft import PeftModel
from polars import col as c
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# %%
model_id = "google/flan-t5-large"

model = AutoModel.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, "Eugleo/results")

model.eval()

# %%
data_path = "/Users/eugen/Downloads/Projects/emotion-in-political-language/data/processed/*.parquet"

data = pl.read_parquet(data_path)

dataset = datasets.Dataset.from_parquet(
    "/Users/eugen/Downloads/Projects/emotion-in-political-language/data/processed/*.parquet",
    streaming=True,
)


# %%


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


def collate_fn(batch):
    return {
        "speech_id": [item["speech_id"] for item in batch],
        "text": [item["text"] for item in batch],
    }


@torch.inference_mode()
def get_predictions(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = outputs.logits[:, 0]
    return probabilities


# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Create a list to store the results
results = []

# Process the dataset
for batch in tqdm(dataloader):
    probabilities = get_predictions(batch["text"])

    results += [
        {"speech_id": speech_id, "prob_emotional": prob}
        for speech_id, prob in zip(batch["speech_id"], probabilities)
    ]


# Convert results to a Polars DataFrame
predictions_df = pl.DataFrame(results)

# Convert the IterableDataset to a Polars DataFrame

# Join predictions with the original dataset
final_df = data.join(predictions_df, on="speech_id")

# %%
from lets_plot import *

timeseries = final_df.group_by("chamber", "date").agg(
    c("prob_emotional").mean().alias("mean_prob_emotional"),
    (c("prob_emotional") > 0.5).sum().alias("count_emotional"),
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
