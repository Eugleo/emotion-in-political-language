import os
import sys
from math import ceil
from pathlib import Path
from typing import Annotated

import datasets
import jsonlines
import polars as pl
import torch
import typer
from dotenv import load_dotenv
from peft import PeftModel
from rich.progress import track
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.emotion import utils  # noqa: E402


def collate_fn(batch):
    return {
        "speech_id": [item["speech_id"] for item in batch],
        "date": [item["date"] for item in batch],
        "chamber": [item["chamber"] for item in batch],
        "text": [item["text"] for item in batch],
    }


@torch.inference_mode()
def get_score(model, tokenizer, texts, device):
    inputs = tokenizer(texts, truncation=True, return_tensors="pt").to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    score = outputs.logits.sigmoid()
    return score


app = typer.Typer()


@app.command()
def train(
    num: Annotated[int, typer.Argument()] = 5000,
    output_dir: Annotated[Path, typer.Argument(dir_okay=True, file_okay=False)] = Path(
        "data/predictions"
    ),
    batch_size: Annotated[int, typer.Option()] = 1,
    dataset_id: Annotated[
        str, typer.Option("--dataset")
    ] = "Eugleo/us-congressional-speeches-subset",
    base_model_id: Annotated[str, typer.Option("--base-model")] = "google/gemma-2-9b",
    trained_model_id: Annotated[
        str, typer.Option("--model")
    ] = "Eugleo/gemma-7b-emotionality",
    seed: Annotated[int, typer.Option()] = 42,
    cuda: Annotated[int, typer.Option()] = 0,
):
    utils.set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, device_map=device, torch_dtype=torch.bfloat16, num_labels=1
    )
    model = PeftModel.from_pretrained(base_model, trained_model_id)
    model = base_model
    model.eval()

    dataset = datasets.load_dataset(dataset_id, split="train")
    assert isinstance(dataset, datasets.Dataset)

    subset = torch.randint(0, len(dataset), (num,)).tolist()

    results = []
    if (output_dir / "predictions.jsonl").exists():
        processed = pl.read_ndjson(output_dir / "predictions.jsonl")
        results = processed.to_dicts()
        processed_ids = set(processed["speech_id"])
        subset = [i for i in subset if i not in processed_ids]

    remaining = ceil(len(subset) / batch_size)

    loader = DataLoader(
        dataset.select(subset),  # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    with (
        torch.inference_mode(),
        jsonlines.open(output_dir / "predictions.jsonl", "w") as writer,
    ):
        # Process the dataset
        for batch in track(loader, total=remaining):
            batch_score = get_score(model, tokenizer, batch["text"], device)
            for speech_id, date, chamber, score in zip(
                batch["speech_id"], batch["date"], batch["chamber"], batch_score
            ):
                result = {
                    "speech_id": speech_id,
                    "date": date,
                    "chamber": chamber,
                    "score": score.item(),
                }
                writer.write(result | {"date": str(date)})
                results.append(result)


if __name__ == "__main__":
    app()
