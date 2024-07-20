import os
import sys
from pathlib import Path
from typing import Annotated

import datasets
import torch
import typer
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.emotion import utils  # noqa: E402


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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (labels == preds).mean()}


app = typer.Typer()


@app.command()
def train(
    dataset_id: Annotated[
        str, typer.Option("--dataset")
    ] = "Eugleo/us-congressional-speeches-emotionality-pairs",
    model_id: Annotated[str, typer.Option("--model")] = "google/gemma-7b",
    trained_model_id: Annotated[
        str, typer.Option("--trained-model")
    ] = "Eugleo/gemma-7b-emotionality",
    seed: Annotated[int, typer.Option()] = 42,
    train_size: Annotated[float, typer.Option()] = 0.9,
    cuda: Annotated[int, typer.Option()] = 0,
    wandb_project: Annotated[str, typer.Option()] = "emotion-in-political-language",
    lora_r: Annotated[int, typer.Option()] = 8,
    lora_alpha: Annotated[int, typer.Option()] = 16,
):
    utils.set_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    os.environ["WANDB_PROJECT"] = wandb_project

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    dataset = datasets.load_dataset(dataset_id)
    assert isinstance(dataset, datasets.DatasetDict)
    train_test = dataset["train"].train_test_split(train_size=train_size, seed=seed)
    dataset = datasets.DatasetDict(
        {"train": train_test["train"], "validation": train_test["test"]}
    ).map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )

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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_length=1024,
        remove_unused_columns=False,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        modules_to_save=["classifier"],
    )

    trainer = RewardTrainer(
        model=model,
        args=training_config,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,  # type: ignore
        compute_metrics=compute_metrics,
    )

    model = trainer.train()

    model.push_to_hub(trained_model_id)


if __name__ == "__main__":
    app()
