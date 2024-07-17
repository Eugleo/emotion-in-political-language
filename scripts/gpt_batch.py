import datetime
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

import jsonlines
import openai
import polars as pl
import typer
from dotenv import load_dotenv
from polars import col as c

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.emotion import utils

app = typer.Typer()

DEFAULT_BATCH_DIR = Path("data/batch/")


def cost_estimate(texts: list[str], token_chars: int = 4, token_cost=0.000005) -> float:
    return sum(len(text) / token_chars for text in texts) * token_cost


def sample_pairs(
    num: int,
    seed: int,
    word_limit: tuple[float, float],
    path: Path,
) -> pl.DataFrame:
    ids = (
        pl.scan_parquet(path)
        .filter(c("word_count") > word_limit[0], c("word_count") < word_limit[1])
        .select("speech_id")
        .collect()
        .get_column("speech_id")
        .sample(num * 2, seed=seed)
    )

    examples = pl.scan_parquet(path).filter(c("speech_id").is_in(ids)).collect()

    result = pl.DataFrame(
        {
            "speech_1": examples[:num].get_column("text"),
            "speech_1_id": examples[:num].get_column("speech_id"),
            "speech_2": examples[num:].get_column("text"),
            "speech_2_id": examples[num:].get_column("speech_id"),
        }
    )

    return result


def write_batch_file(
    speech_pairs: pl.DataFrame,
    path: Path,
    template: str,
    model: str = "gpt-4o",
):
    texts = []
    with jsonlines.open(path, "w") as writer:
        for row in speech_pairs.iter_rows(named=True):
            content = template.format(
                excerpt_0=row["speech_1"], excerpt_1=row["speech_2"]
            )
            texts.append(content)
            writer.write(
                {
                    "custom_id": f"request-{row['speech_1_id']}-{row['speech_2_id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                    },
                }
            )
    return texts


def parse_response(response_path: Path) -> pl.DataFrame:
    pairs = []
    with jsonlines.open(response_path) as reader:
        for r in reader:
            _, speech_1_id, speech_2_id = r["custom_id"].split("-")
            response = r["response"]["body"]["choices"][0]["message"]["content"]

            result = re.findall(r"result: (-?\d+)", response)
            if len(result) != 1:
                print(f"Unable to parse response: {response}")
                result = -1
            else:
                result = int(result[0])
            pairs.append(
                {
                    "speech_1_id": int(speech_1_id),
                    "speech_2_id": int(speech_2_id),
                    "response": response,
                    "result": result,
                }
            )
    result = pl.DataFrame(pairs)

    return result


PROMPT_TEMPLATE = """
Consider the following two excerpts from two political speeches. Note that they contain spelling mistakes and typos since they are were obtained by OCR.

Excerpt 0: {excerpt_0}

Excerpt 1: {excerpt_1}

The goal is to judge which of these two, if any, is more emotional. Don't take the topic itself into account. Sometimes you will need to read between the lines and infer the emotionality level, e.g. in cases when the text is passive aggresive. Is there any difference between Excerpt 1 and Excerpt 2 in terms of emotionality? If there is not, say so.

Only after you’re done with the analysis, end the message with "result: 0" (without quotes) if Excerpt 0 is the more emotional, or "result: 1" if Excerpt 1 is more emotional. Use "result: -1"  if the result is unclear or there is no large difference.
"""[1:]


@app.command()
def submit(
    num: Annotated[int, typer.Option(min=1)],
    batch_dir: Annotated[
        Path, typer.Option(exists=True, file_okay=False, dir_okay=True)
    ] = DEFAULT_BATCH_DIR,
    model: Annotated[str, typer.Option()] = "gpt-4o",
    data_path: Annotated[Path, typer.Option(file_okay=False, dir_okay=True)] = Path(
        "data/processed/*.parquet"
    ),
    min_words: Annotated[int, typer.Option(min=1)] = 64,
    max_words: Annotated[int, typer.Option(min=1)] = 1024,
    seed: Annotated[int, typer.Option()] = 42,
):
    utils.set_seed(seed)

    id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    batch_dir = batch_dir / id
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = batch_dir / "batch.jsonl"

    pairs = sample_pairs(num, seed, (min_words, max_words), data_path)
    pairs.write_parquet(batch_dir / "input.parquet")
    texts = write_batch_file(
        speech_pairs=pairs,
        path=batch_path,
        template=PROMPT_TEMPLATE,
        model=model,
    )

    print(f"Estimated cost: ${cost_estimate(texts):.2f}")

    load_dotenv()
    client = openai.OpenAI()
    batch_file = client.files.create(file=open(batch_path, "rb"), purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"directory": str(batch_dir)},
    )

    with open(batch_dir / "request.json", "w") as f:
        json.dump(batch.to_dict(), f)

    print(f"Batch {batch.id} created.")


@app.command()
def download(
    batch_dir: Annotated[
        Optional[Path], typer.Argument(exists=True, file_okay=False, dir_okay=True)
    ] = None,
):
    if batch_dir is None:
        batch_dir = max(list(DEFAULT_BATCH_DIR.glob("*")), key=lambda x: x.name)

    with open(batch_dir / "request.json") as f:
        request = json.load(f)

    load_dotenv()
    client = openai.OpenAI()
    batch = client.batches.retrieve(request["id"])
    if batch.status != "completed":
        print(f"Batch {batch.id} is not completed yet.")
        print(batch)
        return
    assert batch.output_file_id is not None

    file_response = client.files.content(batch.output_file_id)

    with open(batch_dir / "response.jsonl", "w") as f:
        f.write(file_response.text)

    results = parse_response(batch_dir / "response.jsonl")
    input_pairs = pl.read_parquet(batch_dir / "input.parquet")
    results = input_pairs.join(results, on=["speech_1_id", "speech_2_id"])
    results.write_parquet(batch_dir / "results.parquet")


if __name__ == "__main__":
    app()
