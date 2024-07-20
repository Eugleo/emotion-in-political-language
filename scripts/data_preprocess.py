from pathlib import Path
from typing import Annotated

import datasets
import dotenv
import polars as pl
import typer
from rich.progress import Progress

app = typer.Typer()

AMMENDMENT_INDICATORS = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)",
    "(g)",
    "(h)",
    "(i)",
    "(j)",
    "(k)",
    "(l)",
    "(m)",
    "(n)",
    "(o)",
    "(p)",
    "(q)",
    "(r)",
    "(s)",
    "(t)",
    "(u)",
    "(v)",
    "(w)",
    "(y)",
    "(x)",
    "(z)",
    "(A)",
    "(B)",
    "(C)",
    "(D)",
    "(E)",
    "(F)",
    "(G)",
    "(H)",
    "(I)",
    "(J)",
    "(K)",
    "(L)",
    "(M)",
    "(N)",
    "(O)",
    "(P)",
    "(Q)",
    "(R)",
    "(S)",
    "(T)",
    "(U)",
    "(V)",
    "(W)",
    "(Y)",
    "(X)",
    "(Z)",
    "(ii)",
    "(iii)",
    "(iv)",
    "(vi)",
    "(vii)",
    "(viii)",
    "(ix)",
]


def load_metadata(path: Path, progress: Progress) -> pl.DataFrame:
    dataframes = []
    paths = list(path.glob("metadata/descr_*.txt"))
    for speech in progress.track(paths, description="Loading metadata..."):
        with open(speech, encoding="Windows-1252") as f:
            dataframes.append(pl.read_csv(f.read().encode("utf-8"), separator="|"))
    metadata = pl.concat(dataframes).with_columns(
        pl.col("date").cast(pl.String).str.to_date("%Y%m%d").cast(pl.Datetime)
    )
    return metadata


def write_speeches(
    source: Path, destination: Path, metadata: pl.DataFrame, progress: Progress
) -> None:
    paths = source.glob("speeches/speeches_*.txt")
    for i, speech in progress.track(
        list(enumerate(paths)), description="Loading speeches..."
    ):
        with open(speech, encoding="Windows-1252") as f:
            # Skip the first line
            next(f)
            speeches = []
            for line in f:
                id_span = line.find("|")
                speech_id = int(line[:id_span])
                text = line[id_span + 1 :]
                speeches.append({"speech_id": speech_id, "text": text})
            pl.DataFrame(speeches).join(
                metadata, on="speech_id", how="inner"
            ).write_parquet(destination / f"speeches_{i}.parquet")


def upload_to_huggingface(
    dataset: datasets.Dataset, name: str, progress: Progress
) -> None:
    upload_task = progress.add_task(f"Uploading {name} to HuggingFace...", total=None)
    dataset.push_to_hub(f"Eugleo/{name}")
    progress.stop_task(upload_task)


@app.command()
def main(
    original: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
        ),
    ] = Path("data/original"),
    processed: Annotated[
        Path,
        typer.Option(
            writable=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("data/processed"),
    upload: Annotated[bool, typer.Option()] = False,
) -> None:
    if processed.exists():
        should_write = typer.confirm(
            f"{processed} already exists. Do you want to overwrite it?",
            abort=True,
        )
    else:
        should_write = True
        processed.mkdir(exist_ok=True, parents=True)

    with Progress() as progress:
        if should_write:
            metadata = load_metadata(original, progress)
            write_speeches(original, processed, metadata, progress)

        if upload:
            dotenv.load_dotenv()

            dataset = datasets.Dataset.from_parquet(
                processed / "*.parquet", streaming=True
            )
            assert isinstance(dataset, datasets.Dataset)

            upload_to_huggingface(dataset, "Eugleo/us-congressional-speeches", progress)

            partial_dataset = dataset.filter(
                lambda s: 64 / 1.5 < s["word_count"] < 1024 / 1.5
            ).filter(
                lambda s: not any(
                    ammendment in s["text"] for ammendment in AMMENDMENT_INDICATORS
                )
            )
            upload_to_huggingface(
                partial_dataset, "Eugleo/us-congressional-speeches-subset", progress
            )


if __name__ == "__main__":
    app()
