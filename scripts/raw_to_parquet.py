from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.progress import Progress

app = typer.Typer()


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
            pl.DataFrame(
                [
                    # We need to do this since the text sometimes contains the separator
                    {"speech_id": int(line[:9]), "text": line[10:]}
                    for line in f
                    if not any(
                        ammendment in line for ammendment in AMMENDMENT_INDICATORS
                    )
                ]
            ).join(metadata, on="speech_id", how="inner").write_parquet(
                destination / f"speeches_{i}.parquet"
            )


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
) -> None:
    if processed.exists():
        typer.confirm(
            f"{processed} already exists. Do you want to overwrite it?",
            abort=True,
        )
    else:
        processed.mkdir(exist_ok=True, parents=True)

    with Progress() as progress:
        metadata = load_metadata(original, progress)
        write_speeches(original, processed, metadata, progress)


if __name__ == "__main__":
    app()
