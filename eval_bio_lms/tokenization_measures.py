"""
What can we measure with just the tokenizer?

* total token counts from MIMIC III
* TODO: add specific biomedical entities

"""
import multiprocessing
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import typer

from eval_bio_lms.model_definitions import MODEL_DEFS
from eval_bio_lms.preprocessing import tokenize_map
from eval_bio_lms.dataset_loaders import mimic_noteevents


def count_tokens_map(examples):
    return {"num_tokens": [len(el) for el in examples["input_ids"]]}


def main(
    note_events_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to NOTEEVENTS.csv.gz",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        help="Number of samples to use. If not set, use all samples."
    ),
    text_col: str = typer.Option(
        "text",
        help="Name of text column."
    ),
    num_proc: int = typer.Option(
        multiprocessing.cpu_count(),
        help="Number of processors to use."
    ),
    output_path: Path = typer.Option(
        "data/mimic-corpus_token_counts.csv",
        file_okay=True,
        dir_okay=False,
        help="Path to output file.",
    ),
):

    typer.echo(f"using note_events_path: {note_events_path}")
    typer.echo(f"using num_samples: {num_samples}")
    typer.echo(f"text_col: {text_col}")
    typer.echo(f"num_proc: {num_proc}")
    typer.echo(f"output_path: {output_path}")

    ds_full = load_dataset(
        mimic_noteevents.__file__,
        data_files=str(note_events_path),
        split="train",
    )
    if num_samples is None:
        ds = ds_full
    else:
        ds = ds_full.select(range(num_samples))

    df_num_toks = pd.DataFrame()
    for model_def in MODEL_DEFS:

        tokenizer = AutoTokenizer.from_pretrained(model_def["tokenizer_checkpoint"])

        ds_tokenized = ds.map(
            tokenize_map,
            batched=True,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "text_col": text_col,
            },
        )

        ds_counts = ds_tokenized.map(
            count_tokens_map,
            batched=True,
            num_proc=num_proc,
            remove_columns=ds_tokenized.column_names
        )
        df_num_toks[model_def["name"]] = ds_counts["num_tokens"]

    os.makedirs(output_path.parent, exist_ok=True)
    df_num_toks.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
