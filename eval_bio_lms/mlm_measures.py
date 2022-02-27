"""
Lets measure performance on the MLM task.

* MLM on MIMIC III
* https://github.com/awslabs/mlm-scoring

"""
import multiprocessing
import os
from pathlib import Path
import math
from typing import Optional

from datasets import load_dataset, load_metric
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import typer

from eval_bio_lms.model_definitions import MODEL_DEFS
from eval_bio_lms.preprocessing import tokenize_map, group_texts_map
from eval_bio_lms.dataset_loaders import mimic_noteevents


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


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
    max_seq_len: int = typer.Option(
        128,
        help="Maximum sequence length."
    ),
    tokenizer_batch_size: int = typer.Option(
        1000,
        help="Tokenizer batch size.",
    ),
    mlm_probability: float = typer.Option(
        0.15,
        help="Masked language modeling probability.",
    ),
    mlm_batch_size: int = typer.Option(
        256,
        help="GPU batch size",
    ),
    output_path: Path = typer.Option(
        "data/mimic-corpus-mlm.csv",
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

    metric = load_metric("accuracy")

    ds_full = load_dataset(
        mimic_noteevents.__file__,
        data_files=str(note_events_path),
        split="train",
    )
    if num_samples is None:
        ds = ds_full
    else:
        ds = ds_full.select(range(num_samples))

    eval_dicts = []
    for model_def in MODEL_DEFS:

        tokenizer = AutoTokenizer.from_pretrained(model_def["tokenizer_checkpoint"])

        # We use `return_special_tokens_mask=True` because it makes
        # DataCollatorForLanguageModeling more efficient
        ds_tokenized = ds.map(
            tokenize_map,
            batched=True,
            batch_size=tokenizer_batch_size,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "text_col": text_col,
                "return_special_tokens_mask": True,
            },
        )

        ds_lm = ds_tokenized.map(
            group_texts_map,
            batched=True,
            batch_size=tokenizer_batch_size,
            num_proc=num_proc,
            fn_kwargs={"max_seq_len": max_seq_len},
        )

        model = AutoModelForMaskedLM.from_pretrained(model_def["model_checkpoint"])

        training_args = TrainingArguments(
            "/tmp",
            per_device_eval_batch_size=mlm_batch_size,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=ds_lm,
            data_collator=data_collator,
#            compute_metrics=compute_metrics,
#            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        eval_dict = trainer.evaluate()
        eval_dict["model_name"] = model_def["name"]
        eval_dict["num_seq"] = len(ds_lm)
        try:
            perplexity = math.exp(eval_dict["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        eval_dict["perplexity"] = perplexity

        eval_dicts.append(eval_dict)


    df_eval = pd.DataFrame(eval_dicts)
    os.makedirs(output_path.parent, exist_ok=True)
    df_eval.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
