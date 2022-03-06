"""
https://github.com/cambridgeltl/MTL-Bioinformatics-2016
https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py
https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8
https://huggingface.co/datasets/conll2003/blob/main/conll2003.py
"""
import json
import multiprocessing
import os
from pathlib import Path
import random
from typing import List, Optional

from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import typer

from eval_bio_lms.model_definitions import MODEL_DEFS
from eval_bio_lms.preprocessing import tokenize_map, group_texts_map
from eval_bio_lms.dataset_loaders import crichton_2017


# tokens with this label index will be ignored in the loss function
IGNORED_INDEX = -100


# NOTE NEED add_prefix_space=True for roberta if using is_split_into_words=True
SPECIAL_TOKENIZER_KWARGS = {
    "roberta-base": {"add_prefix_space": True}
}


def tokenize_and_align_labels(
    examples,
    label_all_tokens,
    tokenizer,
):
    """Create NER labels for pre-tokenized text."""

    # examples has,
    #   * id: sample id
    #   * tokens: pretokenized text
    #   * NER tags

    # {
    #     "id": [0, 1, 2, 3, ....],
    #     "tokens": [
    #         [samp1_tok1, ..., samp1_tokn],
    #         ...
    #         [sampn_tok1, ..., sampn_tokn],
    #     ],
    #     "tags": [
    #         [samp1_tag1, ..., samp1_tagn],
    #         ...
    #         [sampn_tag1, ..., sampn_tagn],
    #     ]
    # }

    # tokenized_inputs has a batch of tokenizer output.
    # {
    #     "input_ids": [
    #         [id_0, id_1, ...],
    #         [id_18, id_19, ...],
    #         [id_928, id_929, ...],
    #     ],
    #     "attention_mask": [[...],[...],[...]],
    # }

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    # Because we passed in text that was already split into words
    # our tokenized_inputs has word_ids. These map each token in
    # tokenized_inputs back to an element of examples["tokens"][ii].

    labels = []
    # loop over sequences in batch
    for ii, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=ii)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to
            # IGNORED_INDEX so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(IGNORED_INDEX)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the
            # current label or IGNORED_INDEX, depending on the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else IGNORED_INDEX)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class ComputeMetrics:

    def __init__(self, label_list: List[str]):
        self.label_list = label_list

    def __call__(self, eval_pred: EvalPrediction) -> dict:

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != IGNORED_INDEX]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != IGNORED_INDEX]
            for prediction, label in zip(predictions, labels)
        ]

        metric = load_metric("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return results
        #return {
        #    "precision": results["overall_precision"],
        #    "recall": results["overall_recall"],
        #    "f1": results["overall_f1"],
        #    "accuracy": results["overall_accuracy"],
        #}


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(
    crichton_2017_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help=(
            "Path to local copy of this "
            "https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data"
        ),
    ),
    num_proc: int = typer.Option(
        multiprocessing.cpu_count(),
        help="Number of processors to use."
    ),
    max_seq_len: Optional[int] = typer.Option(
        None,
        help="Maximum sequence length."
    ),
    tokenizer_batch_size: int = typer.Option(
        1000,
        help="Tokenizer batch size.",
    ),
    label_all_tokens: bool = typer.Option(
        True,
        help="If True, label all entity tokens, else label first entity token only."
    ),
    ner_batch_size: int = typer.Option(
        16,
        help="GPU batch size."
    ),
):

    subset_names = ["BC2GM-IOB", "BioNLP13CG-IOB"]
    final_results = {}

    for subset_name in subset_names:

        ds = load_dataset(
            crichton_2017.__file__,
            name=subset_name,
            data_dir=str(crichton_2017_path),
        )

        label_list = ds["train"].features["tags"].feature.names

        for model_def in MODEL_DEFS:

            this_max_seq_len = max_seq_len or model_def["max_seq_len"]
            tokenizer = AutoTokenizer.from_pretrained(
                model_def["tokenizer_checkpoint"],
                model_max_length=this_max_seq_len,
                **SPECIAL_TOKENIZER_KWARGS.get(model_def["name"], {})
            )

            ds_tokenized = ds.map(
                tokenize_and_align_labels,
                batched=True,
                batch_size=tokenizer_batch_size,
                num_proc=num_proc,
                remove_columns=ds['train'].column_names,
                fn_kwargs={
                    "label_all_tokens": label_all_tokens,
                    "tokenizer": tokenizer,
                },
            )

            model = AutoModelForTokenClassification.from_pretrained(
                model_def["model_checkpoint"],
                num_labels=len(label_list),
            )

            model_name = model_def["model_checkpoint"].split("/")[-1]
            args = TrainingArguments(
                os.path.join("ner_output", f"{model_name}-finetuned-ner"),
                evaluation_strategy = "epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=ner_batch_size,
                per_device_eval_batch_size=ner_batch_size,
                num_train_epochs=3,
                weight_decay=0.01,
            )

            data_collator = DataCollatorForTokenClassification(tokenizer)
            compute_metrics = ComputeMetrics(label_list)

            trainer = Trainer(
                model,
                args,
                train_dataset=ds_tokenized["train"],
                eval_dataset=ds_tokenized["validation"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            train_output = trainer.train()
            eval_output = trainer.evaluate()
            predictions, labels, _ = trainer.predict(ds_tokenized["test"])
            results = compute_metrics((predictions, labels))

            results["model_def_name"] = model_def["name"]
            results["subset_name"] = subset_name

            key = f"{model_def['name']}-{subset_name}"
            final_results[key] = results

    with open("data/crichton-2017-ner.json", "w") as fp:
        json.dump(final_results, fp, indent=4, cls=NumpyEncoder)

#    return final_results

if __name__ == "__main__":
    #final_results = typer.run(main)
    typer.run(main)
