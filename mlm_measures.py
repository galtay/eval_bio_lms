"""
Lets measure performance on the MLM task.

* MLM on MIMIC III
* https://github.com/awslabs/mlm-scoring

"""
import itertools
import math

from datasets import load_dataset, load_metric
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

from preprocessing import tokenize_map
from preprocessing import group_texts_map


data_files = (
    "/home/galtay/data/mimic_sandbox/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz"
)
ds = load_dataset("mimic_noteevents.py", data_files=data_files, split="train")
ds = ds.select(range(1000))
TEXT_COL = "text"
MAX_SEQ_LEN = 128
NUM_PROC = 24


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


metric = load_metric("accuracy")


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


model_names = [
    "bert-base-uncased",
    "bert-base-cased",
    "roberta-base",
    "dmis-lab/biobert-v1.1",
    "allenai/scibert_scivocab_uncased",
    "allenai/scibert_scivocab_cased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
    "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
]


eval_dicts = []
for model_name in model_names:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # We use `return_special_tokens_mask=True` because it makes
    # DataCollatorForLanguageModeling more efficient
    ds_tokenized = ds.map(
        tokenize_map,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_col": TEXT_COL,
            "return_special_tokens_mask": True,
        },
    )

    ds_lm = ds_tokenized.map(
        group_texts_map,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        fn_kwargs={"max_seq_len": MAX_SEQ_LEN},
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    output_path = "./eval"
    training_args = TrainingArguments(
        output_path,
        per_device_eval_batch_size=256,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=ds_lm,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    eval_dict = trainer.evaluate()
    eval_dict["model_name"] = model_name
    try:
        perplexity = math.exp(eval_dict["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    eval_dict["perplexity"] = perplexity

    eval_dicts.append(eval_dict)


df_eval = pd.DataFrame(eval_dicts)
df_eval.to_csv("data/mlm_measures.csv", index=False)
