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


data_files = "/home/galtay/data/mimic_sandbox/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz"
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


# We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling
# (see below) is more efficient when it receives the `special_tokens_mask`.
def tokenize_map(examples):
    return tokenizer(examples[TEXT_COL], return_special_tokens_mask=True)


def group_texts_map(examples):

    # input examples is a batch of tokenizer output. for example,
    # {
    #     "input_ids": [
    #         [id_0, id_1, ...],
    #         [id_18, id_19, ...],
    #         [id_928, id_929, ...],
    #     ],
    #     "attention_mask": [[...],[...],[...]],
    #     "special_tokens_mask": [[...],[...],[...]],
    # }

    # concatenated examples flattens the nested input iterables
    # {
    #     "input_ids": [id_0, id_1, ..., id_18, id_19, ..., id_928, id_929, ...],
    #     "attention_mask": [...],
    #     "special_tokens_mask": [...],
    # }
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}


    # all of the keys map to lists of the same length, so we get the first
    arbitrary_key = list(examples.keys())[0]
    total_length = len(concatenated_examples[arbitrary_key])

    # truncate the remainder
    # [max_seq_len    ][max_seq_len    ][max_seq_len    ][remain]
    # ^ keep           ^ keep           ^ keep           ^ remove
    if total_length > MAX_SEQ_LEN:
        total_length = (total_length // MAX_SEQ_LEN) * MAX_SEQ_LEN

    # result stores MAX_SEQ_LEN chunks (example for MAX_SEQ_LEN=128)
    # {
    #     "input_ids": [
    #         [id_0, ..., id_127],
    #         [id_128, ..., id_255],
    #         [id_256, ..., id_383],
    #         [id_384, ..., id_511],
    #         ...
    #     ],
    #     "attention_mask": [[...],[...],[...]],
    #     "special_tokens_mask": [[...],[...],[...]],
    # }
    result = {
        k: [t[i : i + MAX_SEQ_LEN] for i in range(0, total_length, MAX_SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }

    # labels becomes a copy of input_ids
    # masking and/or shifting for causal language modelling is handled later
    result["labels"] = result["input_ids"].copy()

    return result


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
    ds_tokenized = ds.map(
        tokenize_map,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
    )
    ds_lm = ds_tokenized.map(
        group_texts_map,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    output_path = "./eval"
    training_args = TrainingArguments(
        output_path,
        per_device_eval_batch_size = 256,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = None,
        eval_dataset = ds_lm,
        data_collator = data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    eval_dict = trainer.evaluate()
    eval_dict['model_name'] = model_name
    try:
        perplexity = math.exp(eval_dict["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    eval_dict["perplexity"] = perplexity

    eval_dicts.append(eval_dict)


df_eval = pd.DataFrame(eval_dicts)
df_eval.to_csv('data/mlm_measures.csv', index=False)
