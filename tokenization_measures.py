"""
What can we measure with just the tokenizer?

* total token counts from MIMIC III
* TODO: add specific biomedical entities

"""

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer

from preprocessing import tokenize_map


data_files = (
    "/home/galtay/data/mimic_sandbox/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv.gz"
)
ds = load_dataset("mimic_noteevents.py", data_files=data_files, split="train")
ds = ds.select(range(1000))
TEXT_COL = "text"
NUM_PROC = 24


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


df_num_toks = pd.DataFrame()
for model_name in model_names:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # normally we drop all columns
    # keeping text column here just for convenience
    ds_tokenized = ds.map(
        tokenize_map,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=[col for col in ds.column_names if col != TEXT_COL],
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_col": TEXT_COL,
            "return_special_tokens_mask": False,
        },
    )

    df = ds_tokenized.to_pandas()

    print(tokenizer.convert_ids_to_tokens(df.iloc[0]["input_ids"]))

    df_num_toks[model_name] = df["input_ids"].apply(len)


df_num_toks.to_csv("data/corpus_token_counts.csv", index=False)
