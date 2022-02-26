"""
What can we measure with just the tokenizer?

* total token counts from MIMIC III
* TODO: add specific biomedical entities

"""
import multiprocessing
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer

from preprocessing import tokenize_map
from model_definitions import MODEL_DEFS


data_files = "/mnt/disks/data1/galtay-datasets/mimiciii-1.4.physionet.org/NOTEEVENTS.csv.gz"
ds_full = load_dataset("mimic_noteevents.py", data_files=data_files, split="train")
ds = ds_full.select(range(10_000))
TEXT_COL = "text"
NUM_PROC = multiprocessing.cpu_count()


df_num_toks = pd.DataFrame()
for model_def in MODEL_DEFS:

    tokenizer = AutoTokenizer.from_pretrained(model_def["tokenizer_checkpoint"])

    # normally we drop all input columns
    # keeping text column here just for convenience
    ds_tokenized = ds.map(
        tokenize_map,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=[col for col in ds.column_names if col != TEXT_COL],
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_col": TEXT_COL,
        },
    )

    df = ds_tokenized.to_pandas()

    print(tokenizer.convert_ids_to_tokens(df.iloc[0]["input_ids"]))

    df_num_toks[model_def["name"]] = df["input_ids"].apply(len)


df_num_toks.to_csv("data/corpus_token_counts.csv", index=False)
