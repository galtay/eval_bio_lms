"""
https://github.com/cambridgeltl/MTL-Bioinformatics-2016
https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py
https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8
https://huggingface.co/datasets/conll2003/blob/main/conll2003.py
"""
import random

from datasets import Sequence, ClassLabel
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

from eval_bio_lms.model_definitions import MODEL_DEFS
from eval_bio_lms.preprocessing import tokenize_map, group_texts_map
from eval_bio_lms.dataset_loaders import crichton_2017


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])

    return df



def tokenize_and_align_labels(examples, label_all_tokens):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to
            # -100 so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the
            # current label or -100, depending on the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }





tokenizer_batch_size = 1000
num_proc = 24
label_all_tokens = True
batch_size = 16

data_dir = "/home/galtay/repos/MTL-Bioinformatics-2016/data"
name = "BC2GM-IOB"
ds_full = load_dataset(
    crichton_2017.__file__,
    name=name,
    data_dir=data_dir,
)
ds = ds_full

# NOTE NEED add_prefix_space=True for roberta if using is_split_into_words=True
special_tokenizer_kwargs = {
    "roberta-base": {"add_prefix_space": True}
}

label_list = ds["train"].features["tags"].feature.names
print(label_list)

final_results = {}
for model_def in MODEL_DEFS:

    tokenizer = AutoTokenizer.from_pretrained(
        model_def["tokenizer_checkpoint"],
        **special_tokenizer_kwargs.get(model_def["name"], {})
    )

    ds_tokenized = ds.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=num_proc,
        remove_columns=ds['train'].column_names,
        fn_kwargs={
            "label_all_tokens": label_all_tokens,
        },
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_def["model_checkpoint"],
        num_labels=len(label_list),
    )

    model_name = model_def["model_checkpoint"].split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-ner",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    trainer = Trainer(
        model,
        args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    train_output = trainer.train()
    eval_output = trainer.evaluate()
    predictions, labels, _ = trainer.predict(ds_tokenized["validation"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(
        predictions=true_predictions,
        references=true_labels
    )
    final_results[model_def["name"]] = results
