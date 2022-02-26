"""
For the core models of Hugging Face, things mostly work out of the box.
For some contributed models, the casing is not handled consistently.
* https://github.com/allenai/scibert/issues/79
* https://github.com/EmilyAlsentzer/clinicalBERT/issues/26
For this reason, we manually set the casing strategy for each model.
"""


MODEL_DEFS = [
    {
        "tokenizer_checkpoint": "bert-base-uncased",
        "model_checkpoint": "bert-base-uncased",
        "name": "bert-base-uncased",
        "cased": False,
    },
    {
        "tokenizer_checkpoint": "bert-base-cased",
        "model_checkpoint": "bert-base-cased",
        "name": "bert-base-cased",
        "cased": True,
    },
    {
        "tokenizer_checkpoint": "roberta-base",
        "model_checkpoint": "roberta-base",
        "name": "roberta-base",
        "cased": True,
    },
    {
        "tokenizer_checkpoint": "dmis-lab/biobert-v1.1",
        "model_checkpoint": "dmis-lab/biobert-v1.1",
        "name": "biobert-v1.1",
        "cased": True,
    },
    {
        "tokenizer_checkpoint": "allenai/scibert_scivocab_uncased",
        "model_checkpoint": "allenai/scibert_scivocab_uncased",
        "name": "scibert_scivocab_uncased",
        "cased": False,
    },
    {
        "tokenizer_checkpoint": "allenai/scibert_scivocab_cased",
        "model_checkpoint": "allenai/scibert_scivocab_cased",
        "name": "scibert_scivocab_cased",
        "cased": True,
    },
    {
        "tokenizer_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "model_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "name": "BiomedNLP-PubMedBERT-base-uncased-abstract",
        "cased": False,
    },
    {
        "tokenizer_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "model_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "name": "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "cased": False,
    },
    {
        "tokenizer_checkpoint": "emilyalsentzer/Bio_ClinicalBERT",
        "model_checkpoint": "emilyalsentzer/Bio_ClinicalBERT",
        "name": "Bio_ClinicalBERT",
        "cased": False,
    },
    {
        "tokenizer_checkpoint": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
        "model_checkpoint": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
        "name": "bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
        "cased": False,
    },
    {
        "tokenizer_checkpoint": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        "model_checkpoint": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        "name": "bluebert_pubmed_uncased_L-12_H-768_A-12",
        "cased": False
    },
]
