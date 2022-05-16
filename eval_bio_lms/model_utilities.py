"""
For the core models of Hugging Face, things mostly work out of the box.
For some contributed models, the casing is not handled consistently.
* https://github.com/allenai/scibert/issues/79
* https://github.com/EmilyAlsentzer/clinicalBERT/issues/26
For this reason, we manually set model parameters.
"""
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class HuggingFaceModelDefinition:
    name: str
    tokenizer_checkpoint: str
    model_checkpoint: str
    cased: bool
    max_seq_len: int
    arxiv: str


MODEL_DEFS = [
    {
        "name": "BioLinkBERT-base",
        "tokenizer_checkpoint": "michiyasunaga/BioLinkBERT-base",
        "model_checkpoint": "michiyasunaga/BioLinkBERT-base",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/2203.15827",
    },
    {
        "name": "BioLinkBERT-large",
        "tokenizer_checkpoint": "michiyasunaga/BioLinkBERT-large",
        "model_checkpoint": "michiyasunaga/BioLinkBERT-large",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/2203.15827",
    },
    {
        "name": "bert-base-uncased",
        "tokenizer_checkpoint": "bert-base-uncased",
        "model_checkpoint": "bert-base-uncased",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1810.04805",
    },
    {
        "name": "bert-base-cased",
        "tokenizer_checkpoint": "bert-base-cased",
        "model_checkpoint": "bert-base-cased",
        "cased": True,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1810.04805",
    },
    {
        "name": "roberta-base",
        "tokenizer_checkpoint": "roberta-base",
        "model_checkpoint": "roberta-base",
        "cased": True,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1907.11692",
    },
    {
        "name": "roberta-large",
        "tokenizer_checkpoint": "roberta-large",
        "model_checkpoint": "roberta-large",
        "cased": True,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1907.11692",
    },
    {
        "name": "biobert-base-cased-v1.2",
        "tokenizer_checkpoint": "dmis-lab/biobert-base-cased-v1.2",
        "model_checkpoint": "dmis-lab/biobert-base-cased-v1.2",
        "cased": True,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1901.08746",
    },
    {
        "name": "scibert_scivocab_uncased",
        "tokenizer_checkpoint": "allenai/scibert_scivocab_uncased",
        "model_checkpoint": "allenai/scibert_scivocab_uncased",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1903.10676",
    },
    {
        "name": "scibert_scivocab_cased",
        "tokenizer_checkpoint": "allenai/scibert_scivocab_cased",
        "model_checkpoint": "allenai/scibert_scivocab_cased",
        "cased": True,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1903.10676",
    },
    {
        "name": "BiomedNLP-PubMedBERT-base-uncased-abstract",
        "tokenizer_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "model_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/2007.15779",
    },
    {
        "name": "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "tokenizer_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "model_checkpoint": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/2007.15779",
    },
    {
        "name": "Bio_ClinicalBERT",
        "tokenizer_checkpoint": "emilyalsentzer/Bio_ClinicalBERT",
        "model_checkpoint": "emilyalsentzer/Bio_ClinicalBERT",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1904.03323",
    },
    {
        "name": "bluebert_pubmed_uncased_L-24_H-1024_A-16",
        "tokenizer_checkpoint": "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
        "model_checkpoint": "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1906.05474",
    },
    {
        "name": "bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
        "tokenizer_checkpoint": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
        "model_checkpoint": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1906.05474",
    },
    {
        "name": "bluebert_pubmed_uncased_L-12_H-768_A-12",
        "tokenizer_checkpoint": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        "model_checkpoint": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1906.05474",
    },
    {
        "name": "bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "tokenizer_checkpoint": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "model_checkpoint": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/1906.05474",
    },
]


def load_tokenizer(model_def, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_def["tokenizer_checkpoint"],
        do_lower_case = not model_def["cased"],
        **kwargs,
    )
    return tokenizer
