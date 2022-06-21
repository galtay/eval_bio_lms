"""
For the core models of Hugging Face, things mostly work out of the box.
For some contributed models, the casing is not handled consistently.
* https://github.com/allenai/scibert/issues/79
* https://github.com/EmilyAlsentzer/clinicalBERT/issues/26
For this reason, we manually set model parameters.
"""
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification


@dataclass
class HuggingFaceModelDefinition:
    name: str
    tokenizer_checkpoint: str
    model_checkpoint: str
    cased: bool
    max_seq_len: int
    arxiv: str

    def load_tokenizer(self, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_checkpoint,
            do_lower_case = not self.cased,
            model_max_length = self.max_seq_len,
            **kwargs,
        )
        return tokenizer

    def load_masked_lm(self, **kwargs):
        model = AutoModelForMaskedLM.from_pretrained(
            self.model_checkpoint,
            **kwargs,
        )
        return model

    def load_token_classification(self, num_labels, **kwargs):
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=num_labels,
            **kwargs,
        )
        return model


MODEL_DEF_DICTS = [
    {
        "name": "BioLinkBERT-base",
        "tokenizer_checkpoint": "michiyasunaga/BioLinkBERT-base",
        "model_checkpoint": "michiyasunaga/BioLinkBERT-base",
        "cased": False,
        "max_seq_len": 512,
        "arxiv": "https://arxiv.org/abs/2203.15827",
    },
#    {
#        "name": "BioLinkBERT-large",
#        "tokenizer_checkpoint": "michiyasunaga/BioLinkBERT-large",
#        "model_checkpoint": "michiyasunaga/BioLinkBERT-large",
#        "cased": False,
#        "max_seq_len": 512,
#        "arxiv": "https://arxiv.org/abs/2203.15827",
#    },
    {
        "name": "BioLinkBERT-large",
        "tokenizer_checkpoint": "/home/galtay/data/bio-link-bert/BioLinkBERT-large-with-heads",
        "model_checkpoint": "/home/galtay/data/bio-link-bert/BioLinkBERT-large-with-heads",
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
        "max_seq_len": 128,
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


MODEL_DEFS = [
    HuggingFaceModelDefinition(**kwargs)
    for kwargs in MODEL_DEF_DICTS
]
