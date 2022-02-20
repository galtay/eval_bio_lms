"""
A hugging face dataset loader script for Mimic III Note Events.
"""

from typing import List

import datasets
from datasets import Features, Value
import pandas as pd


_HOMEPAGE = """https://physionet.org/content/mimiciii/1.4/"""
_DESCRIPTION = """Mimic III Note Events Dataset."""
_LICENSE = """https://physionet.org/content/mimiciii/view-license/1.4/"""
_CITATION = """@article{article,
author = {Johnson, Alistair and Pollard, Tom and Shen, Lu and Lehman, Li-wei and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Celi, Leo and Mark, Roger},
year = {2016},
month = {05},
pages = {160035},
title = {MIMIC-III, a freely accessible critical care database},
volume = {3},
journal = {Scientific Data},
doi = {10.1038/sdata.2016.35}
}"""


class Mimic3NoteEventsDataset(datasets.GeneratorBasedBuilder):
    """Mimic III Note Events Dataset"""

    _VERSION = datasets.Version("1.4.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=_VERSION,
            description="Mimic III Note Events",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):

        if self.config.name == "default":
            features = Features(
                {
                    "row_id": Value("int32"),
                    "subject_id": Value("int32"),
                    "hadm_id": Value("string"),
                    "chartdate": Value("string"),
                    "chartime": Value("string"),
                    "storetime": Value("string"),
                    "category": Value("string"),
                    "description": Value("string"),
                    "cgid": Value("string"),
                    "iserror": Value("string"),
                    "text": Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            ),
        ]


    def _generate_examples(self, split):
        """Generate samples using the info passed in from _split_generators."""

        df = pd.read_csv(self.config.data_files[split][0])
        _id = 0
        for _, row in df.iterrows():
            yield _id, {
                "row_id": row["ROW_ID"],
                "subject_id": row["SUBJECT_ID"],
                "hadm_id": row["HADM_ID"],
                "chartdate": row["CHARTDATE"],
                "chartime": row["CHARTTIME"],
                "storetime": row["STORETIME"],
                "category": row["CATEGORY"],
                "description": row["DESCRIPTION"],
                "cgid": row["CGID"],
                "iserror": row["ISERROR"],
                "text": row["TEXT"],
            }
            _id += 1
