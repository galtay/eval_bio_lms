import json
import os
from typing import List
import zipfile

import datasets
from datasets import Features, Value


TRAINING_ZIPS = [
    "BioASQ-trainingDataset2b.zip",
    "BioASQ-trainingDataset3b.zip",
    "BioASQ-training4b.zip",
    "BioASQ-training5b.zip",
    "BioASQ-training6b.zip",
    "BioASQ-training7b.zip",
    "BioASQ-training8b.zip",
    "BioASQ-training9b.zip",
]

GOLDEN_ZIPS = [
    "Task2BGoldenEnriched.zip",
    "Task3BGoldenEnriched.zip",
    "Task4BGoldenEnriched.zip",
    "Task5BGoldenEnriched.zip",
    "Task6BGoldenEnriched.zip",
    "Task7BGoldenEnriched.zip",
    "Task8BGoldenEnriched.zip",
    "Task9BGoldenEnriched.zip",
]


_HOMEPAGE = """http://participants-area.bioasq.org/datasets"""
_DESCRIPTION = """BioASQ task b."""
_LICENSE = """https://www.nlm.nih.gov/databases/download/terms_and_conditions.html"""
_CITATION = """\
@article{tsatsaronis2015overview,
	title        = {
		An overview of the BIOASQ large-scale biomedical semantic indexing and
		question answering competition
	},
	author       = {
		Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos and
		Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R and
		Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and
		Polychronopoulos, Dimitris and others
	},
	year         = 2015,
	journal      = {BMC bioinformatics},
	publisher    = {BioMed Central Ltd},
	volume       = 16,
	number       = 1,
	pages        = 138
}
"""



def _read_zip(file_path):
    questions = []
    with zipfile.ZipFile(file_path) as zf:
        for info in zf.infolist():
            base, filename = os.path.split(info.filename)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            if ext == "json":
                content = zf.read(info).decode("utf-8")
                content_dict = json.loads(content)
                questions.extend(content_dict["questions"])

    return questions



class BioAsqTaskBDataset(datasets.GeneratorBasedBuilder):
    """BioASQ task b"""

    _VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="source",
            version=_VERSION,
            description="original format",
        ),
    ]

    def _info(self):

        if self.config.name == "source":
            features = datasets.Features(
                {
                    "id": Value("string"),
                    "type": Value("string"),
                    "body": Value("string"),
                    "documents": [Value("string")],
                    "concepts": [Value("string")],
                    "ideal_answer": [Value("string")],
                    "exact_answer": [Value("string")],
                    "triples": [
                        {
                            "p": Value("string"),
                            "s": Value("string"),
                            "o": Value("string"),
                        }
                    ],
                    "snippets": [
                        {
                            "offsetInBeginSection": Value("int32"),
                            "offsetInEndSection": Value("int32"),
                            "text": Value("string"),
                            "beginSection": Value("string"),
                            "endSection": Value("string"),
                            "document": Value("string"),
                        }
                    ],
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

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "file_path": os.path.join(self.config.data_dir, TRAINING_ZIPS[-1]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "file_path": os.path.join(self.config.data_dir, GOLDEN_ZIPS[-1]),
                },
            ),
        ]


    def _generate_examples(self, split, file_path):
        """Generate samples using the info passed in from _split_generators."""

        questions = _read_zip(file_path)

        if self.config.name == "source":
            for key, question in enumerate(questions):
                yield key, {
                    "id": question["id"],
                    "type": question["type"],
                    "body": question["body"],
                    "documents": question["documents"],
                    "concepts": question.get("concepts", []),
                    "triples": question.get("triples", []),
                    "ideal_answer": question["ideal_answer"],
                    "exact_answer": question.get("exact_answer"),
                    "snippets": question["snippets"],
                }






if __name__ == "__main__":


    data_dir = "/home/galtay/data/galtay-datasets/bioasq-taskb"

    from datasets import load_dataset

    ds = load_dataset("bioasq_taskb.py", name="source", data_dir=data_dir)


#train_zip = TRAINING_ZIPS[-1]
#file_path = os.path.join(data_dir, train_zip)
#train_questions = _read_zip(file_path)

#gold_zip = GOLDEN_ZIPS[-1]
#file_path = os.path.join(data_dir, gold_zip)
#gold_questions = _read_zip(file_path)
