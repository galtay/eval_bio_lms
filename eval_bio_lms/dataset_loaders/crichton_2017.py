"""
A hugging face dataset loader for the 15 NER datasets used in
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1776-8
"""
import json
import os
import pathlib
from typing import List

import datasets
from datasets import Features, Sequence, Value


_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = ""
_LICENSE = ""
_VERSION = "1.0.0"

CONFIGS = [
    "AnatEM-IOB",
    "AnatEM-IOBES",
    "BC2GM-IOB",
    "BC2GM-IOBES",
    "BC4CHEMD",
    "BC4CHEMD-IOBES",
    "BC5CDR-chem-IOB",
    "BC5CDR-chem-IOBES",
    "BC5CDR-disease-IOB",
    "BC5CDR-disease-IOBES",
    "BC5CDR-IOB",
    "BC5CDR-IOBES",
    "BioNLP09-IOB",
    "BioNLP09-IOBES",
    "BioNLP11EPI-IOB",
    "BioNLP11EPI-IOBES",
    "BioNLP11ID-chem-IOB",
    "BioNLP11ID-chem-IOBES",
    "BioNLP11ID-ggp-IOB",
    "BioNLP11ID-ggp-IOBES",
    "BioNLP11ID-IOB",
    "BioNLP11ID-IOBES",
    "BioNLP11ID-species-IOB",
    "BioNLP11ID-species-IOBES",
    "BioNLP13CG-cc-IOB",
    "BioNLP13CG-cc-IOBES",
    "BioNLP13CG-cell-IOB",
    "BioNLP13CG-cell-IOBES",
    "BioNLP13CG-chem-IOB",
    "BioNLP13CG-chem-IOBES",
    "BioNLP13CG-ggp-IOB",
    "BioNLP13CG-ggp-IOBES",
    "BioNLP13CG-IOB",
    "BioNLP13CG-IOBES",
    "BioNLP13CG-species-IOB",
    "BioNLP13CG-species-IOBES",
    "BioNLP13GE-IOB",
    "BioNLP13GE-IOBES",
    "BioNLP13PC-cc-IOB",
    "BioNLP13PC-cc-IOBES",
    "BioNLP13PC-chem-IOB",
    "BioNLP13PC-chem-IOBES",
    "BioNLP13PC-ggp-IOB",
    "BioNLP13PC-ggp-IOBES",
    "BioNLP13PC-IOB",
    "BioNLP13PC-IOBES",
    "CRAFT-cc-IOB",
    "CRAFT-cc-IOBES",
    "CRAFT-cell-IOB",
    "CRAFT-cell-IOBES",
    "CRAFT-chem-IOB",
    "CRAFT-chem-IOBES",
    "CRAFT-ggp-IOB",
    "CRAFT-ggp-IOBES",
    "CRAFT-IOB",
    "CRAFT-IOBES",
    "CRAFT-species-IOB",
    "CRAFT-species-IOBES",
    "Ex-PTM-IOB",
    "Ex-PTM-IOBES",
    "GENIA-pos",
    "JNLPBA",
    "JNLPBA-IOBES",
    "linnaeus-IOB",
    "linnaeus-IOBES",
    "NCBI-disease-IOB",
    "NCBI-disease-IOBES",
]


CLASS_LABELS = {
    "AnatEM-IOB": ["B-Anatomy", "I-Anatomy", "O"],
    "AnatEM-IOBES": ["B-Anatomy", "E-Anatomy", "I-Anatomy", "O", "S-Anatomy"],
    "BC2GM-IOB": ["B-GENE", "I-GENE", "O"],
    "BC2GM-IOBES": ["B-GENE", "E-GENE", "I-GENE", "O", "S-GENE"],
    "BC4CHEMD": ["B-Chemical", "I-Chemical", "O"],
    "BC4CHEMD-IOBES": ["B-Chemical", "E-Chemical", "I-Chemical", "O", "S-Chemical"],
    "BC5CDR-chem-IOB": ["B-Chemical", "I-Chemical", "O"],
    "BC5CDR-chem-IOBES": ["B-Chemical", "E-Chemical", "I-Chemical", "O", "S-Chemical"],
    "BC5CDR-disease-IOB": ["B-Disease", "I-Disease", "O"],
    "BC5CDR-disease-IOBES": ["B-Disease", "E-Disease", "I-Disease", "O", "S-Disease"],
    "BC5CDR-IOB": ["B-Chemical", "B-Disease", "I-Chemical", "I-Disease", "O"],
    "BC5CDR-IOBES": [
        "B-Chemical",
        "B-Disease",
        "E-Chemical",
        "E-Disease",
        "I-Chemical",
        "I-Disease",
        "O",
        "S-Chemical",
        "S-Disease",
    ],
    "BioNLP09-IOB": ["B-Protein", "I-Protein", "O"],
    "BioNLP09-IOBES": ["B-Protein", "E-Protein", "I-Protein", "O", "S-Protein"],
    "BioNLP11EPI-IOB": ["B-Protein", "I-Protein", "O"],
    "BioNLP11EPI-IOBES": ["B-Protein", "E-Protein", "I-Protein", "O", "S-Protein"],
    "BioNLP11ID-chem-IOB": ["B-Chemical", "I-Chemical", "O"],
    "BioNLP11ID-chem-IOBES": [
        "B-Chemical",
        "E-Chemical",
        "I-Chemical",
        "O",
        "S-Chemical",
    ],
    "BioNLP11ID-ggp-IOB": ["B-Protein", "I-Protein", "O"],
    "BioNLP11ID-ggp-IOBES": ["B-Protein", "E-Protein", "I-Protein", "O", "S-Protein"],
    "BioNLP11ID-IOB": [
        "B-Chemical",
        "B-Organism",
        "B-Protein",
        "B-Regulon-operon",
        "I-Chemical",
        "I-Organism",
        "I-Protein",
        "I-Regulon-operon",
        "O",
    ],
    "BioNLP11ID-IOBES": [
        "B-Chemical",
        "B-Organism",
        "B-Protein",
        "B-Regulon-operon",
        "E-Chemical",
        "E-Organism",
        "E-Protein",
        "E-Regulon-operon",
        "I-Chemical",
        "I-Organism",
        "I-Protein",
        "I-Regulon-operon",
        "O",
        "S-Chemical",
        "S-Organism",
        "S-Protein",
        "S-Regulon-operon",
    ],
    "BioNLP11ID-species-IOB": ["B-Organism", "I-Organism", "O"],
    "BioNLP11ID-species-IOBES": [
        "B-Organism",
        "E-Organism",
        "I-Organism",
        "O",
        "S-Organism",
    ],
    "BioNLP13CG-cc-IOB": ["B-Cellular_component", "I-Cellular_component", "O"],
    "BioNLP13CG-cc-IOBES": [
        "B-Cellular_component",
        "E-Cellular_component",
        "I-Cellular_component",
        "O",
        "S-Cellular_component",
    ],
    "BioNLP13CG-cell-IOB": ["B-Cell", "I-Cell", "O"],
    "BioNLP13CG-cell-IOBES": ["B-Cell", "E-Cell", "I-Cell", "O", "S-Cell"],
    "BioNLP13CG-chem-IOB": ["B-Simple_chemical", "I-Simple_chemical", "O"],
    "BioNLP13CG-chem-IOBES": [
        "B-Simple_chemical",
        "E-Simple_chemical",
        "I-Simple_chemical",
        "O",
        "S-Simple_chemical",
    ],
    "BioNLP13CG-ggp-IOB": ["B-Gene_or_gene_product", "I-Gene_or_gene_product", "O"],
    "BioNLP13CG-ggp-IOBES": [
        "B-Gene_or_gene_product",
        "E-Gene_or_gene_product",
        "I-Gene_or_gene_product",
        "O",
        "S-Gene_or_gene_product",
    ],
    "BioNLP13CG-IOB": [
        "B-Amino_acid",
        "B-Anatomical_system",
        "B-Cancer",
        "B-Cell",
        "B-Cellular_component",
        "B-Developing_anatomical_structure",
        "B-Gene_or_gene_product",
        "B-Immaterial_anatomical_entity",
        "B-Multi-tissue_structure",
        "B-Organ",
        "B-Organism",
        "B-Organism_subdivision",
        "B-Organism_substance",
        "B-Pathological_formation",
        "B-Simple_chemical",
        "B-Tissue",
        "I-Amino_acid",
        "I-Anatomical_system",
        "I-Cancer",
        "I-Cell",
        "I-Cellular_component",
        "I-Developing_anatomical_structure",
        "I-Gene_or_gene_product",
        "I-Immaterial_anatomical_entity",
        "I-Multi-tissue_structure",
        "I-Organ",
        "I-Organism",
        "I-Organism_subdivision",
        "I-Organism_substance",
        "I-Pathological_formation",
        "I-Simple_chemical",
        "I-Tissue",
        "O",
    ],
    "BioNLP13CG-IOBES": [
        "B-Amino_acid",
        "B-Anatomical_system",
        "B-Cancer",
        "B-Cell",
        "B-Cellular_component",
        "B-Developing_anatomical_structure",
        "B-Gene_or_gene_product",
        "B-Immaterial_anatomical_entity",
        "B-Multi-tissue_structure",
        "B-Organ",
        "B-Organism",
        "B-Organism_subdivision",
        "B-Organism_substance",
        "B-Pathological_formation",
        "B-Simple_chemical",
        "B-Tissue",
        "E-Amino_acid",
        "E-Anatomical_system",
        "E-Cancer",
        "E-Cell",
        "E-Cellular_component",
        "E-Developing_anatomical_structure",
        "E-Gene_or_gene_product",
        "E-Immaterial_anatomical_entity",
        "E-Multi-tissue_structure",
        "E-Organ",
        "E-Organism",
        "E-Organism_subdivision",
        "E-Organism_substance",
        "E-Pathological_formation",
        "E-Simple_chemical",
        "E-Tissue",
        "I-Amino_acid",
        "I-Anatomical_system",
        "I-Cancer",
        "I-Cell",
        "I-Cellular_component",
        "I-Developing_anatomical_structure",
        "I-Gene_or_gene_product",
        "I-Immaterial_anatomical_entity",
        "I-Multi-tissue_structure",
        "I-Organ",
        "I-Organism",
        "I-Organism_subdivision",
        "I-Organism_substance",
        "I-Pathological_formation",
        "I-Simple_chemical",
        "I-Tissue",
        "O",
        "S-Amino_acid",
        "S-Anatomical_system",
        "S-Cancer",
        "S-Cell",
        "S-Cellular_component",
        "S-Developing_anatomical_structure",
        "S-Gene_or_gene_product",
        "S-Immaterial_anatomical_entity",
        "S-Multi-tissue_structure",
        "S-Organ",
        "S-Organism",
        "S-Organism_subdivision",
        "S-Organism_substance",
        "S-Pathological_formation",
        "S-Simple_chemical",
        "S-Tissue",
    ],
    "BioNLP13CG-species-IOB": ["B-Organism", "I-Organism", "O"],
    "BioNLP13CG-species-IOBES": [
        "B-Organism",
        "E-Organism",
        "I-Organism",
        "O",
        "S-Organism",
    ],
    "BioNLP13GE-IOB": ["B-Protein", "I-Protein", "O"],
    "BioNLP13GE-IOBES": ["B-Protein", "E-Protein", "I-Protein", "O", "S-Protein"],
    "BioNLP13PC-cc-IOB": ["B-Cellular_component", "I-Cellular_component", "O"],
    "BioNLP13PC-cc-IOBES": [
        "B-Cellular_component",
        "E-Cellular_component",
        "I-Cellular_component",
        "O",
        "S-Cellular_component",
    ],
    "BioNLP13PC-chem-IOB": ["B-Simple_chemical", "I-Simple_chemical", "O"],
    "BioNLP13PC-chem-IOBES": [
        "B-Simple_chemical",
        "E-Simple_chemical",
        "I-Simple_chemical",
        "O",
        "S-Simple_chemical",
    ],
    "BioNLP13PC-ggp-IOB": ["B-Gene_or_gene_product", "I-Gene_or_gene_product", "O"],
    "BioNLP13PC-ggp-IOBES": [
        "B-Gene_or_gene_product",
        "E-Gene_or_gene_product",
        "I-Gene_or_gene_product",
        "O",
        "S-Gene_or_gene_product",
    ],
    "BioNLP13PC-IOB": [
        "B-Cellular_component",
        "B-Complex",
        "B-Gene_or_gene_product",
        "B-Simple_chemical",
        "I-Cellular_component",
        "I-Complex",
        "I-Gene_or_gene_product",
        "I-Simple_chemical",
        "O",
    ],
    "BioNLP13PC-IOBES": [
        "B-Cellular_component",
        "B-Complex",
        "B-Gene_or_gene_product",
        "B-Simple_chemical",
        "E-Cellular_component",
        "E-Complex",
        "E-Gene_or_gene_product",
        "E-Simple_chemical",
        "I-Cellular_component",
        "I-Complex",
        "I-Gene_or_gene_product",
        "I-Simple_chemical",
        "O",
        "S-Cellular_component",
        "S-Complex",
        "S-Gene_or_gene_product",
        "S-Simple_chemical",
    ],
    "CRAFT-cc-IOB": ["B-GO", "I-GO", "O"],
    "CRAFT-cc-IOBES": ["B-GO", "E-GO", "I-GO", "O", "S-GO"],
    "CRAFT-cell-IOB": ["B-CL", "I-CL", "O"],
    "CRAFT-cell-IOBES": ["B-CL", "E-CL", "I-CL", "O", "S-CL"],
    "CRAFT-chem-IOB": ["B-CHEBI", "I-CHEBI", "O"],
    "CRAFT-chem-IOBES": ["B-CHEBI", "E-CHEBI", "I-CHEBI", "O", "S-CHEBI"],
    "CRAFT-ggp-IOB": ["B-GGP", "I-GGP", "O"],
    "CRAFT-ggp-IOBES": ["B-GGP", "E-GGP", "I-GGP", "O", "S-GGP"],
    "CRAFT-IOB": [
        "B-CHEBI",
        "B-CL",
        "B-GGP",
        "B-GO",
        "B-SO",
        "B-Taxon",
        "I-CHEBI",
        "I-CL",
        "I-GGP",
        "I-GO",
        "I-SO",
        "I-Taxon",
        "O",
    ],
    "CRAFT-IOBES": [
        "B-CHEBI",
        "B-CL",
        "B-GGP",
        "B-GO",
        "B-SO",
        "B-Taxon",
        "E-CHEBI",
        "E-CL",
        "E-GGP",
        "E-GO",
        "E-SO",
        "E-Taxon",
        "I-CHEBI",
        "I-CL",
        "I-GGP",
        "I-GO",
        "I-SO",
        "I-Taxon",
        "O",
        "S-CHEBI",
        "S-CL",
        "S-GGP",
        "S-GO",
        "S-SO",
        "S-Taxon",
    ],
    "CRAFT-species-IOB": ["B-NCBITaxon", "I-NCBITaxon", "O"],
    "CRAFT-species-IOBES": [
        "B-NCBITaxon",
        "E-NCBITaxon",
        "I-NCBITaxon",
        "O",
        "S-NCBITaxon",
    ],
    "Ex-PTM-IOB": ["B-Protein", "I-Protein", "O"],
    "Ex-PTM-IOBES": ["B-Protein", "E-Protein", "I-Protein", "O", "S-Protein"],
    "GENIA-pos": [
        "''",
        "(",
        ")",
        ",",
        ".",
        ":",
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNP",
        "NNPS",
        "NNS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
        "``",
    ],
    "JNLPBA": [
        "B-DNA",
        "B-RNA",
        "B-cell_line",
        "B-cell_type",
        "B-protein",
        "I-DNA",
        "I-RNA",
        "I-cell_line",
        "I-cell_type",
        "I-protein",
        "O",
    ],
    "JNLPBA-IOBES": [
        "B-DNA",
        "B-RNA",
        "B-cell_line",
        "B-cell_type",
        "B-protein",
        "E-DNA",
        "E-RNA",
        "E-cell_line",
        "E-cell_type",
        "E-protein",
        "I-DNA",
        "I-RNA",
        "I-cell_line",
        "I-cell_type",
        "I-protein",
        "O",
        "S-DNA",
        "S-RNA",
        "S-cell_line",
        "S-cell_type",
        "S-protein",
    ],
    "linnaeus-IOB": ["B-Species", "I-Species", "O"],
    "linnaeus-IOBES": ["B-Species", "E-Species", "I-Species", "O", "S-Species"],
    "NCBI-disease-IOB": ["B-Disease", "I-Disease", "O"],
    "NCBI-disease-IOBES": ["B-Disease", "E-Disease", "I-Disease", "O", "S-Disease"],
}


def _parse_file(file_path):
    sentences = []
    unique_tags = set()
    with open(file_path, "r") as fp:
        tokens = []
        tags = []
        for line in fp:
            if line.strip() == "":
                sentences.append((tokens, tags))
                tokens = []
                tags = []
                continue
            else:
                token, tag = [el.strip() for el in line.split("\t")]
                tokens.append(token)
                tags.append(tag)
                unique_tags.add(tag)
        if len(tokens) > 0:
            sentences.append((tokens, tags))

    unique_tags = sorted(list(unique_tags))
    return sentences, unique_tags


def _prep_class_labels(data_dir):
    class_labels = {}
    for config in CONFIGS:
        print(config)
        unique_tags = set()
        for split in ["train", "devel", "test"]:
            file_path = os.path.join(data_dir, config, f"{split}.tsv")
            sentences, split_tags = _parse_file(file_path)
            unique_tags.update(split_tags)

        unique_tags = sorted(list(unique_tags))
        class_labels[config] = unique_tags
    with open("crichton_2017_class_labels.json", "w") as fp:
        json.dump(class_labels, fp)


class Crichton2017Dataset(datasets.GeneratorBasedBuilder):
    """Crichton, Pyysalo, Chiu, Korhonen 2017 NER Dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=config,
            version=datasets.Version(_VERSION),
            description=config,
        )
        for config in CLASS_LABELS.keys()
    ]

    def _info(self):

        features = Features(
            {
                "id": Value("string"),
                "tokens": Sequence(Value("string")),
                "tags": Sequence(
                    datasets.features.ClassLabel(names=CLASS_LABELS[self.config.name])
                ),
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
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        base_path = os.path.join(self.config.data_dir, self.config.name)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": os.path.join(base_path, "train.tsv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"file_path": os.path.join(base_path, "devel.tsv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_path": os.path.join(base_path, "test.tsv")},
            ),
        ]

    def _generate_examples(self, file_path):
        """Generate samples using the info passed in from _split_generators."""

        sentences, split_tags = _parse_file(file_path)
        _id = 0
        for (tokens, tags) in sentences:
            yield _id, {
                "id": _id,
                "tokens": tokens,
                "tags": tags,
            }
            _id += 1
