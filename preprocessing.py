import itertools

# We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling
# (see below) is more efficient when it receives the `special_tokens_mask`.
def tokenize_map(examples, tokenizer, text_col, return_special_tokens_mask=False):
    return tokenizer(
        examples[text_col], return_special_tokens_mask=return_special_tokens_mask
    )


def group_texts_map(examples, max_seq_len):
    """Concatenate and chunk to max_seq_len all dataset columns.

    A robustly commented version of the script from hugging face mlm example
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm_no_trainer.py

    """

    # input examples is a batch of tokenizer output.
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

    concatenated_examples = {
        key: list(itertools.chain(*examples[key])) for key in examples.keys()
    }

    # we want the total length of the concatenated sequences
    # all of the keys map to lists of the same length
    # so grab an arbitrary key and get the length of that value

    arbitrary_key = list(examples.keys())[0]
    total_length = len(concatenated_examples[arbitrary_key])

    # truncate the remainder
    # [max_seq_len    ][max_seq_len    ][max_seq_len    ]...[remain]
    # ^ keep           ^ keep           ^ keep              ^ remove

    num_chunks = 1
    if total_length > max_seq_len:
        num_chunks = total_length // max_seq_len
        total_length = num_chunks * max_seq_len

    # result stores num_chunks chunks each of length max_seq_len (example for max_seq_len=128)
    # {
    #     "input_ids": [
    #         [id_0, ..., id_127],
    #         [id_128, ..., id_255],
    #         [id_256, ..., id_383],
    #         [id_384, ..., id_511],
    #         ...
    #     ],
    #     "attention_mask": [[...], [...], [...], ...],
    #     "special_tokens_mask": [[...], [...], [...], ...],
    # }

    result = {
        key: [
            concat_seq[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)
        ]
        for key, concat_seq in concatenated_examples.items()
    }

    # labels becomes a copy of input_ids
    # masking and/or shifting for causal language modelling is handled later
    result["labels"] = result["input_ids"].copy()

    return result
