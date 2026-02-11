"""
Merged utility functions and classes for MoE knowledge editing
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer


@dataclass
class EditRequest:
    """Single knowledge edit request"""
    prompt: str          # "The capital of {} is"
    subject: str         # "France"
    target_new: str      # " Berlin"
    case_id: int
    prefixes: Optional[List[str]] = None  


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=False,
) -> int:
    ret = None

    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)

    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret


def get_words_idxs_in_templates(
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    subtoken: str
) -> List[List[int]]:

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []

    for i, context in enumerate(context_templates):
        prefix, suffix = context.split("{}")

        prefix_len = len(tok.encode(prefix))
        prompt_len = len(tok.encode(prefix + words[i]))
        input_len = len(tok.encode(prefix + words[i] + suffix))

        prefixes_len.append(prefix_len)
        words_len.append(prompt_len - prefix_len)
        suffixes_len.append(input_len - prompt_len)
        inputs_len.append(input_len)

    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(len(context_templates))
        ]
    elif subtoken == "first":
        return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")

def get_tqdm_kwargs(**override_kwargs):
    """Get standardized tqdm configuration"""
    default_kwargs = {
        "disable": False,
        "ascii": True,
        "ncols": 80,
        "dynamic_ncols": False,
        "leave": True,
        "file": None,
    }
    default_kwargs.update(override_kwargs)
    return default_kwargs