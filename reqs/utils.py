"""
This file contains some utility functions like parsing the config file, converting objects, etc
"""

import re
import itertools
from typing import List, Dict, Callable, Iterable
import string

def parse_sh_args(filepath):
    """Parses a shell script line into a dictionary of arguments.

    Args:
        sh_line: The shell script line to parse.

    Returns:
        A dictionary containing the parsed arguments.
    """

    # Read whole file
    with open(filepath, "r") as file:
        sh_line = file.read()
        
    # Skip comments
    sh_line = re.sub(r"^\s*#.*$", "", sh_line, flags=re.MULTILINE)

    # Parse arguments, format is --name value
    # if no value is provided (\ is observed), it is set to True
    args = {}
    for match in re.finditer(r"--(\S+)(?:\s+(\S+))?", sh_line):
        name, value = match.groups()
        args[name] = value if value is not None else True

        # replace \\ with True boolean
        if args[name] == "\\":
            args[name] = True

        # Convert to number if possible
        try:
            args[name] = int(args[name])
        except ValueError:
            try:
                args[name] = float(args[name])
            except ValueError:
                pass

    return args

# All of the below functions are obtained from here, we provide our explanation for each of them
# https://github.com/huggingface/transformers/blob/9b0a8ea7d1d6226b76cfdc645ce65e21157e2b50/examples/research_projects/rag/utils_rag.py#

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def flatten_list(summary_ids: List[List]):
    return list(itertools.chain.from_iterable(summary_ids))

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def calculate_exact_match(output_lns: List[str], reference_lns: List[str]) -> Dict:
    assert len(output_lns) == len(reference_lns)
    em = 0
    for hypo, pred in zip(output_lns, reference_lns):
        em += exact_match_score(hypo, pred)
    if len(output_lns) > 0:
        em /= len(output_lns)
    return {"em": em}