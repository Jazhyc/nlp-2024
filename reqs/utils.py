"""
This file contains some utility functions like parsing the config file, converting objects, etc
"""

import re
import itertools
from typing import List, Dict, Callable, Iterable
import string
import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from transformers import BartTokenizer

logger = logging.getLogger(__name__)

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

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    
def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_checkpoint_callback(output_dir, metric):
    """Saves the best model by validation EM score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "em":
        exp = "{val_avg_em:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2 and bleu, got {metric}, You can make your own by adding to this"
            " function."
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=exp,
        monitor=f"val_{metric}",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )


class Seq2SeqLoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
    ) -> None:
        logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        # Uncommenting this will save val generations
        # return self._write_logs(trainer, pl_module, "valid")