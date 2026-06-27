"""Utilities for sag_refseg (ported from kdwonn/SaG utils.py + local checkpoint conventions)."""
from __future__ import absolute_import, division, print_function

import json
import logging
import os
import os.path as osp
import random
import re
import shutil

import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.val)
        return '%.4f (%.4f)' % (self.val, self.avg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir, name="sag_refseg"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(osp.join(log_dir, "train.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger


def save_checkpoint(model, hparams, out_dir, epoch, is_best=False, filename="latest.pth"):
    """Save checkpoint in local proto-VLM convention: {'state_dict': ..., 'hparams': ...}."""
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, filename)
    torch.save({'state_dict': model.state_dict(), 'hparams': hparams, 'epoch': epoch}, path)
    if is_best:
        best_path = osp.join(out_dir, "best.pth")
        shutil.copyfile(path, best_path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint saved by save_checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    # Support both local format and upstream format
    state_dict = ckpt.get('state_dict', ckpt.get('model'))
    model.load_state_dict(state_dict)
    epoch = ckpt.get('epoch', 0)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return epoch


def update_training_history(history_path, epoch, metrics):
    """Append per-epoch metrics to training_history.json (local convention)."""
    history = []
    if osp.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    history.append({'epoch': epoch, **metrics})
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


# Vocabulary utilities (ported from upstream)
UNK_IDENTIFIER = '<unk>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f]
    return {w: i for i, w in enumerate(words)}


def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [
        (vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words
    ]
    return vocab_indices


PAD_IDENTIFIER = '<pad>'
EOS_IDENTIFIER = '<eos>'


def preprocess_sentence(sentence, vocab_dict, T):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices


def f_out_hook(l):
    return lambda m, i, o: l.append(o)
