import os
import tempfile
import logging
import numpy as np
from typing import *
from transformers import BertTokenizer
from math import floor
import utils


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PAD = "$"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"
AMINO_ACIDS_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AMINO_ACIDS_WITH_ALL_ADDITIONAL = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS
AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX = {
    aa: i for i, aa in enumerate(AMINO_ACIDS_WITH_ALL_ADDITIONAL)
}


AA_TRIPLET_TO_SINGLE = {
    "ARG": "R",
    "HIS": "H",
    "LYS": "K",
    "ASP": "D",
    "GLU": "E",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "CYS": "C",
    "GLY": "G",
    "PRO": "P",
    "ALA": "A",
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
}
AA_SINGLE_TO_TRIPLET = {v: k for k, v in AA_TRIPLET_TO_SINGLE.items()}

def get_aa_bert_tokenizer(
    max_len: int, d=AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX
) -> BertTokenizer:
    """
    Tokenizer for amino acid sequences. Not *exactly* the same as BertTokenizer
    but mimics its behavior, encoding start with CLS and ending with SEP
    >>> get_aa_bert_tokenizer(10).encode(insert_whitespace("RKDES"))
    [25, 0, 2, 3, 4, 5, 24]
    """
    with tempfile.TemporaryDirectory() as tempdir:
        vocab_fname = write_vocab(d, os.path.join(tempdir, "vocab.txt"))
        tok = BertTokenizer(
            vocab_fname,
            do_lower_case=False,
            do_basic_tokenize=True,
            tokenize_chinese_chars=False,
            pad_token=PAD,
            mask_token=MASK,
            unk_token=UNK,
            sep_token=SEP,
            cls_token=CLS,
            model_max_len=max_len,
            padding_side="right",
        )
    return tok

def write_vocab(vocab: Iterable[str], fname: str) -> str:
    """
    Write the vocabulary to the fname, one entry per line
    Mostly for compatibility with transformer BertTokenizer
    """
    with open(fname, "w") as sink:
        for v in vocab:
            sink.write(v + "\n")
    return fname

def pad_or_trunc_sequence(seq: str, l: int, right_align: bool = False, pad=PAD) -> str:
    """
    Pad the given sequence to the given length
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=False)
    'RKDES$$$'
    >>> pad_or_trunc_sequence("RKDES", 8, right_align=True)
    '$$$RKDES'
    >>> pad_or_trunc_sequence("RKDESRKRKR", 3, right_align=False)
    'RKD'
    >>> pad_or_trunc_sequence("RKDESRRK", 3, right_align=True)
    'RRK'
    """
    delta = len(seq) - l
    if len(seq) > l:
        if right_align:
            retval = seq[delta:]
        else:
            retval = seq[:-delta]
    elif len(seq) < l:
        insert = pad * np.abs(delta)
        if right_align:
            retval = insert + seq
        else:
            retval = seq + insert
    else:
        retval = seq
    assert len(retval) == l, f"Got mismatched lengths: {len(retval)} {l}"
    return retval

def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))

def adheres_to_vocab(s: str, vocab: str = AMINO_ACIDS) -> bool:
    """
    Returns whether a given string contains only characters from vocab
    >>> adheres_to_vocab("RKDES")
    True
    >>> adheres_to_vocab(AMINO_ACIDS + AMINO_ACIDS)
    True
    """
    return set(s).issubset(set(vocab))

def insert_whitespace(seq: str) -> str:
    """
    Return the sequence of characters with whitespace after each char
    >>> insert_whitespace("RKDES")
    'R K D E S'
    """
    return " ".join(list(seq))

def is_whitespaced(seq: str) -> bool:
    """
    Return whether the sequence has whitespace inserted
    >>> is_whitespaced("R K D E S")
    True
    >>> is_whitespaced("RKDES")
    False
    >>> is_whitespaced("R K D ES")
    False
    >>> is_whitespaced("R")
    True
    >>> is_whitespaced("RK")
    False
    >>> is_whitespaced("R K")
    True
    """
    tok = list(seq)
    spaces = [t for t in tok if t.isspace()]
    if len(spaces) == floor(len(seq) / 2):
        return True
    return False

def one_hot(seq: str, alphabet: Optional[str] = AMINO_ACIDS) -> np.ndarray:
    """
    One-hot encode the input string. Since pytorch convolutions expect
    input of (batch, channel, length), we return shape (channel, length)
    When one hot encoding, we ignore the pad characters, encoding them as
    a vector of 0's instead
    """
    if not seq:
        assert alphabet
        return np.zeros((len(alphabet), 1), dtype=np.float32)
    if not alphabet:
        alphabet = utils.dedup(seq)
        logging.info(f"No alphabet given, assuming alphabet of: {alphabet}")
    seq_arr = np.array(list(seq))
    # This implementation naturally ignores the pad character if not provided
    # in the alphabet
    retval = np.stack([seq_arr == char for char in alphabet]).astype(float).T
    assert len(retval) == len(seq), f"Mismatched lengths: {len(seq)} {retval.shape}"
    return retval.astype(np.float32).T