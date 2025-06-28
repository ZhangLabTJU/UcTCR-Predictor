import logging
import gzip
import shutil
import inspect
import os, sys
import pandas as pd
import numpy as np
import torch
import json
import utils
import featurization as ft
from torch.utils.data import Dataset
from typing import *


class MLMDataset(Dataset):
    def __init__(self, seqs: Iterable[str], vgenes, tokenizer, round_len: bool = True):
        self.seqs = seqs
        self.vgenes = vgenes
        logging.info(
            f"Creating self supervised dataset with {len(self.seqs)} sequences"
        )
        self.max_len = max([len(s) for s in self.seqs])
        logging.info(f"Maximum sequence length: {self.max_len}")
        if round_len:
            self.max_len = int(utils.min_power_greater_than(self.max_len, 2))
            logging.info(f"Rounded maximum length to {self.max_len}")
        self.tokenizer = tokenizer
        self._has_logged_example = False

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        seq = self.seqs[i]
        retval = self.tokenizer.encode(ft.insert_whitespace(seq))
        retval.append(self.vgenes[i])
        if not self._has_logged_example:
            logging.info(f"Example of tokenized input: {seq} -> {retval}")
            self._has_logged_example = True
        output = np.array(retval)
        return output


class TcrFineTuneSingleDataset(Dataset):
    """Dataset for fine tuning from only TRA or TRB sequences"""

    def __init__(
        self,
        aa: Sequence[str],
        vgenes: Sequence[str],
        labels: MutableSequence[float],
        label_continuous: bool = False,
        label_labels: Optional[Sequence[str]] = None,
        drop_rare_labels: bool = True,
        drop_ratio: int = 1e-3
    ):
        assert len(aa) == len(
            labels
        ), f"Got differing lengths for aa and labels: {len(aa)}, {len(labels)}"
        self.aa = [ft.insert_whitespace(item) for item in aa]
        self.vgenes = vgenes
        self.tokenizer = ft.get_aa_bert_tokenizer(26)

        self.continuous = label_continuous
        label_dtype = np.float32 if self.continuous else np.int64
        self.labels = np.array(labels, dtype=label_dtype).squeeze()
        assert len(self.labels) == len(self.aa)
        self.label_labels = label_labels
        if self.continuous:
            assert self.label_labels is None

        if drop_rare_labels and not self.continuous and not self.is_multilabel:
            # Get the mean positive rate for each label
            labels_expanded = np.zeros((len(labels), np.max(labels) + 1))
            labels_expanded[np.arange(len(labels)), self.labels] = 1
            per_label_prop = np.mean(labels_expanded, axis=0)
            # Find the labels with high enough positive rate
            good_idx = np.where(per_label_prop >= drop_ratio)[0]
            if len(good_idx) < labels_expanded.shape[1]:
                logging.info(
                    f"Retaining {len(good_idx)}/{labels_expanded.shape[1]} labels with sufficient examples"
                )
                # Reconstruct labels based only on retained good_idx
                # nonzero returns indices of element that are nonzero
                self.labels = np.array(
                    [
                        np.nonzero(good_idx == label)[0][0]
                        if label in good_idx
                        else len(good_idx)  # "other" labels
                        for label in self.labels
                    ],
                    dtype=label_dtype,
                )
                assert np.max(self.labels) == len(good_idx)
                # Subset label labels
                self.label_labels = [self.label_labels[i] for i in good_idx] + ["other"]
                assert len(self.label_labels) == len(good_idx) + 1

    @property
    def is_multilabel(self) -> bool:
        """Return True if labels represent multilabel classification"""
        return len(self.labels.shape) > 1

    def get_ith_sequence(self, idx: int) -> str:
        """Get the ith sequence"""
        return self.aa[idx]

    def get_ith_label(self, idx: int) -> np.ndarray:
        """Gets the ith label"""
        return np.atleast_1d(self.labels[idx])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label = torch.tensor(self.get_ith_label(idx))
        if self.is_multilabel:
            # Multilabel -> BCEWithLogitsLoss which wants float target
            label = label.float()
        # We already inserted whitespaces in init
        enc = self.tokenizer(
            self.aa[idx], padding="max_length", max_length=26, return_tensors="pt"
        )
        enc = {k: v.squeeze() for k, v in enc.items()}
        enc["labels"] = label
        enc["vgene_ids"] = torch.tensor([self.vgenes[idx]]*len(enc["input_ids"]), dtype=torch.long)
        return enc

class DatasetSplit(Dataset):
    """
    Dataset split. Thin wrapper on top a dataset to provide data split functionality.
    Can also enable dynamic example generation for train fold if supported by
    the wrapped dataset (NOT for valid/test folds) via dynamic_training flag

    kwargs are forwarded to shuffle_indices_train_valid_test
    """

    def __init__(
        self,
        full_dataset: Dataset,
        split: str,
        dynamic_training: bool = False,
        **kwargs,
    ):
        self.dset = full_dataset
        split_to_idx = {"train": 0, "valid": 1, "test": 2}
        assert split in split_to_idx
        self.split = split
        self.dynamic = dynamic_training
        if self.split != "train":
            assert not self.dynamic, "Cannot have dynamic examples for valid/test"
        self.idx = shuffle_indices_train_valid_test(
            np.arange(len(self.dset)), **kwargs
        )[split_to_idx[self.split]]
        logging.info(f"Split {self.split} with {len(self)} examples")

    def all_labels(self, **kwargs) -> np.ndarray:
        """Get all labels"""
        if not hasattr(self.dset, "get_ith_label"):
            raise NotImplementedError("Wrapped dataset must implement get_ith_label")
        labels = [
            self.dset.get_ith_label(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return np.stack(labels)

    def all_sequences(self, **kwargs) -> Union[List[str], List[Tuple[str, str]]]:
        """Get all sequences"""
        if not hasattr(self.dset, "get_ith_sequence"):
            raise NotImplementedError(
                f"Wrapped dataset {type(self.dset)} must implement get_ith_sequence"
            )
        # get_ith_sequence could return a str or a tuple of two str (TRA/TRB)
        sequences = [
            self.dset.get_ith_sequence(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return sequences

    def to_file(self, fname: str, compress: bool = True) -> str:
        """
        Write to the given file
        """
        if not (
            hasattr(self.dset, "get_ith_label")
            and hasattr(self.dset, "get_ith_sequence")
        ):
            raise NotImplementedError(
                "Wrapped dataset must implement both get_ith_label & get_ith_sequence"
            )
        assert fname.endswith(".json")
        all_examples = []
        for idx in range(len(self)):
            seq = self.dset.get_ith_sequence(self.idx[idx])
            label_list = self.dset.get_ith_label(self.idx[idx]).tolist()
            all_examples.append((seq, label_list))

        with open(fname, "w") as sink:
            json.dump(all_examples, sink, indent=4)

        if compress:
            with open(fname, "rb") as source:
                with gzip.open(fname + ".gz", "wb") as sink:
                    shutil.copyfileobj(source, sink)
            os.remove(fname)
            fname += ".gz"
        assert os.path.isfile(fname)
        return os.path.abspath(fname)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, idx: int):
        if (
            self.dynamic
            and self.split == "train"
            and "dynamic" in inspect.getfullargspec(self.dset.__getitem__).args
        ):
            return self.dset.__getitem__(self.idx[idx], dynamic=True)
        return self.dset.__getitem__(self.idx[idx])


def shuffle_indices_train_valid_test(
    idx: np.ndarray, valid: float = 0.15, test: float = 0.15, seed: int = 1234
) -> Tuple[np.ndarray]:
    """
    Given an array of indices, return indices partitioned into train, valid, and test indices
    The following tests ensure that ordering is consistent across different calls
    >>> np.all(shuffle_indices_train_valid_test(np.arange(100))[0] == shuffle_indices_train_valid_test(np.arange(100))[0])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(10000))[1] == shuffle_indices_train_valid_test(np.arange(10000))[1])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(20000))[2] == shuffle_indices_train_valid_test(np.arange(20000))[2])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1] == shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1])
    True
    """
    np.random.seed(seed)  # For reproducible subsampling
    indices = np.copy(idx)  # Make a copy because shuffling occurs in place
    np.random.shuffle(indices)  # Shuffles inplace
    num_valid = int(round(len(indices) * valid)) if valid > 0 else 0
    num_test = int(round(len(indices) * test)) if test > 0 else 0
    num_train = len(indices) - num_valid - num_test
    assert num_train > 0 and num_valid >= 0 and num_test >= 0
    assert num_train + num_valid + num_test == len(
        indices
    ), f"Got mismatched counts: {num_train} + {num_valid} + {num_test} != {len(indices)}"

    indices_train = indices[:num_train]
    indices_valid = indices[num_train : num_train + num_valid]
    indices_test = indices[-num_test:]
    assert indices_train.size + indices_valid.size + indices_test.size == len(idx)

    return indices_train, indices_valid, indices_test

