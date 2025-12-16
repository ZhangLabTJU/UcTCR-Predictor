"""
UcTCRPredictor.species.human.predict
-----------------------------------
function:
    ucpredict_human(csv_path: str | os.PathLike, *,
                 batch_size: int = 1024) -> pandas.DataFrame
usage:
    >>> from UcTCRPredictor.species.human import ucpredict_human
    >>> df = ucpredict_human("test_data.csv").head()
"""

from __future__ import annotations
import os, re, time, logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from importlib import resources

from UcTCRPredictor.common import logger, get_model_dir         
from ...modelhub import ensure_model

# --------------------------------------------------------------------- #
# 1. Resources path
# --------------------------------------------------------------------- #
#MODEL_DIR = get_model_dir("human")                   # .../species/human/models
MODEL_DIR = ensure_model("human")
V_VOCAB   = np.load(MODEL_DIR / "v_vocab.npy", allow_pickle=True).tolist()
MODEL_PATH = MODEL_DIR / "human_cls.pt"
VGENE_LIST = pd.read_csv(MODEL_DIR / "HUMAN_V_Gene_list.csv")      

# V gene map
V_DICT = dict(VGENE_LIST.iloc[:, -2:].values)
V_DICT.update({                 # The unconventional V gene
    'TRBV3-1':'TCRBV03-01', 
    'TRBV23-1':'TCRBV23',
    'TRBV12-3':'TCRBV12-03',
    'TRBV12-4':'TCRBV12-03',
    'TRBV6-2':'TCRBV06-02',
    'TRBV6-3':'TCRBV06-02',
}) 

# --------------------------------------------------------------------- #
# 2. Model & Data tool
# --------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_utils():
    utils_dir = MODEL_DIR / "tcr_utils"   
    if not utils_dir.exists():
        raise FileNotFoundError(f"tcr_utils dir not found: {utils_dir}")

    import sys, importlib

    sys.path.insert(0, str(utils_dir))
    sys.path.insert(0, str(utils_dir.parent))   #  `import tcr_utils` 
    importlib.invalidate_caches() 
    utils_mod = importlib.import_module("tcr_utils.utils")
    return utils_mod
    #tu_pkg = importlib.import_module("tcr_utils")    
    #return tu_pkg.utils   

_UTILS = _load_utils()

def _regex_valid(seq: str) -> bool:
    return bool(re.fullmatch(r"C[ACDEFGHIKLMNPQRSTVWY]{6,24}F", seq))

class _EmbedDataset(Dataset):
    def __init__(self, arr: np.ndarray):
        self.arr = arr
    def __len__(self):  return len(self.arr)
    def __getitem__(self, idx): return torch.from_numpy(self.arr[idx]).float()

class _FCModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 768 → 256 → 64 → 4
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),     
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        return self.classifier(x)

# --------------------------------------------------------------------- #
# 3. Main function
# --------------------------------------------------------------------- #
def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"Vgene","cdr3aa"}.issubset(df.columns):
        raise ValueError("Input must contain Vgene & cdr3aa columns")
    logger.info("Loaded %d rows", len(df))
    return df

def _filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = df[df["Vgene"].isin(V_DICT.values())].dropna(subset=["cdr3aa","Vgene"])
    df["cdr3aa"] = df["cdr3aa"].str.split(";").str[0]
    df = df[df["cdr3aa"].map(_regex_valid)]
    df = df[df["Vgene"].isin(V_VOCAB)]

    v_map = {v: i for i, v in enumerate(V_VOCAB)}
    v_indices = df["Vgene"].map(v_map).values
    return df, df["cdr3aa"].values, torch.tensor(v_indices)

def _embed(cdrs: np.ndarray, vgenes: torch.Tensor) -> np.ndarray:
    from tcr_utils.featurization import get_aa_bert_tokenizer
    max_len = int(_UTILS.min_power_greater_than(25, base=2)) 
    tok = get_aa_bert_tokenizer(max_len)
    emb = _UTILS.get_transformer_embeddings(
        model_dir=MODEL_DIR,
        tok=tok, seqs=cdrs, vgene=vgenes, layers=[-4],
        method="mean", device=0, max_len=max_len
    )
    return emb

def _predict(emb: np.ndarray, batch: int) -> np.ndarray:
    mdl = _FCModel().to(device)
    mdl.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    mdl.eval()

    dl = DataLoader(_EmbedDataset(emb), batch_size=batch)
    out = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            logits = mdl(x)
            out.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(out).numpy()

# --------------------------------------------------------------------- #
# 4. Public API
# --------------------------------------------------------------------- #
def ucpredict_human(csv_path: str | os.PathLike,
                 *,
                 batch_size: int = 1024,
                 save_path: str | os.PathLike | None = None
                 ) -> pd.DataFrame:
    """
    Read single CSV/TSV, output DataFrame with prediction.
    If `save_path` set, then save `.tsv.gz`.

    Returns
    -------
    pandas.DataFrame
        origin + ['Conv','MAIT','iNKT']
    """
    t0 = time.time()
    csv_path = Path(csv_path).expanduser().resolve()
    df_raw  = _load_csv(csv_path)
    df_used, cdr3s, vgenes = _filter(df_raw)

    logger.info("Embedding %d sequences", len(df_used))
    emb = _embed(cdr3s, vgenes)

    logger.info("Predicting on embeddings")
    prob = _predict(emb, batch_size)
    prob_df = pd.DataFrame(prob, index=df_used.index,
                           columns=["Conv","MAIT","iNKT","AV25_Conv"])
    prob_df['Conv'] = prob_df['Conv'] + prob_df['AV25_Conv']
    prob_df.drop(columns=['AV25_Conv'], inplace=True)
    df_out = pd.concat([df_raw, prob_df], axis=1)

    if save_path:
        save_path = Path(save_path).with_suffix(".tsv.gz")
        df_out.to_csv(save_path, index=False)
        logger.info("Saved → %s", save_path)

    logger.info("Finished in %.1fs", time.time() - t0)
    return df_out
