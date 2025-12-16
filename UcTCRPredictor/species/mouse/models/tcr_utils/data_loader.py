import os
import sys
import logging
import collections
import itertools
import multiprocessing
import numpy as np
import pandas as pd
import utils
import featurization as ft
from typing import *

path_root = os.path.join(os.getcwd(), "..") # root directory
LOCAL_DATA_DIR = os.path.join(path_root, "data")
# LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname("__file__")), "data")

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


def load_vdjdb(
    fname: str = os.path.join(LOCAL_DATA_DIR, "vdjdb-2021-02-02", "vdjdb.slim.txt"),
    species_filter: Optional[Iterable[str]] = ["MusMusculus", "HomoSapiens"],
    tra_trb_filter: Optional[Iterable[str]] = ["TRA", "TRB"],
    drop_null: bool = True,
    vocab_check: bool = True,
) -> pd.DataFrame:
    """
    Load VDJdb as a dataframe. 'cdr3' column is the column containing sequences
    ~62k examples, spanning 352 distinct antigens
    """
    df = pd.read_csv(fname, sep="\t")
    if species_filter is not None:
        logging.info(f"Filtering VDJdb species to: {species_filter}")
        keep_idx = [i for i in df.index if df.loc[i, "species"] in species_filter]
        df = df.loc[keep_idx]
    logging.info(f"Species distribution: {collections.Counter(df['species'])}")
    if drop_null:
        keep_idx = [~pd.isnull(aa) for aa in df["cdr3"]]
        logging.info(
            f"VDJdb: dropping {np.sum(keep_idx==False)} entries for null cdr3 sequence"
        )
        df = df.iloc[np.where(keep_idx)]
    if vocab_check:
        pass_idx = np.array([ft.adheres_to_vocab(aa) for aa in df["cdr3"]])
        logging.info(
            f"VDJdb: dropping {np.sum(pass_idx==False)} entries for unrecognized AAs"
        )
        df = df.iloc[np.where(pass_idx)]
    nonnull_antigens_df = df.loc[~pd.isnull(df["antigen.epitope"])]
    logging.info(
        f"Entries with antigen sequence: {nonnull_antigens_df.shape[0]}/{df.shape[0]}"
    )
    logging.info(
        f"Unique antigen sequences: {len(set(nonnull_antigens_df['antigen.epitope']))}"
    )
    if tra_trb_filter is not None:
        logging.info(f"Filtering TRA/TRB to: {tra_trb_filter}")
        keep_idx = [i for i in df.index if df.loc[i, "gene"] in tra_trb_filter]
        df = df.loc[keep_idx]
    ab_counter = collections.Counter(df["gene"])
    logging.info(f"TRA: {ab_counter['TRA']} | TRB: {ab_counter['TRB']}")
    return df


def load_pird(
    fname: str = os.path.join(LOCAL_DATA_DIR, "pird", "pird_tcr_ab.csv"),
    tra_trb_only: bool = True,
    vocab_check: bool = True,
    with_antigen_only: bool = False,
) -> pd.DataFrame:
    """
    Load PIRD (pan immune repertoire database) TCR A/B data
    https://db.cngb.org/pird/tbadb/
    For TRA we want the column CDR3.alpha.aa
    For TRB we want the column CDR3.beta.aa
    The PIRD dataset also has ~8k examples with antigens (73 unique)
    """
    if not tra_trb_only:
        raise NotImplementedError
    df = pd.read_csv(fname, encoding='ISO-8859-1', na_values="-", low_memory=False)
    # df_orig = pd.read_csv(fname, na_values="-", low_memory=False)
    # df = df_orig.dropna(axis=0, how="all", subset=["CDR3.alpha.aa", "CDR3.beta.aa"])
    # logging.info(
    #     f"Dropped {len(df_orig) - len(df)} entires with null sequence in both TRA/TRB"
    # )
    antigen_null_rate = np.sum(pd.isnull(df["Antigen.sequence"])) / df.shape[0]
    logging.info(
        f"PIRD data {1.0 - antigen_null_rate:.4f} data labelled with antigen sequence"
    )
    # Filter out entries that have weird characters in their aa sequences
    if vocab_check:
        tra_pass = [
            pd.isnull(aa) or ft.adheres_to_vocab(aa) for aa in df["CDR3.alpha.aa"]
        ]
        trb_pass = [
            pd.isnull(aa) or ft.adheres_to_vocab(aa) for aa in df["CDR3.beta.aa"]
        ]
        both_pass = np.logical_and(tra_pass, trb_pass)
        logging.info(
            f"PIRD: Removing {np.sum(both_pass == False)} entires with non amino acid residues"
        )
        df = df.iloc[np.where(both_pass)]
    # Collect instances where we have antigen information
    nonnull_antigens_df = df.loc[~pd.isnull(df["Antigen.sequence"])]
    nonnull_antigens = nonnull_antigens_df["Antigen.sequence"]
    logging.info(
        f"Entries with antigen sequence: {len(nonnull_antigens)}/{df.shape[0]}"
    )
    logging.info(f"Unique antigen sequences: {len(set(nonnull_antigens))}")
    logging.info(f"PIRD data TRA/TRB instances: {collections.Counter(df['Locus'])}")
    retval = nonnull_antigens_df if with_antigen_only else df

    # Report metrics
    # print(df.loc[:, ["CDR3.alpha.aa", "CDR3.beta.aa"]])
    has_tra = ~pd.isnull(df["CDR3.alpha.aa"])
    has_trb = ~pd.isnull(df["CDR3.beta.aa"])
    has_both = np.logical_and(has_tra, has_trb)
    logging.info(f"PIRD entries with TRB sequence: {np.sum(has_tra)}")
    logging.info(f"PIRD entries with TRB sequence: {np.sum(has_trb)}")
    logging.info(f"PIRD entries with TRA and TRB:  {np.sum(has_both)}")
    # print(retval.iloc[np.where(has_both)[0]].loc[:, ["CDR3.alpha.aa", "CDR3.beta.aa"]])

    return retval


def load_lcmv_table(
    fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_tetramer_tcr.txt"),
    metadata_fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_all_metadata.txt.gz"),
    vdj_fname: str = os.path.join(LOCAL_DATA_DIR, "lcmv_tcr_vdj_unsplit.txt.gz"),
    drop_na: bool = True,
    drop_unsorted: bool = True,
) -> pd.DataFrame:
    """Load the LCMV data table"""
    table = pd.read_csv(fname, delimiter="\t")
    logging.info(f"Loaded in table of {len(table)} entries")
    if drop_na:
        table.dropna(axis=0, how="any", subset=["tetramer", "TRB", "TRA"], inplace=True)
        logging.info(f"{len(table)} entries remain after dropping na")
    if drop_unsorted:
        drop_idx = table.index[table["tetramer"] == "Unsorted"]
        table.drop(index=drop_idx, inplace=True)
        logging.info(f"{len(table)} entries remain after dropping unsorted")

    # Take entires with multiple TRA or TRB sequences and split them, carrying over
    # all of the other metadata to each row
    dedup_rows = []
    for _i, row in table.iterrows():
        for tra, trb in itertools.product(row["TRA"].split(";"), row["TRB"].split(";")):
            new_row = row.copy(deep=True)
            new_row["TRA"] = tra
            new_row["TRB"] = trb
            dedup_rows.append(new_row)
    dedup_table = pd.DataFrame(dedup_rows, columns=table.columns)
    logging.info(f"{len(dedup_table)} entries after expanding multiple entries")
    gp33_antigen = utils.read_newline_file(
        os.path.join(os.path.dirname(fname), "lcmv_antigen.txt")
    ).pop()
    dedup_table["antigen.sequence"] = gp33_antigen  # gp33 tetramer

    # Load metadata and match it up with prior table
    metadata_df = pd.read_csv(metadata_fname, delimiter="\t", low_memory=False)
    if drop_na:
        metadata_df.dropna(axis=0, how="any", subset=["TRA", "TRB"], inplace=True)
    table_ab_pairs = list(dedup_table["tcr_cdr3s_aa"])
    metadata_ab_pairs = list(metadata_df["tcr_cdr3s_aa"])
    idx_map = np.array([metadata_ab_pairs.index(p) for p in table_ab_pairs])
    metadata_df_reorder = metadata_df.iloc[idx_map]
    assert (
        all(
            [
                i == j
                for i, j in zip(
                    metadata_df_reorder["tcr_cdr3s_aa"], dedup_table["tcr_cdr3s_aa"]
                )
            ]
        )
        and metadata_df_reorder.shape[0] == dedup_table.shape[0]
    )
    metadata_df_reorder = metadata_df_reorder.drop(
        columns=[
            col for col in metadata_df_reorder.columns if col in dedup_table.columns
        ]
    )
    metadata_df_reorder.index = dedup_table.index

    # Load in VDJ annotations and match it up with prior table
    vdj_df = pd.read_csv(vdj_fname, delimiter="\t", low_memory=False)
    vdj_ab_pairs = list(vdj_df["tcr_cdr3s_aa"])
    idx_map = np.array([vdj_ab_pairs.index(p) for p in table_ab_pairs])
    vdj_df_reorder = vdj_df.iloc[idx_map]
    assert all(
        [
            i == j
            for i, j in zip(vdj_df_reorder["tcr_cdr3s_aa"], dedup_table["tcr_cdr3s_aa"])
        ]
    )
    vdj_df_reorder = vdj_df_reorder.drop(
        columns=[
            col
            for col in vdj_df_reorder.columns
            if col in dedup_table.columns or col in metadata_df_reorder.columns
        ]
    )
    vdj_df_reorder.index = dedup_table.index
    retval = pd.concat([dedup_table, metadata_df_reorder, vdj_df_reorder], axis=1)

    # Check that the TRA/TRB are the same as the "dedup_table" object that we were previously returning
    assert all([i == j for i, j in zip(retval["TRA"], dedup_table["TRA"])])
    assert all([i == j for i, j in zip(retval["TRB"], dedup_table["TRB"])])

    # Report basic metadata
    cnt = collections.Counter(dedup_table["tetramer"])
    for k, v in cnt.items():
        logging.info(f"Class {k}: {v}")

    return retval

def _tcrdb_df_to_entries(fname: str) -> List[tuple]:
    """Helper function for processing TCRdb tables"""

    def tra_trb_from_str(s: str) -> str:
        if s.startswith("TRA"):
            return "TRA"
        elif s.startswith("TRB"):
            return "TRB"
        return "UNK"

    def infer_row_tra_trb(row) -> str:
        """Takes in a row from itertuples and return inferred TRA/TRB"""
        infers = []
        if "Vregion" in row._fields:
            infers.append(row.Vregion)
        if "Dregion" in row._fields:
            infers.append(row.Dregion)
        if "Jregion" in row._fields:
            infers.append(row.Jregion)
        if len(infers) == 0:
            return "UNK"
        return infers
        # Use majority voting
        cnt = collections.Counter(infers)
        consensus, consensus_prop = cnt.most_common(1).pop()
        # if consensus_prop / len(infers) > 0.5:
        #     return consensus
        # return "UNK"  # No majority

    acc = os.path.basename(fname).split(".")[0]
    df = pd.read_csv(fname, delimiter="\t")
    entries = [
        (acc, row.RunId, row.AASeq, row.cloneFraction, infer_row_tra_trb(row))
        for row in df.itertuples(index=False)
    ]
    return entries

def load_tcrdb(
    dirname: str = os.path.join(LOCAL_DATA_DIR, "tcrdb"),
    drop_unk: bool = True,
    vocab_check: bool = True,
) -> pd.DataFrame:
    """
    Load TCRdb
    https://academic.oup.com/nar/article/49/D1/D468/5912818
    http://bioinfo.life.hust.edu.cn/TCRdb/#/
    """

    accessions_list_fname = os.path.join(dirname, "tcrdb_accessions_21_03_22.txt")
    with open(accessions_list_fname, "r") as source:
        accessions = [line.strip() for line in source if not line.startswith("#")]
    # Load in each accession
    accession_fnames = [os.path.join(dirname, f"{acc}.tsv.gz") for acc in accessions]
    pool = multiprocessing.Pool(8)
    entries = pool.map(_tcrdb_df_to_entries, accession_fnames)
    pool.close()
    pool.join()
    retval = pd.DataFrame(
        itertools.chain.from_iterable(entries),
        columns=["accession", "RunId", "AASeq", "cloneFraction", "vdj" ],
    )
    if drop_unk:
        drop_idx = np.where(retval["vdj"] == "UNK")[0]
        logging.info(
            f"Dropping {len(drop_idx)} TCRdb entries for unknown TRA TRB status"
        )
        retval.drop(index=drop_idx, inplace=True)
    if vocab_check:
        is_valid_aa = np.array([ft.adheres_to_vocab(aa) for aa in retval["AASeq"]])
        logging.info(
            f"TCRdb: Removing {np.sum(is_valid_aa == False)} entries with non-amino acid residues"
        )
        retval = retval.iloc[np.where(is_valid_aa)]
    return retval


def load_CMV():
    HIPs = []
    filePath = os.path.join(LOCAL_DATA_DIR,'CMV_TCRB') # HLA数据地址
    filenames = os.listdir(filePath)

    for filename in filenames:
        if filename[0:3]=="HIP":
            HIPs.append(filename) 
    print("HIPs len: ", len(HIPs)) # HIP的数据数量
    HIPs.sort()

    cdrs = []
    vs = []
    for filename in HIPs:
        try:
            TCR_table=pd.read_table(os.path.join(filePath,filename) ,header=0)
            # print("success:", filename)
            cdr3aa = TCR_table['cdr3aa'].values
            v = TCR_table['v'].values
            vs.extend(v)
            cdrs.extend(cdr3aa)
        except:
            # TCR_table=pd.read_table(os.path.join(filePath,filename) ,header=0)
            # cdr3aa = TCR_table['amino_acid'].values
            # v = TCR_table['v_gene'].values
            # cdrs.extend(cdr3aa)
            # vs.extend(v)
            print("error:", filename)
    return cdrs,vs

def load_Mouseprocessed() -> pd.DataFrame:
    """Load in the Mouse_sampled dataset"""
    df = pd.read_csv('/data1/mouse_process/mouse_processed', low_memory=False, header="infer")
    for i in range(1,12):
        df['Vgene']=df['Vgene'].str.replace('TRBV'+str(i)+'-1','TRBV'+str(i))
    for i in range(14,32):
        df['Vgene']=df['Vgene'].str.replace('TRBV'+str(i)+'-1','TRBV'+str(i))
    cdrs = []
    vs = []
    cdr3aa = df['CDR3'].values
    v = df['Vgene'].values
    vs.extend(v)
    cdrs.extend(cdr3aa)
    return cdrs,vs