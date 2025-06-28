import json
import itertools
import pandas as pd
import featurization as ft

def load_blosum(
    fname: str = "utils/blosum62.json"
) -> pd.DataFrame:
    """Return the blosum matrix as a dataframe"""
    with open(fname) as source:
        d = json.load(source)
        retval = pd.DataFrame(d)
    retval = pd.DataFrame(0, index=list(ft.AMINO_ACIDS), columns=list(ft.AMINO_ACIDS))
    for x, y in itertools.product(retval.index, retval.columns):
        retval.loc[x, y] = d[x][y]
    return retval

if __name__ == "__main__":
    print(load_blosum("utils/blosum62.json"))


