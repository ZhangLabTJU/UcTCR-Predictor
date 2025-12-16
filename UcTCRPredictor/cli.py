import argparse, sys
from importlib import import_module
from .common import logger

def _main():
    p = argparse.ArgumentParser(prog="UcTCRPredictor", description="TCR cell-type predictor")
    p.add_argument("--species", choices=["human","mouse"], required=True)
    p.add_argument("infile",  help="CSV/TSV path")
    p.add_argument("-o","--out", help="output .tsv.gz path (optional)")
    p.add_argument("--batch", type=int, default=1024, help="batch size")
    args = p.parse_args()

    mod = import_module(f"UcTCRPredictor.species.{args.species}.predict")
    func_name = "ucpredict_human" if args.species == "human" else "ucpredict_mouse"
    predict_fn = getattr(mod, func_name)

    df  = predict_fn(args.infile, batch_size=args.batch, save_path=args.out)
    
    logger.info("Prediction shape: %s", df.shape)

if __name__ == "__main__":
    _main()
