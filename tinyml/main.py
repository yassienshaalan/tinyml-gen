
import argparse
import logging
import sys
from pathlib import Path

from experiments import run_suite, available_datasets, ExpCfg
from models import MODEL_BUILDERS

LOG_PATH = Path(__file__).parent / "run.log"

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run TinyML paper experiments")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Which dataset to run: apnea_ecg, ptbxl, mitdb, or all")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated list of models (default: all registered)")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Epochs for training (overrides ExpCfg)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for loaders")
    parser.add_argument("--input_len", type=int, default=1800,
                        help="ECG window length for loaders")
    args = parser.parse_args()

    logging.info("Available datasets: %s", available_datasets())
    logging.info("Available models:   %s", list(MODEL_BUILDERS.keys()))

    datasets = available_datasets() if args.dataset == "all" else [args.dataset]
    if args.models == "all":
        models = list(MODEL_BUILDERS.keys())
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = ExpCfg(epochs=args.epochs, batch_size=args.batch_size, input_len=args.input_len)
    logging.info("Config: %s", cfg)

    run_suite(datasets=datasets, models=models, cfg=cfg)

if __name__ == "__main__":
    main()
