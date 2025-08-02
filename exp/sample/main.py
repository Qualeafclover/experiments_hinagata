#!/usr/bin/env python3

from pprint import pformat
from pathlib import Path

import logging
import datetime
import argparse

from shared.utils.config import ConfigDict
from shared.utils.logger import setup_logging


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs='+', help="Config file paths, later configs are prioritized", required=True)
    parser.add_argument("--runs-dir", type=Path, help="Directory to store run outputs", default="runs")
    parser.add_argument("--run-name", type=str, help="Name of the run", default=datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    run_dir = args.runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    setup_logging(out_file=run_dir / "run.log")
    logging.info(f"Arguments: \n{pformat(vars(args))}")
    config = ConfigDict.from_yaml(*args.configs)
    logging.info(f"Configurations: \n{pformat(config)}")


if __name__ == "__main__":
    main()
