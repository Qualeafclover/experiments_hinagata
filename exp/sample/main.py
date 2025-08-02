#!/usr/bin/env python3

import argparse
from shared.utils.config import NamedDict


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--configs", nargs='+', help="Config file paths", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args.configs)
    # config = NamedDict.from_file(args.config)
    # print(config)


if __name__ == "__main__":
    main()

def main():
    ...

if __name__ == "__main__":
    main()
