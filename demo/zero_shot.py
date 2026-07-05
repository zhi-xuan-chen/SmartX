"""Reviewer-facing zero shot demo entry point.

Implementation pending: this script is a stable interface for the evaluation
code that will be added before public release.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "zero_shot demo is scaffolded; add implementation from the evaluation repository. "
        f"config={args.config}, checkpoint={args.checkpoint}"
    )


if __name__ == "__main__":
    main()
