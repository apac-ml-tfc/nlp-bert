"""SageMaker configuration parsing for HuggingFace BERT"""

# Python Built-Ins:
import argparse
import json
import logging
import os


def parse_args():
    hps = json.loads(os.environ["SM_HPS"])
    parser = argparse.ArgumentParser(description="Train HuggingFace BERT for Q&A")

    ## Training process parameters:
    parser.add_argument("--batch-size", type=int, default=hps.get("batch-size", 4),
        help="Training mini-batch size"
    )

    # I/O Settings:
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )

    args = parser.parse_args()

    ## Post-argparse validations & transformations:
    # ...None yet

    return args
