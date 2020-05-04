"""Train HuggingFace BERT on SQuAD Question Answering Dataset"""

# Python Built-Ins:
import logging

# External Dependencies:
import torch
import transformers

# Local Dependencies:
import config


logger = logging.getLogger("train")


def train(args):
    logger.info("TODO: Train a model!")
    pass


if __name__ == "__main__":
    args = config.parse_args()

    # Set up logger:
    logging.basicConfig()
    logger = logging.getLogger()
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)

    # Start training:
    train(args)
