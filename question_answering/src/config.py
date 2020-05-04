"""SageMaker configuration parsing for HuggingFace BERT"""

# Python Built-Ins:
import argparse
import json
import logging
import os

# External Dependencies:
import transformers as txf


MODEL_CONFIG_CLASSES = list(txf.MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)


def boolean_hyperparam(raw):
    """Boolean argparse type for convenience in SageMaker

    SageMaker HPO supports categorical variables, but doesn't have a specific type for booleans -
    so passing `command --flag` to our container is tricky but `command --arg true` is easy.
    Using argparse with the built-in `type=bool`, the only way to set false would be to pass an
    explicit empty string like: `command --arg ""`... which looks super weird and isn't intuitive.
    Using argparse with `type=boolean_hyperparam` instead, the CLI will support all the various
    ways to indicate 'yes' and 'no' that you might expect: e.g. `command --arg false`.
    """
    valid_false = ("0", "false", "n", "no", "")
    valid_true = ("1", "true", "y", "yes")
    raw_lower = raw.lower()
    if raw_lower in valid_false:
        return False
    elif raw_lower in valid_true:
        return True
    else:
        raise argparse.ArgumentTypeError(
        f"'{raw}' value for case-insensitive boolean hyperparam is not in valid falsy "
        f"{valid_false} or truthy {valid_true} value list"
    )


def parse_args():
    hps = json.loads(os.environ["SM_HPS"])
    parser = argparse.ArgumentParser(description="Train HuggingFace BERT for Q&A")

    ## Network parameters:
    parser.add_argument("--base-model", type=str, default=hps.get("base-model", "bert-base-uncased")
        help="Base pre-trained model to fine-tune"
    )
    parser.add_argument(
        "--config-name", type=str,
        default=hps.get("config-name", None),
        help=f"Path to pre-trained model or shortcut name selected in the list: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--model-type", type=str, default=hps.get("model-type", None),
        help=f"Model type selected in the list: {', '.join(MODEL_TYPES)}"
    )

    ## Training process parameters:
    parser.add_argument("--adam-epsilon", type=float, default=hps.get("adam-epsilon", 1e-8),
        help="Epsilon for Adam optimizer"
    )
    parser.add_argument("--batch-size", type=int, default=hps.get("batch-size", 4),
        help="Training mini-batch size"
    )
    parser.add_argument("--epochs", type=int, default=hps.get("epochs", 10),
        help="Number of epochs to train for"
    )
    parser.add_argument("--grad-acc-steps", type=int, default=hps.get("grad-acc-steps", 1),
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument("--has-unanswerable", type=boolean_hyperparam,
        default=hps.get("has-unanswerable", False),
        help="Set true for datasets (e.g. SQuAD v2) with unanswerable questions",
    )
    parser.add_argument("--lr", "--learning-rate", type=float,
        default=hps.get("lr", hps.get("learning-rate", 5e-5)),
        help="Learning rate (main training cycle)"
    )
    parser.add_argument("--max-grad-norm", type=float, default=hps.get("max-grad-norm", 1.0),
        help="Max gradient norm"
    )
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)"
    )
    parser.add_argument("--warmup-steps", type=int, default=hps.get("warmup-steps", 0),
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--wd", "--weight-decay", type=float,
        default=hps.get("wd", hps.get("weight-decay", 0.0)),
        help="Weight decay (if any)"
    )

    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 0),
        help="Number of GPUs to use in training."
    )

    # I/O Settings:
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args = parser.parse_args()

    ## Post-argparse validations & transformations:
    # ...None yet

    return args
