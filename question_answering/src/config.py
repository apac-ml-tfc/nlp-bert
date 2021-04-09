"""SageMaker configuration parsing for HuggingFace BERT"""

# Python Built-Ins:
import argparse
import json
import logging
import os
import sys

# External Dependencies:
import torch
import transformers as txf


MODEL_CONFIG_CLASSES = list(txf.MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)


def configure_logger(logger, args):
    """Configure a logger's level and handler (since base container already configures top level logging)"""
    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    logger.addHandler(consolehandler)
    logger.setLevel(args.log_level)


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
    parser.add_argument(
        "--config-name", type=str,
        default=hps.get("config-name", "bert-base-uncased"),
        #help=f"Path to pre-trained model or shortcut name selected in the list: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--model-type", type=str, default=hps.get("model-type", "bert"),
        help=f"Model type selected in the list: {', '.join(MODEL_TYPES)}"
    )
    parser.add_argument("--uncased-model", type=boolean_hyperparam, default=hps.get("uncased-model"),
        help="Set this flag if you are using an uncased model (Auto-set from config-name by default)."
    )

    ## Data processing parameters:
    parser.add_argument("--doc-stride", type=int, default=hps.get("doc-stride", 128),
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument("--max-answer-len", type=int, default=hps.get("max-answer-len", 30),
        help="The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
    )
    parser.add_argument("--max-query-len", type=int, default=hps.get("max-query-len", 64),
        help="The maximum number of tokens for the question. Questions longer than this will be truncated "
            "to this length.",
    )
    parser.add_argument("--max-seq-len", type=int, default=hps.get("max-seq-length", 384),
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument("--n-best-size", type=int, default=hps.get("n-best-size", 20),
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument("--null-score-diff-thresh", type=float,
        default=hps.get("null-score-diff-thresh", 0.0),
        help="If null_score - best_non_null is greater than the threshold predict null."
    )

    ## Training process parameters:
    parser.add_argument("--adam-epsilon", type=float, default=hps.get("adam-epsilon", 1e-8),
        help="Epsilon for Adam optimizer"
    )
    parser.add_argument("--per-gpu-eval-batch-size", type=int, default=hps.get("per-gpu-eval-batch-size", 8),
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--per-gpu-train-batch-size", type=int,
        default=hps.get("per-gpu-train-batch-size", 8),
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument("--epochs", type=int, default=hps.get("epochs", 3),
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
    parser.add_argument("--max-steps", type=int, default=hps.get("max-steps", -1),
        help="If > 0: set maximum total number of training steps to perform - overriding epochs.",
    )
    parser.add_argument("--seed", "--random-seed", type=int,
        default=hps.get("seed", hps.get("random-seed", None)),
        help="Random seed fixed for reproducibility (off by default)"
    )
    parser.add_argument("--warmup-steps", type=int, default=hps.get("warmup-steps", 0),
        help="Linear warmup over warmup steps."
    )
    parser.add_argument("--wd", "--weight-decay", type=float,
        default=hps.get("wd", hps.get("weight-decay", 0.0)),
        help="Weight decay (if any)"
    )

    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 0),
        help="Number of GPUs to use in training."
    )
    parser.add_argument("--num-workers", "-j", type=int,
        default=hps.get("num-workers", max(0, int(os.environ.get("SM_NUM_CPUS", 0)) - 2)),
        help="Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful"
    )

    # I/O Settings:
    parser.add_argument("--checkpoint-dir", type=str,
        default=hps.get("checkpoint-dir", "/opt/ml/checkpoints")
    )
    parser.add_argument("--checkpoint-interval", type=int,
        default=hps.get("checkpoint-interval", 0),
        help="Steps between saving checkpoints (set 0 to disable)"
    )
    parser.add_argument("--log-interval", type=int, default=hps.get("log-interval", 100),
        help="Logging mini-batch interval. Default is 100."
    )
    parser.add_argument("--log-level", default=hps.get("log-level", logging.INFO),
        help="Log level (per Python specs, string or int)."
    )
    parser.add_argument("--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--output-data-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    parser.add_argument("--overwrite-cache", type=boolean_hyperparam,
        default=hps.get("overwrite-cache", False),
        help="Overwrite and ignore any cached files (data preprocessing, etc)"
    )
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args()

    ## Post-argparse validations & transformations:

    # Set up log level: (Convert e.g. "20" to 20 but leave "DEBUG" alone)
    try:
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    # Note basicConfig has already been called by our parent container, so calling it won't do anything.
    logger = logging.getLogger("config")
    configure_logger(logger, args)

    if args.uncased_model is None:
        if "uncased" in args.config_name:
            logger.warning(f"Assuming uncased model from name '{args.config_name}'")
            args.uncased_model = True
        elif "cased" in args.config_name:
            args.uncased_model = False
            logger.warning(f"Assuming cased model from name '{args.config_name}'")
        else:
            parser.error(
                f"--uncased-model not specified and could not infer from --config-name '{args.config_name}'"
            )

    if args.num_gpus and not torch.cuda.is_available():
        parser.error(
            f"Got --num-gpus {args.num_gpus} but torch says cuda is not available: Cannot use GPUs"
        )

    return args
