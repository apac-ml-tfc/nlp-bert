"""Train HuggingFace BERT on SQuAD Question Answering Dataset"""

# Python Built-Ins:
import logging
import random

# External Dependencies:
import numpy as np
import torch
import transformers as txf

# Local Dependencies:
import config


logger = logging.getLogger("train")


def set_seed(args):
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.num_gpus > 0:
            torch.cuda.manual_seed_all(args.seed)


def save_progress(model, args):
    """Save the model and associated tokenizer"""
    logger.info("Saving model checkpoint to %s", args.model_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.model_dir, "training_args.bin"))


def train(args):
    config = txf.AutoConfig.from_pretrained(args.config_name)
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         do_lower_case=args.do_lower_case,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )
    model = txf.AutoModelForQuestionAnswering.from_pretrained(
        args.config_name,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # TODO: Multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus else "cpu")

    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("TODO: Train a model!")
    return

def evaluate(args, model, tokenizer, prefix=""):
    pass


if __name__ == "__main__":
    args = config.parse_args()

    # Set up logger:
    logging.basicConfig()
    logger = logging.getLogger("train")
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)

    set_seed(args)

    # Start training:
    train(args)
