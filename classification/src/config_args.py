"""SageMaker configuration parsing for HuggingFace BERT"""

# Python Built-Ins:
import argparse
import json
import logging
import os
from transformers import glue_processors as processors
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,
                                  AlbertConfig,
                                  AlbertForSequenceClassification, 
                                  AlbertTokenizer,
                                )


logging.basicConfig()
logger = logging.getLogger("config")

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())
print(ALL_MODELS)

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
    parser = argparse.ArgumentParser()
    
    hps = json.loads(os.environ["SM_HPS"])

    ## Required parameters
    
    # Resource Management:
    parser.add_argument("--num-gpus", type=int, default=int(os.environ["SM_NUM_GPUS"]),help="Number of GPUs to use in training.")
    parser.add_argument("--num-workers", type=int, default=int(os.environ["SM_NUM_CPUS"]),
        help='Number of data workers: set higher to accelerate data loading, if CPU and GPUs are powerful'
    )
    
    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    parser.add_argument("--model_type", default=hps.get("mmodel-type", 'bert'), type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name", default=hps.get("mmodel-name", 'bert-base-uncased'), type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=hps.get("task-name", 'cola'), type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))


    ## Other parameters
    parser.add_argument("--config_name", default=hps.get("config-name", "bert-base-uncased"), type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=hps.get("tokenizer-name", 'bert-base-uncased'), type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=hps.get("max-seq-length", 128), type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=hps.get("per_gpu_train_batch_size", 8), type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=hps.get("per_gpu_eval_batch_size", 8), type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=hps.get("learning-rate", 5e-5), type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=hps.get("weight-decay", 0.0), type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=hps.get("num_train_epochs", 3.0), type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=hps.get("logging_steps", 50),
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=hps.get("save_steps", 50),
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    return parser.parse_args()


