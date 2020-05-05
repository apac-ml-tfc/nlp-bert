# Python Built-Ins:
import logging
import os

# External Dependencies:
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import squad_convert_examples_to_features, SquadV1Processor, SquadV2Processor


logger = logging.getLogger("data")


def get_dataloader(dataset, batch_size=32, evaluate=False):
    sampler = SequentialSampler(dataset) if evaluate else RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def load_and_cache_examples(
    data_path,
    tokenizer,
    args,
    evaluate=False,
    output_examples=False,
):
    """Kind of like the HuggingFace function, but better aligned to SageMaker workflow

    data_path : str
        Either a path to a JSON file or a folder containing exactly one JSON file - the data to load
    tokenizer : transformers.Tokenizer
        Companion tokenizer for the model
    args : argparse.Namespace
        Extra parameters controlling e.g. caching, length limits etc.
    evaluate : bool=False
        Whether the dataset is for evaluation (different featurizations than training)
    output_examples : bool=False
        Whether to return additional features (for evaluation etc.)

    Returns
    -------
    dataset : torch.Dataset
        Processed SQuAD dataset
    examples : unknown, optional
        Processed SQuAD examples IF output_examples is True, else only dataset is returned
    features : unknown, optional
        Processed SQuAD features IF output_examples is True, else only dataset is returned
    """
    if os.path.isfile(data_path):
        # A specific file
        input_dir = os.path.dirname(data_path) or "."
        filename = os.path.basename(data_path)
    elif os.path.isdir(data_path):
        # A folder
        files = list(filter(lambda n: n.lower().endswith(".json"), os.listdir(data_path)))
        assert len(files) == 1, (
            f"data_path folder must contain exactly one JSON file. Found: {files}"
        )
        input_dir = data_path
        filename = files[0]
    else:
        # Neither file nor folder??
        raise ValueError(
            f"data_path must resolve to a file or a folder containing exactly one JSON file. Got {data_path}"
        )

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.config_name.split("/"))).pop(),
            str(args.max_seq_len),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        processor = SquadV2Processor() if args.has_unanswerable else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(input_dir, filename=filename)
        else:
            examples = processor.get_train_examples(input_dir, filename=filename)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_len,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_len,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.num_workers,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({ "features": features, "dataset": dataset, "examples": examples }, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset
