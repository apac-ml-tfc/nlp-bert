"""SageMaker PyTorch framework container inference overrides for HuggingFace question-answering"""

# Python Built-Ins:
import logging
import json
import os
import zipfile

# External Dependencies:
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, pipeline


logger = logging.getLogger()


def input_fn(input_bytes, content_type):
    if content_type == 'application/json':
        logger.info(input_bytes)
        return json.loads(input_bytes)
    else:
        raise ValueError('Content type must be application/json')


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        logger.info(prediction_output)
        return prediction_output, 'application/json'
    else:
        raise ValueError('Accept header must be application/json')


def model_fn(model_dir):
    print(os.listdir(model_dir))

    serving_hack_filepath = os.path.join(model_dir, "model.pth")
    if os.path.isfile(serving_hack_filepath):
        logger.info(f"Extracting archived model (SageMaker workaround)")
        # To work around https://github.com/pytorch/serve/pull/814 our 'model.pth' is actually a zip file, so
        # we first need to extract it to load the other files into model_dir:
        try:
            with zipfile.ZipFile(os.path.join(model_dir, "model.pth"), "r") as mdlzip:
                mdlzip.extractall(model_dir)
        except BadZipFile as e:
            logger.error(
                f"Failed to load 'model.pth': which should be a zip archive containing various model files"
            )
            raise e

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    model.eval()
    config = AutoConfig.from_pretrained(model_dir)
    return model, tokenizer, config


def predict_fn(input_data, model):
    trained_model, tokenizer, config = model
    nlp = pipeline(
        task="question-answering",
        model=trained_model,
        config=config,
        tokenizer=tokenizer,
        framework="pt",
    )
    result = nlp(input_data)
    return result
