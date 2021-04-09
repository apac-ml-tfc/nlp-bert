import logging
import json
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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    return model, tokenizer, config


def predict_fn(input_data, model):
    trained_model, tokenizer, config = model
    nlp = pipeline(task="question-answering", model=trained_model, config=config, tokenizer=tokenizer, framework="pt")
    result = nlp(input_data)
    return result