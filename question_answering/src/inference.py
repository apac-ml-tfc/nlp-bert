import json
import numpy as np
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, pipeline
from data import read_examples, get_dataloader


def _json_loads(data):
    stream = BytesIO(data)
    return json.loads(stream.getvalue().decode())


def _json_dumps(data):
    buffer = BytesIO()
    buffer.write(json.dumps(data).encode())
    return buffer.getvalue()


def input_fn(input_bytes, content_type):
    if content_type == 'application/json':
        return _json_loads(input_bytes)
    else:
        raise ValueError('Content type must be application/json')


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        return _json_dumps(prediction_output), 'application/json'
    else:
        raise ValueError('Accept header must be application/json')


def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    return model, tokenizer, config


def predict_fn(input_data, model):
    trained_model, tokenizer, config = model
    nlp = pipeline(task="question-answering", model=model, config=config, tokenizer=tokenizer, framework="pt")
    result = nlp(input_data)
    return result
    
