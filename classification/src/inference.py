from __future__ import absolute_import

import os
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
import subprocess
import sys

import json
import torch
import logging


from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TextClassificationPipeline


logger = logging.getLogger(__name__)



def model_fn(model_dir):
    logger.info('***** MODEL_FN ********')
    logger.info("******* Model dir contains: ")
    logger.info("Files: %s", os.listdir(model_dir))
    logger.info("Files: %s", os.listdir(os.path.join(model_dir,'code/')))
    logger.info("Files: %s", os.listdir(os.path.join(model_dir,'bert_models/distilbert/')))
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir,'bert_models/distilbert/'))
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_dir,'bert_models/distilbert/'))
    #config = AutoConfig.from_pretrained(os.path.join(model_dir,'bert_models/distilbert/'))

    return model, tokenizer


def input_fn(input_bytes, content_type):
    logger.info('***** INPUT_FN ********')
    if content_type == 'application/json':
        return json.loads(input_bytes)['text']
    else:
        raise ValueError('Content type must be application/json')

        
def predict_fn(input_data, model):
    logger.info('***** PREDICT_FN ********')
    trained_model, tokenizer = model
    pipe=TextClassificationPipeline(model=trained_model, tokenizer=tokenizer)
    logger.info('***** TEXT INPUT : %s',input_data)
    output = pipe(input_data)
    return output

def output_fn(prediction_output, accept):
    logger.info('***** OUTPUT_FN ********')
    if accept == 'application/json':
        logger.info('***** Prediction output: %s',prediction_output)
        return json.dumps(str(prediction_output)), 'application/json'
    else:
        raise ValueError('Accept header must be application/json')




    
    
