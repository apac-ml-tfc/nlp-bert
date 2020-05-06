# Python Built-Ins:
import json
import logging
import os

# External Dependencies:
import boto3
import torch
import transformers as txf

logger = logging.getLogger()

def handler(event, context):
    model_s3uri = os.environ.get("MODEL_ARTIFACT")
    response = {}
    response["model"] = model_s3uri
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Credentials": True,
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(response),
    }
