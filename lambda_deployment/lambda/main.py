"""Dummy Lambda handler that retrieves a PyTorch model.tar.gz from S3"""

# Extract compressed dependencies first:
try:
    import unzip_packages
except ImportError as err:
    print("ImportError extracting compressed dependencies", err)

# Python Built-Ins:
import json
import logging
import os
import sys
import traceback

# External Dependencies:
import boto3
import torch
#import transformers as txf

logger = logging.getLogger()
model_cache = {}

CORS_HEADERS = {
    "Access-Control-Allow-Credentials": True,
    "Access-Control-Allow-Origin": "*",
}

def get_model(s3uri):
    print(f"Fetching model from {s3uri}")
    botosess = boto3.session.Session()
    #region = botosess.region_name
    s3 = botosess.resource("s3")
    bucket, _, key = s3uri[len("s3://"):].partition("/")
    obj = s3.Object(bucket, key)
    
    zipdata = s3.Object(bucket, key).get()["Body"].read()
    return zipdata

print("Defining handler")
def handler(event, context):
    print("Executing function!")
    model = model_cache.get("model")
    if model is None:
        try:
            model_s3uri = os.environ.get("MODEL_ARTIFACT")
            model = get_model(model_s3uri)
            model_cache["model"] = model
        except Exception as err:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("Unable to fetch model")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
            return {
                "statusCode": 500,
                "headers": CORS_HEADERS,
                "body": json.dumps({ "message": "Unable to fetch model" })
            }
    else:
        print("Using cached model")
    
    response = {}
    #response["model"] = model_s3uri
    response = { "message": "Howdy!" }
    print("Returning result")
    return {
        "statusCode": 200,
        "headers": CORS_HEADERS,
        "body": json.dumps(response),
    }
