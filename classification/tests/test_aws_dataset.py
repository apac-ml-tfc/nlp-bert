from aws_bert_classification.dataset import AwsImdbExampleDataset
import unittest
import os
import sys
import sagemaker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_aws_imdb_example_dataset():
    print("running AwsImdbExampleDataset tests")
    sess = sagemaker.Session()
    
    BUCKET_NAME = sess.default_bucket()
    PREFIX = 'bert-classification-janossch'
    
    dataset = AwsImdbExampleDataset()
    dataset.prepare_and_upload_to_s3(bucket_name=BUCKET_NAME, bucket_prefix=PREFIX, session=sess)
    result = True
    assert result == True
