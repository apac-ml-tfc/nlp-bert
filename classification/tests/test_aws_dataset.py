from aws_bert_classification.dataset import AwsImdbExampleDataset
import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_aws_imdb_example_dataset():
    print("running AwsImdbExampleDataset tests")
    dataset = AwsImdbExampleDataset()
    dataset.prepare_and_upload_to_s3(bucket_name=, bucket_prefix=)
    result = True
    assert result == True
