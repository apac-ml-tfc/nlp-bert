from aws_bert_classification.dataset import AwsImdbExampleDataset
import unittest
import os
import sys
import sagemaker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_aws_imdb_example_dataset_without_sagemaker():
    print("running AwsImdbExampleDataset tests")
    BUCKET_NAME = '2020-05-gym-bert'
    PREFIX = 'bert-classification-janossch'
    
    dataset = AwsImdbExampleDataset()
    train, test = dataset.prepare_and_upload_to_s3(bucket_name=BUCKET_NAME, bucket_prefix=PREFIX)
    assert 's3://{}/{}'.format(BUCKET_NAME, PREFIX) in train
    assert 's3://{}/{}'.format(BUCKET_NAME, PREFIX) in test
    
def test_aws_imdb_example_dataset_with_sagemaker():
    print("running AwsImdbExampleDataset tests")
    sess = sagemaker.Session()
    
    BUCKET_NAME = sess.default_bucket()
    PREFIX = 'bert-classification-janossch'
    
    dataset = AwsImdbExampleDataset()
    train, test = dataset.prepare_and_upload_to_s3(bucket_name=BUCKET_NAME, bucket_prefix=PREFIX, session=sess)
    assert 's3://{}/{}'.format(BUCKET_NAME, PREFIX) in train
    assert 's3://{}/{}'.format(BUCKET_NAME, PREFIX) in test
