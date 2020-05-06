import unittest
from . import test_aws_dataset

def aws_bert_classification_test_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_aws_dataset)
    return suite
