from os.path import isfile, join
from torch.utils.data.dataset import Dataset, TensorDataset
import os
import numpy as np
import pandas as pd
import csv


# class TensorDataset(Dataset):
#     r"""Dataset wrapping tensors.

#     Each sample will be retrieved by indexing tensors along the first dimension.

#     Arguments:
#         *tensors (Tensor): tensors that have the same size of the first dimension.
#     """

#     def __init__(self, *tensors):
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#         self.tensors = tensors

#     def __getitem__(self, index):
#         return tuple(tensor[index] for tensor in self.tensors)

#     def __len__(self):
#         return self.tensors[0].size(0)


class AwsImdbExampleDataset(TensorDataset):
    
    def __init__(self, use_subset=False):
        self._download_set_and_convert(use_subset)

    def _download_set_and_convert(self, use_subset=False):
        from torchnlp.datasets import imdb_dataset
        train, test = imdb_dataset(train=True, test=True)
        # writing the dataset as CSV
        with open('data/train.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['text', 'sentiment'])
            for i in train:
                if 'pos' in i['sentiment']:
                    csvwriter.writerow([i['text'], 1])
                else:
                    csvwriter.writerow([i['text'], 0])

        with open('data/test.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['text', 'sentiment'])
            for i in test:
                if 'pos' in i['sentiment']:
                    csvwriter.writerow([i['text'], 1])
                else:
                    csvwriter.writerow([i['text'], 0])
        # loading with pandas and migrate to numpy
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        if use_subset:
            train_df = train_df.sample(int(len(train_df)*0.1))
            test_df = test_df.sample(int(len(test_df)*0.1))
        train_sentences = train_df.text.values
        train_labels = train_df.sentiment.values
        test_sentences = test_df.text.values
        test_labels = test_df.sentiment.values
        os.makedirs("./data/train", exist_ok=True)
        np.save("./data/train/train_sentences.npy", train_sentences)
        np.save("./data/train/train_labels.npy", train_labels)
        os.makedirs("./data/test", exist_ok=True)
        np.save("./data/test/test_sentences.npy", test_sentences)
        np.save("./data/test/test_labels.npy", test_labels)

    def prepare_and_upload_to_s3(self, bucket_name, bucket_prefix, session=None):
        traindata_s3_prefix = f"{bucket_prefix}/datasets/train"
        testdata_s3_prefix = f"{bucket_prefix}/datasets/test"
        if session == None:
            # not running on SageMaker. Uploading with boto3
            import boto3
            from os import listdir
            s3 = boto3.resource('s3')
            path = './data/train/'

            files = [f for f in listdir(path) if isfile(join(path, f))]
            for f in files:
                s3.meta.client.upload_file(
                    path+f, bucket_name, bucket_prefix+'/'+f)

            path = './data/test/'

            files = [f for f in listdir(path) if isfile(join(path, f))]
            for f in files:
                s3.meta.client.upload_file(
                    path+f, bucket_name, bucket_prefix+'/'+f)
            train_s3 = 's3://{}/{}'.format(bucket_name, bucket_prefix)
            test_s3 = 's3://{}/{}'.format(bucket_name, bucket_prefix)
        else:
            train_s3 = session.upload_data(path="./data/train/", bucket=bucket_name, key_prefix=bucket_prefix)
            test_s3 = session.upload_data(path="./data/test/", bucket=bucket_name, key_prefix=bucket_prefix)
            print(train_s3)
        return (train_s3, test_s3)
        
        
        

