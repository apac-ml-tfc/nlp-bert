from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import SquadV1Processor, squad_convert_examples_to_features, BertTokenizer


def load_dataloader(filename, max_seq_length=384, doc_stride=128, max_query_length=64, threads=1, batch_size=32, is_training=True, tokenizer=BertTokenizer, pretrained_weights='bert-base-uncased', processor=SquadV1Processor()):
    '''
    Load Dataset into DataLoader with predetermined parameters and return features and examples as well (needed for evaluation)
    
    filename: full file path of json file to be loaded
    max_seq_length: max len of tokens in sentence
    doc_stride: stride length of document
    max_query_length: max len of query
    threads: num of threads used
    is_training: whether examples are for training
    tokenizer: type of Tokenizer used
    pretrained_weights: name of weights used to initialise model and tokenizer
    processor: preprocessing class by huggingface to get each example
    
    :return: examples - examples parsed by processor
             features - features of examples
             dataloader - DataLoader of Dataset created
    
    '''
    tokenizer = tokenizer.from_pretrained(pretrained_weights)
    data_dir, filename = '/'.join(filename.split('/')[:-1]), filename.split('/')[-1] # split to get directory and filename separately
    examples = processor.get_train_examples(data_dir, filename)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=is_training,
        return_dataset="pt",
        threads=threads,
    )
    if is_training:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return examples, features, dataloader
    