from torch.utils.data import DataLoader, RandomSampler
from transformers import SquadV1Processor, squad_convert_examples_to_features, BertTokenizer


def load_dataloader(filename, max_seq_length=384, doc_stride=128, max_query_length=64, threads=1, is_training=True, tokenizer=BertTokenizer, pretrained_weights='bert-base-uncased', processor=SquadV1Processor()):
    '''
    Load Dataset into DataLoader with predetermined parameters and return features and examples as well (needed for evaluation)
    
    '''
    tokenizer = tokenizer.from_pretrained(pretrained_weights)
    examples = processor.get_train_examples('.', filename)
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
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    return examples, features, dataloader
    