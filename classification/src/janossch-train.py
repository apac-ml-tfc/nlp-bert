# Python Built-Ins:
import logging
import os
import json
import random
import argparse
import time
from typing import Dict, Optional

# External Dependencies:
import numpy as np
import timeit
import torch
from torch.utils.data.dataset import TensorDataset
from dataclasses import dataclass, field

import transformers as txf

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# from transformers import (
#     HfArgumentParser,
#     Trainer,
#     TrainingArguments,
#     glue_compute_metrics,
#     glue_output_modes,
#     glue_tasks_num_labels,
#     set_seed,
# )

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def set_seed(args):
    if args.seed:
        logger.debug('setting seed: {}'.format(args))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.num_gpus > 0:
            torch.cuda.manual_seed_all(args.seed)

def save(model, model_dir):
    model.export(os.path.join(model_dir, 'bert'))


def prepare_data_and_tokenize(args, tokenizer):
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    logger.debug('loading files as numpy array')
    train_sentences = np.load("{}/train_sentences.npy".format(args.train))
    train_labels = np.load("{}/train_labels.npy".format(args.train))
    test_sentences = np.load("{}/test_sentences.npy".format(args.test))
    test_labels = np.load("{}/test_labels.npy".format(args.test))
    
    # restore np.load for future normal usage
    np.load = np_load_old
    
    # tokenize and convert
    train_input_ids = []
    train_attention_masks = []
    test_input_ids = []
    test_attention_masks = []
    
    logger.debug('starting with tokenizing the training sentences')
    for t in train_sentences:
        encoded_dict = tokenizer.encode_plus(
            t,
            add_special_tokens = True,
            max_length = 50,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        train_input_ids.append(encoded_dict['input_ids'])
        train_attention_masks.append(encoded_dict['attention_mask'])
    logger.debug('finished with tokenizing the training sentences')
    logger.debug('starting with tokenizing the test sentences')
    for t in test_sentences:
        encoded_dict = tokenizer.encode_plus(
            t,
            add_special_tokens = True,
            max_length = 50,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        test_input_ids.append(encoded_dict['input_ids'])
        test_attention_masks.append(encoded_dict['attention_mask'])
    logger.debug('finished with tokenizing the test sentences')
    
    train_input_ids = torch.cat(train_input_ids, dim=0)
    train_attention_mask = torch.cat(train_attention_masks, dim=0)
    train_labels = torch.tensor(train_labels)
    
    test_input_ids = torch.cat(test_input_ids, dim=0)
    test_attention_mask = torch.cat(test_attention_masks, dim=0)
    test_labels = torch.tensor(test_labels)
    
    logger.debug('Original: {}'.format(train_sentences[0]))
    logger.debug('Token IDs: {}'.format(train_input_ids[0]))
    
    train_dataset = TensorDataset(train_input_ids, train_attention_mask,  train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    return (train_dataset, test_dataset)


def train(args):
    logger.debug('training function')
    set_seed(args)
    
    num_labels = args.num_labels
    output_mode = args.output_mode
    logger.debug('num_labels: {}'.format(num_labels))
    logger.debug('output_mode: {}'.format(output_mode))
    
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.base_model_name,
                                       num_labels=num_labels,
                                       finetuning_task=args.base_job_name,
                                       )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
                                        args.base_model_name, 
                                        config=config)

#     from transformers import BertForSequenceClassification, AdamW, BertConfig

#     # Load BertForSequenceClassification, the pretrained BERT model with a single 
#     # linear classification layer on top. 
#     model = BertForSequenceClassification.from_pretrained(
#         "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
#         num_labels = 2, # The number of output labels--2 for binary classification.
#                         # You can increase this for multi-class tasks.   
#         output_attentions = False, # Whether the model returns attentions weights.
#         output_hidden_states = False, # Whether the model returns all hidden-states.
#     )

    from transformers import AdamW


    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )



    
    logger.debug('train set: {}'.format(args.train))
    logger.debug('test set: {}'.format(args.test))
    train_dataset, test_dataset = prepare_data_and_tokenize(args, tokenizer)
    
    # FIXME: this need to go into the SageMaker parts
    # If there's a GPU available...
    if torch.cuda.is_available():    
        
        # Tell pytorch to run this model on the GPU.
        model.cuda()

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
    # size of 16 or 32.
    batch_size = args.batch_size

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    
    
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
    epochs = args.epochs
    
    
    from transformers import get_linear_schedule_with_warmup
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            #logger.debug('dataset: {}'.format(train_dataset[5]))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
              
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in test_dataloader:

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(test_dataset)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model


def evaluate(args, model, tokenizer, device, prefix=""):
    return



if __name__ =='__main__':
    sm_env = json.loads(os.environ['SM_TRAINING_ENV'])

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", 0),
        help="Number of GPUs to use in training."
    )
    
    parser.add_argument('--base_job_name', type=str, default=sm_env.get('job_name'))
    
    parser.add_argument('--log_level', default=logging.INFO)

    # training args
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--output_mode', type=str, default='classification')
    parser.add_argument('--base_model_name', type=str, default='bert-base-uncased')
    
    
    args, _ = parser.parse_known_args()
    

    # ... load from args.train and args.test, train a model, write model to args.model_dir.
    
    # Set up logger:
    logging.basicConfig()
    logger = logging.getLogger("train")
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)
    

    logger.info("Starting!")

    logger.debug(args)
    
    # Start training:
    net = train(args)
    save(net, args.model_dir)
