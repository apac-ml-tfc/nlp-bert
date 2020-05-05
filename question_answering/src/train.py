"""Train HuggingFace BERT on SQuAD Question Answering Dataset"""

# Python Built-Ins:
import logging
import os
import random

# External Dependencies:
import numpy as np
import timeit
import torch
import transformers as txf
from transformers.data.metrics import squad_metrics
from transformers.data.processors.squad import SquadResult

# Local Dependencies:
import config
import data


logger = logging.getLogger()


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def set_seed(seed, use_gpus=True):
    """Seed all the random number generators we can think of for reproducibility"""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_gpus:
            torch.cuda.manual_seed_all(seed)


def save_progress(model, tokenizer, args, checkpoint=None, optimizer=None, scheduler=None):
    """Save the model and associated tokenizer"""
    logger.info("Saving current model to %s", args.model_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.model_dir, "training_args.bin"))

    if checkpoint is not None:
        assert optimizer and scheduler, "Must supply optimizer and scheduler args when saving checkpoints"
        ckpt_dir = os.path.join(args.checkpoint_dir, f"checkpoint-{checkpoint}")
        logger.info("Saving checkpoint %s", checkpoint)
        os.makedirs(ckpt_dir, exist_ok=True)
        model_to_save.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        logger.info("Checkpoint %s saved", checkpoint)


def train(args):
    logger.info("Creating config and model")
    config = txf.AutoConfig.from_pretrained(args.config_name)
    tokenizer = txf.AutoTokenizer.from_pretrained(
        args.config_name,
        do_lower_case=args.uncased_model,
    )
    model = txf.AutoModelForQuestionAnswering.from_pretrained(
        args.config_name,
        from_tf=bool(".ckpt" in args.config_name),
        config=config,
    )

    # TODO: Multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus else "cpu")

    logger.info("Loading model to %s", device)
    model.to(device)

    logger.info("Creating data loader")
    train_dataset = data.load_and_cache_examples(
        args.train,
        tokenizer,
        args,
        evaluate=False,
        output_examples=False,
    )
    train_dataloader = data.get_dataloader(train_dataset, args.per_gpu_train_batch_size, evaluate=False)

#     if args.max_steps > 0:
#         t_total = args.max_steps
#         args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.grad_acc_steps) + 1
#     else:
    t_total = len(train_dataloader) // args.grad_acc_steps * args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = txf.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = txf.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    ## Train!
    logger.info("Training model")
    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    for epoch in range(args.epochs):
        logger.info(f"[Epoch {epoch}] Starting")
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({ "cls_index": batch[5], "p_mask": batch[6] })
                if args.has_unanswerable:
                    inputs.update({ "is_impossible": batch[7] })
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update({
                        "langs": (
                            torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id
                        ).to(device),
                    })

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if (args.log_interval > 0 and global_step % args.log_interval == 0):
                    logstr = (
                        f"lr={scheduler.get_lr()[0]}; loss={(tr_loss - logging_loss) / args.log_interval};"
                    )

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.validation:
                        logger.info(f"[Epoch {epoch} Global Step {global_step}] Starting evaluation...")
                        results = evaluate(args, model, tokenizer, device, prefix=global_step)
                        for key, value in results.items():
                            logstr += f" eval_{key}={value:.3f};"

                    logger.info(f"[Epoch {epoch} Global Step {global_step}] Metrics: {logstr}")
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0:
                    save_progress(
                        model,
                        tokenizer,
                        args,
                        checkpoint=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    logger.info("Training complete: Saving model")
    save_progress(model, tokenizer, args)
    return

def evaluate(args, model, tokenizer, device, prefix=""):
    eval_dataset, examples, features = data.load_and_cache_examples(
        args.validation,
        tokenizer,
        args,
        evaluate=True,
        output_examples=True,
    )
    eval_dataloader = data.get_dataloader(eval_dataset, args.per_gpu_eval_batch_size, evaluate=True)

    all_results = []
    start_time = timeit.default_timer()
    eval_batches = 0

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        eval_batches += 1

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
                    )

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            # TODO: i and feature_index are the same number! Simplify by removing enumerate?
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f sec per example)",
        evalTime,
        evalTime / (eval_batches * args.per_gpu_eval_batch_size)
    )

    # Compute predictions
    output_prediction_file = os.path.join(args.output_data_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_data_dir, "nbest_predictions_{}.json".format(prefix))

    if args.has_unanswerable:
        output_null_log_odds_file = os.path.join(args.output_data_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = squad_metrics.compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_len,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.has_unanswerable,
            tokenizer,
            logger.level < logging.INFO,
        )
    else:
        predictions = squad_metrics.compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_len,
            args.uncased_model,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            logger.level < logging.INFO,
            args.has_unanswerable,
            args.null_score_diff_thresh,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_metrics.squad_evaluate(examples, predictions)
    return results


if __name__ == "__main__":
    args = config.parse_args()

    for l in (logger, data.logger):
        config.configure_logger(l, args)

    logger.info("Starting!")
    set_seed(args.seed, use_gpus=args.num_gpus > 0)

    # Start training:
    train(args)
