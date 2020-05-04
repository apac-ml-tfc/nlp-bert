"""Train HuggingFace BERT on SQuAD Question Answering Dataset"""

# Python Built-Ins:
import logging
import random

# External Dependencies:
import numpy as np
import timeit
import torch
import transformers as txf
from transformers.data.metrics import squad_metrics

# Local Dependencies:
import config


logger = logging.getLogger("train")


def set_seed(args):
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.num_gpus > 0:
            torch.cuda.manual_seed_all(args.seed)


def save_progress(model, args):
    """Save the model and associated tokenizer"""
    logger.info("Saving model checkpoint to %s", args.model_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.model_dir, "training_args.bin"))


def train(args):
    # Added here for reproductibility
    set_seed(args)

    config = txf.AutoConfig.from_pretrained(args.config_name)
#     tokenizer = AutoTokenizer.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         do_lower_case=args.do_lower_case,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )
    model = txf.AutoModelForQuestionAnswering.from_pretrained(
        args.config_name,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # TODO: Multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus else "cpu")

    model.to(device)

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

    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = tuple(t.to(args.device) for t in batch)

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
            if args.version_2_with_negative:
                inputs.update({ "is_impossible": batch[7] })
            if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                )

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
            if (
                args.local_rank in [-1, 0]
                and args.logging_steps > 0
                and global_step % args.logging_steps == 0
            ):
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar(f"eval_{key}", value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            # Save model checkpoint
#             if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
#                 output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 # Take care of distributed/parallel training
#                 model_to_save = model.module if hasattr(model, "module") else model
#                 model_to_save.save_pretrained(output_dir)
#                 tokenizer.save_pretrained(output_dir)

#                 torch.save(args, os.path.join(output_dir, "training_args.bin"))
#                 logger.info("Saving model checkpoint to %s", output_dir)

#                 torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#                 torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
#                 logger.info("Saving optimizer and scheduler states to %s", output_dir)

    logger.info("TODO: Train a model!")
    return

def evaluate(args, model, tokenizer, prefix=""):
    # TODO: Data loading
#     dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    # Note that DistributedSampler samples randomly
#     eval_sampler = SequentialSampler(dataset)
#     eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))

    all_results = []
    start_time = timeit.default_timer()

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

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
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
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
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
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
            args.max_answer_length,
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
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            logger.level < logging.INFO,
            args.has_unanswerable,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_metrics.squad_evaluate(examples, predictions)
    return results


if __name__ == "__main__":
    args = config.parse_args()

    # Set up logger:
    logging.basicConfig()
    logger = logging.getLogger("train")
    try:
        # e.g. convert "20" to 20, but leave "DEBUG" alone
        args.log_level = int(args.log_level)
    except ValueError:
        pass
    logger.setLevel(args.log_level)

    set_seed(args)

    # Start training:
    train(args)
