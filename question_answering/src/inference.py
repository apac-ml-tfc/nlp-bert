import json
import numpy as np
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, SquadV1Processor, squad_convert_examples_to_features, SquadExample
from data import read_examples, get_dataloader


def _json_loads(data):
    stream = BytesIO(data)
    return json.loads(stream.getvalue().decode())


def _json_dumps(data):
    buffer = BytesIO()
    buffer.write(json.dumps(data).encode())
    return buffer.getvalue()


def input_fn(input_bytes, content_type):
    if content_type == 'application/json':
        return _json_loads(input_bytes)
    else:
        raise ValueError('Content type must be application/json')


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        return _json_dumps(prediction_output), 'application/json'
    else:
        raise ValueError('Accept header must be application/json')


def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    return model, tokenizer


def predict_fn(input_data, model):
    trained_model, tokenizer = model
    eval_dataset = read_examples(input_data)
    eval_dataloader = get_dataloader(eval_dataset, evaluate=True)
    
    all_results = []

    for batch in eval_dataloader:
        trained_model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            #if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            #    del inputs["token_type_ids"]
            feature_indices = batch[3]
            # XLNet and XLM use more arguments for their predictions
            #if args.model_type in ["xlnet", "xlm"]:
            #    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            #    # for lang_id-sensitive xlm models
            #    if hasattr(trained_model, "config") and hasattr(trained_model.config, "lang2id"):
            #        inputs.update(
            #            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
            #        )
            outputs = trained_model(**inputs)

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

    return all_results
