
import argparse
import os
import json

import numpy as np
from six import BytesIO

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, SquadV1Processor, squad_convert_examples_to_features, SquadExample


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default=os.environ['CONFIG'])
    parser.add_argument('--tokenizer-name', type=str, default='bert-base-uncased')
    parser.add_argument('--model-name', type=str, default=os.environ['MODEL'])
    parser.add_argument('--uncased-model', type=bool, default=False)
    parser.add_argument('--max-seq-length', type=int, default=384)
    parser.add_argument('--doc-stride', type=int, default=128)
    parser.add_argument('--max-query-length', type=int, default=64)
    
    args, _ = parser.parse_known_args()


def model_fn(model_dir):
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.uncased_model,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.config_name),
        config=config,
    )

    return model, tokenizer


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


def predict_fn(input_data, model):
    trained_model, tokenizer = model
    examples = _read_squad_examples(input_data)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    #TODO: extract function
    all_results = []
    eval_batches = 0
    for batch in dataloader:
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
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(device)}
                    )

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs]
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



def _read_squad_examples(input_data):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = ""
                is_impossible = False
                start_position = -1
                end_position = -1

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def output_fn(prediction_output, accept):
    if accept == 'application/json':
        return _json_dumps(prediction_output), 'application/json'
    else:
        raise ValueError('Accept header must be application/json')