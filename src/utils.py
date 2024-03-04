from datasets import load_dataset
import torch
import sys
import argparse
import logging
import os
import json
from typing import List
import jsonlines
from collections import defaultdict



def test_metric(mono_path, cross_path):
    predict_results_mono = defaultdict(list)
    predict_results_cross = defaultdict(list)
    
    with jsonlines.open(mono_path, 'r') as mono, jsonlines.open(cross_path, 'r') as cross:
        for line in mono:
            for k, v in line.items():
                predict_results_mono[k].append(v)
        
        for line in cross:
            for k, v in line.items():
                predict_results_cross[k].append(v)
    
    return predict_results_mono, predict_results_cross
        


def postprocess_text(preds:List[str], labels:List[str], bleu=True):
    processed_preds, processed_labels = [], []

    for label in labels:
        if any(map(lambda c:'\u4e00' <= c <= '\u9fa5', label)):
            label = ' '.join(list(label.strip()))
        else:
            label = label.strip()
        
        if bleu:
            processed_labels.append([label])
        else:
            processed_labels.append(label)
            
    
    for pred in preds:
        if any(map(lambda c:'\u4e00' <= c <= '\u9fa5', pred)):
            pred = ' '.join(list(pred.strip()))
        else:
            pred = pred.strip()
        
        processed_preds.append(pred)

    return processed_preds, processed_labels



def load_json(path):
    return json.load(open(path, 'r'))

def dump_json(path, item):
    with open(path, "w+") as f:
        json.dump(item, f, indent=4, sort_keys=True)





def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    
    return shifted_input_ids

