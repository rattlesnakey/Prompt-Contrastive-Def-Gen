import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load
from nltk import ngrams
from rouge import Rouge
import os
import json
import jsonlines
from bert_score import BERTScorer
from nltk.translate import nist_score, bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.nn import CrossEntropyLoss
from evaluate import logging
import numpy as np
from collections import defaultdict
from utils import interact_generate, batch_generate, postprocess_text, dump_json, load_json, test_metric
from typing import Dict, List
from soft_prompt import set_soft_embedding_and_freeze
import deepspeed
import shutil



def interact_generate_predictions(args):
    
    """
        Return: 
            prediction_result_mono:Dict[List], src_lang and tgt_lang are the same
            prediction_result_cross:Dict[List], src_lang and tgt_lang are not he same
            

    """
    
    predict_results_mono = defaultdict(list)
    predict_results_cross = defaultdict(list)
    
    
    def add_sample(predict_results, tokenizer):
        predict_results['word'].append(word)
        predict_results['exp'].append(exp)
        predict_results['definition'].append(definition)
        predict_results['prediction'].append(cur_prediction)
        predict_results['src_lang'].append(tokenizer.src_lang)
        predict_results['tgt_lang'].append(tokenizer.tgt_lang)
        
    if args.model_type == 'M2M':
        tgt_langs = ['en', 'zh', 'ja', 'ar', 'fr', 'ko']
    elif args.model_type == 'MBART':
        tgt_langs = ['en_XX','zh_CN','ja_XX','ar_AR','fr_XX', 'ko_KR']
        
    with jsonlines.open(args.test_dataset_path, 'r') as f:
        for cur_json in tqdm(f, desc='Generating ....'):
            word, exp, definition = cur_json['word'], cur_json['exp'], cur_json['definition']
            
            if any(map(lambda c:'\u4e00' <= c <= '\u9fa5', word)):
                tokenizer.src_lang = 'zh'
            else:
                tokenizer.src_lang = 'en'
            cur_input = word.strip() + tokenizer.sep_token + exp.strip()
            
            for tgt_lang in tgt_langs:
                tokenizer.tgt_lang = tgt_lang
                cur_prediction = interact_generate(model, cur_input, tokenizer, device, args.n_prompt_tokens, num_beams=args.beams)
                
                if tokenizer.src_lang == tokenizer.tgt_lang:
                    add_sample(predict_results_mono, tokenizer)
                else:
                    add_sample(predict_results_cross, tokenizer)
    
    return predict_results_mono, predict_results_cross
            
            

def batch_generate_predictions(args):
    
    """
        Return: 
            prediction_result_mono:Dict[List], src_lang and tgt_lang are the same
            prediction_result_cross:Dict[List], src_lang and tgt_lang are not he same
            

    """
    
    predict_results_mono = defaultdict(list)
    predict_results_cross = defaultdict(list)
    
    def add_prediction(predict_results):
        predict_results['prediction'].append(cur_prediction)
    
    def add_sample(predict_results):
        predict_results['word'].append(word)
        predict_results['exp'].append(exp)
        predict_results['definition'].append(definition)
        predict_results['src_lang'].append(src_lang)
        predict_results['tgt_lang'].append(tgt_lang)
        
    if args.model_type == 'M2M':
        tgt_langs = ['en', 'zh', 'ar']
    elif args.model_type == 'MBART':
        tgt_langs = ['en_XX','zh_CN','ar_AR']
        
    batch_src_langs = []; batch_tgt_langs = []
    batch_inputs = []; batch_targets = []
    cur_count = 0

    with jsonlines.open(args.test_dataset_path, 'r') as f:
        for idx, cur_json in enumerate(tqdm(f, desc='Generating ....')):
            word, exp, definition = cur_json['word'], cur_json['exp'], cur_json['definition']
            
            #! Chinese word
            if any(map(lambda c:'\u4e00' <= c <= '\u9fa5', word)):
                if args.model_type == 'M2M':
                    src_lang = 'zh'
                elif args.model_type == 'MBART':
                    src_lang = 'zh_CN'
            else:
                if args.model_type == 'M2M':
                    src_lang = 'en'
                elif args.model_type == 'MBART':
                    src_lang = 'en_XX'
                
                
                
            if args.model_type == 'M2M':
                cur_input = "__" + src_lang + "__" + word.strip() + tokenizer.sep_token + exp.strip() + tokenizer.eos_token
            elif args.model_type == 'MBART':
                cur_input = src_lang + word.strip() + tokenizer.sep_token + exp.strip() + tokenizer.eos_token
            
            for tgt_lang in tgt_langs:
    
                if src_lang == tgt_lang:
                    add_sample(predict_results_mono)
                else:
                    add_sample(predict_results_cross)
                    
                batch_inputs.append(cur_input)
                if args.model_type == 'M2M':
                    cur_decoder_input = "__" + tgt_lang + "__" + tokenizer.eos_token
                elif args.model_type == 'MBART':
                    cur_decoder_input = tgt_lang + tokenizer.eos_token
                    
                batch_targets.append(cur_decoder_input)
                batch_src_langs.append(src_lang)
                batch_tgt_langs.append(tgt_lang)
                cur_count += 1
                
                if cur_count % args.eval_batch_size == 0 or len(predict_results_mono['word'] + predict_results_cross['word']) == (idx + 1) * len(tgt_langs):
                    batch_predictions = batch_generate(model, batch_inputs, batch_targets, tokenizer, device, args.n_prompt_tokens, num_beams=args.beams)        
    
                    for src_lang, tgt_lang, cur_prediction in zip(batch_src_langs, batch_tgt_langs, batch_predictions):
                        if src_lang == tgt_lang:
                            add_prediction(predict_results_mono,)
                        else:
                            add_prediction(predict_results_cross,)
                    
                    batch_inputs = []; batch_targets = []; batch_src_langs = []; batch_tgt_langs = []
                    cur_count = 0
    
    return predict_results_mono, predict_results_cross




def get_bleu(predictions, references):
    preds, refs = postprocess_text(predictions, references, bleu=True)
    bleus = []
    try:
        for pred, ref in zip(preds, refs):
            bleu = sentence_bleu(ref, pred, smoothing_function=SmoothingFunction().method2, auto_reweigh=True)
            bleus.append(bleu)
        result = {'Average_BLEU':sum(bleus)/len(bleus)}
    except ZeroDivisionError:
        result = {'Average_BLEU':0}
    return result 


def get_rouge(predictions, references):
    
    preds, refs = postprocess_text(predictions, references, bleu=False)
    try:
        result = Rouge().get_scores(refs=preds, hyps=refs, avg=True)
    except ValueError: 
        result = {'Rouge':0}

    return result


def get_BERTScore(model_name, predictions, references):
    scorer = BERTScorer(model_type=model_name, lang='cs', rescale_with_baseline=True)
    P, R, F1 = scorer.score(predictions, references)
    results = {'BERTScore-Precision':P.mean().item(), 'BERTScore-Recall':R.mean().item(), 'BERTScore-F1':F1.mean().item()}
    return results



def get_distinct(predictions, references):

    preds, _ = postprocess_text(predictions, references, bleu=False)
    result = {}
    try:
        for i in range(1, 5):
            all_ngram, all_ngram_num = {}, 0.
            for k, pred in enumerate(preds):
                ngs = ["_".join(c) for c in ngrams(pred.strip().split(), i)]
                all_ngram_num += len(ngs)
                for s in ngs:
                    if s in all_ngram:
                        all_ngram[s] += 1
                    else:
                        all_ngram[s] = 1
            result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)
    except ZeroDivisionError:
        result = {'distinct': 0}
    return result


def get_nist(preds, refs):
    n = 5
    preds, refs = postprocess_text(preds, refs, bleu=True)
    nists = []
    try:
        for pred, ref in zip(preds, refs):
            pred = pred.split()
            ref[0] = ref[0].split()
            if len(pred) < n:
                n = len(pred)
            nist = nist_score.sentence_nist(ref, pred, n)
            nists.append(nist)
        result = {'Average_NIST':sum(nists)/len(nists)}
    except ZeroDivisionError:
        result = {'Average_NIST':0}
    return result 



def evaluate(preds, refs, mono=True):
    if mono:
        bert_score = get_BERTScore(args.model_name, preds, refs)
        bleu = get_bleu(preds, refs)
        rouge = get_rouge(preds, refs)
        distinct = get_distinct(preds, refs)
        nist = get_nist(preds, refs)
        return [bleu, rouge, distinct, bert_score, nist]
    else:
        bert_score = get_BERTScore(args.model_name, preds, refs)
        # distinct = get_distinct(preds, refs)
        return [bert_score]
        

def reorganize(predict_results):
    keys = predict_results.keys()
    
    for idx, (src_lang, tgt_lang) in enumerate(zip(predict_results['src_lang'], predict_results['tgt_lang'])):
        cur_key = '-'.join([src_lang, tgt_lang])
        
        for k in keys:
            organized_predictions[cur_key][k].append(predict_results[k][idx])
        
    

def save_predictions(args, organized_predictions):
    
    for lang_pair, predict_results in organized_predictions.items():
        cur_output_path = os.path.join(args.model_path, 'predictions', f'{args.output_prefix}_{lang_pair}_prediction.json')
        cur_dir = os.path.dirname(cur_output_path)
        
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir, exist_ok=True)
            
        with jsonlines.open(cur_output_path, 'w') as o:
            words = predict_results['word']; exps = predict_results['exp']
            definitions = predict_results['definition']; predictions = predict_results['prediction']
            src_langs = predict_results['src_lang']; tgt_langs = predict_results['tgt_lang']
            
            for word, exp, definition, prediction, src_lang, tgt_lang in zip(words, exps, definitions, predictions, src_langs, tgt_langs):
                o.write({
                    'word':word, 
                    'exp':exp, 
                    'definition':definition, 
                    'prediction':prediction, 
                    'src_lang':src_lang,
                    'tgt_lang':tgt_lang,
                })
        

def save_metrics(args, organized_metrics):
    for lang_pair, metrics in organized_metrics.items():
        cur_output_path = os.path.join(args.model_path, 'metrics', f'{args.output_prefix}_{lang_pair}_metrics.json')
        cur_dir = os.path.dirname(cur_output_path)
        
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir, exist_ok=True)
        
        dump_json(cur_output_path, metrics)
        

def del_rest_ckpts(best_model_path:str):
    cur_dir, best_ckpt = os.path.split(best_model_path)
    for file in os.listdir(cur_dir):
        if file.startswith('checkpoint-') and file != best_ckpt:
            deleted_path = os.path.join(cur_dir, file)
            print(f'deleting ckpt : {deleted_path}')
            shutil.rmtree(deleted_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='M2M', type=str, required=True, help='M2M or MBART')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_batch_size", default=1, type=int, required=True)
    parser.add_argument("--n_prompt_tokens", default=100, type=int, required=True)
    parser.add_argument('--beams', default=5, type=int, required=False)
    parser.add_argument('--max_output_length', default=150, type=int, required=False)
    parser.add_argument('--repetition_penalty', default=1.4, type=float, required=False)
    parser.add_argument('--log_path', default='log/evaluating.log', type=str, required=False, help='logging path')
    parser.add_argument('--model_path', type=str, required=True, help='best model path')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='test dataset path')
    parser.add_argument('--output_prefix', type=str, default='test', required=False, help='output_prefix_dir')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available()  else 'cpu'

    trainer_state = load_json(os.path.join(args.model_path, 'trainer_state.json'))
    best_model_path = trainer_state['best_model_checkpoint']
    
    del_rest_ckpts(best_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    
    if args.n_prompt_tokens:
        config = AutoConfig.from_pretrained(best_model_path)
        model = AutoModelForSeq2SeqLM.from_config(config=config)
        n_prompt_tokens = args.n_prompt_tokens
        set_soft_embedding_and_freeze(model, args.n_prompt_tokens)
 
        model.load_state_dict(torch.load(os.path.join(best_model_path, 'pytorch_model.bin')), strict=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
    model.to(device)
    model.eval()
    
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float, replace_method='auto')
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0: 
        predict_results_mono, predict_results_cross = batch_generate_predictions(args)
        
    organized_predictions = defaultdict(lambda:defaultdict(list))
    
    reorganize(predict_results_mono)
    reorganize(predict_results_cross)
    
    save_predictions(args, organized_predictions)
    
    organized_metrics = defaultdict(list)
    
    for lang_pair, predictions in organized_predictions.items():
        src_lang, tgt_lang = lang_pair.split('-')
        if src_lang == tgt_lang:
            cur_metrics = evaluate(predictions['prediction'], predictions['definition'], mono=True)
            organized_metrics[lang_pair] = cur_metrics
        else:
            cur_metrics = evaluate(predictions['prediction'], predictions['definition'], mono=False)
            organized_metrics[lang_pair] = cur_metrics
            
    
    save_metrics(args, organized_metrics)
    



