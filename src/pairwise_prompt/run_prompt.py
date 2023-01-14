import os
import logging
import json
from collections import defaultdict
from tqdm.auto import tqdm
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.pairwise_prompt.arg import parse_args
from src.pairwise_prompt.data import KBPCoref, KBPCorefTiny, get_dataLoader
from src.pairwise_prompt.data import BERT_SPECIAL_TOKENS, ROBERTA_SPECIAL_TOKENS
from src.pairwise_prompt.utils import create_new_event_sent, get_prompt
from src.pairwise_prompt.modeling import BertForPrompt, RobertaForPrompt, LongformerForPrompt

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

MODEL_CLASSES = {
    'bert': BertForPrompt,
    'roberta': RobertaForPrompt, 
    'longformer': LongformerForPrompt
}
PROMPT_LENGTH = {
    'hb_d': 40, 'd_hb': 40,  # hard base template
    'hq_d': 40, 'd_hq': 70,  # hard question-style template
    'sb_d': 40, 'd_sb': 70   # soft base template
}
CONTEXT_K = 2

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k == 'batch_event_idx':
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, dataset, model, neg_id, pos_id):
    results = []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            token_logits = outputs[1]
            results.extend(token_logits[:, [neg_id, pos_id]].cpu().numpy())
        true_labels = [
            int(dataset[s_idx]['label']) for s_idx in range(len(dataset))
        ]
        predictions = np.asarray(results).argmax(axis=-1).tolist()
    return classification_report(true_labels, predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer, add_mark, collote_fn_type, prompt_type, verbalizer):
    """ Train the model """
    train_dataloader = get_dataLoader(
        args, train_dataset, tokenizer, add_mark=add_mark, collote_fn_type=collote_fn_type, 
        prompt_type=prompt_type, verbalizer=verbalizer, shuffle=True
    )
    dev_dataloader = get_dataLoader(
        args, dev_dataset, tokenizer, add_mark=add_mark, collote_fn_type=collote_fn_type, 
        prompt_type=prompt_type, verbalizer=verbalizer, shuffle=False
    )
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    best_f1 = 0.
    pos_id = tokenizer.convert_tokens_to_ids(verbalizer['COREF_TOKEN'])
    neg_id = tokenizer.convert_tokens_to_ids(verbalizer['NONCOREF_TOKEN'])
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + "-" * 30)
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, dev_dataset, model, neg_id=neg_id, pos_id=pos_id)
        dev_p, dev_r, dev_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
        logger.info(f'Dev: P - {(100*dev_p):0.4f} R - {(100*dev_r):0.4f} F1 - {(100*dev_f1):0.4f}')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_f1_{(100*dev_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
        with open(os.path.join(args.output_dir, 'dev_metrics.txt'), 'at') as f:
            f.write(f'epoch_{epoch+1}\n' + json.dumps(metrics, cls=NpEncoder) + '\n\n')
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list, add_mark, collote_fn_type, prompt_type, verbalizer):
    test_dataloader = get_dataLoader(
        args, test_dataset, tokenizer, add_mark=add_mark, collote_fn_type=collote_fn_type, 
        prompt_type=prompt_type, verbalizer=verbalizer, shuffle=False
    )
    pos_id = tokenizer.convert_tokens_to_ids(verbalizer['COREF_TOKEN'])
    neg_id = tokenizer.convert_tokens_to_ids(verbalizer['NONCOREF_TOKEN'])
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, test_dataset, model, neg_id=neg_id, pos_id=pos_id)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def predict(args, model, tokenizer, 
    e1_start:int, e1_trigger:str, e2_start:int, e2_trigger:str, 
    sents:list, sents_lens:list, context_k:int, context_max_length:int, 
    add_mark, prompt_type, verbalizer):

    def find_event_sent(event_start, trigger, sent_list):
        '''find out which sentence the event come from
        '''
        for idx, sent in enumerate(sent_list):
            s_start, s_end = sent['start'], sent['start'] + len(sent['text']) - 1
            if s_start <= event_start <= s_end:
                e_s_start = event_start - s_start
                assert sent['text'][e_s_start:e_s_start+len(trigger)] == trigger
                return idx, e_s_start
        print(event_start, trigger, '\n')
        for sent in sent_list:
            print(sent['start'], sent['start'] + len(sent['text']) - 1)
        return None
    
    e1_sent_idx, e1_sent_start = find_event_sent(e1_start, e1_trigger, sents)
    e2_sent_idx, e2_sent_start = find_event_sent(e2_start, e2_trigger, sents)
    new_event_sent = create_new_event_sent(
        e1_sent_idx, e1_sent_start, e1_trigger, 
        e2_sent_idx, e2_sent_start, e2_trigger, 
        sents, sents_lens, add_mark, context_k, context_max_length, tokenizer
    )
    if add_mark == 'bert':
        special_token_dict = {
            'e1s_token': '[EVENT1_START]', 'e1e_token': '[EVENT1_END]', 
            'e2s_token': '[EVENT2_START]', 'e2e_token': '[EVENT2_END]', 
            'l_token1': '[L_TOKEN1]', 'l_token2': '[L_TOKEN2]', 'l_token3': '[L_TOKEN3]', 
            'l_token4': '[L_TOKEN4]', 'l_token5': '[L_TOKEN5]', 'l_token6': '[L_TOKEN6]', 
            'mask_token': '[MASK]'
        }
    else:
        special_token_dict = {
            'e1s_token': '<event1_start>', 'e1e_token': '<event1_end>', 
            'e2s_token': '<event2_start>', 'e2e_token': '<event2_end>', 
            'l_token1': '<l_token1>', 'l_token2': '<l_token2>', 'l_token3': '<l_token3>', 
            'l_token4': '<l_token4>', 'l_token5': '<l_token5>', 'l_token6': '<l_token6>', 
            'mask_token': '<mask>'
        }
    prompt_data = get_prompt(
        prompt_type, special_token_dict, new_event_sent['new_sent'], 
        new_event_sent['e1_trigger'], new_event_sent['e1_sent_start'], new_event_sent['e1s_sent_start'], new_event_sent['e1e_sent_start'], 
        new_event_sent['e2_trigger'], new_event_sent['e2_sent_start'], new_event_sent['e2s_sent_start'], new_event_sent['e2e_sent_start'], 
        tokenizer
    )
    prompt_text = prompt_data['prompt']
    mask_idx = prompt_data['mask_idx']
    if add_mark == 'longformer':
        event_idx = [prompt_data['e1s_idx'], prompt_data['e1e_idx'], prompt_data['e2s_idx'], prompt_data['e2e_idx']]
    inputs = tokenizer(
        prompt_text, 
        max_length=args.max_seq_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx], 
        'batch_event_idx': [event_idx]
    } if add_mark == 'longformer' else {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx]
    }
    pos_id = tokenizer.convert_tokens_to_ids(verbalizer['COREF_TOKEN'])
    neg_id = tokenizer.convert_tokens_to_ids(verbalizer['NONCOREF_TOKEN'])
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        token_logits = outputs[1][:, [neg_id, pos_id]]
    pred = token_logits.argmax(dim=-1)[0].cpu().numpy().tolist()
    prob = torch.nn.functional.softmax(token_logits, dim=-1)[0].cpu().numpy().tolist()[pred]
    return pred, prob

if __name__ == '__main__':
    args = parse_args()
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, max_length=args.max_seq_length, cache_dir=args.cache_dir)
    model = MODEL_CLASSES[args.model_type].from_pretrained(
        args.model_checkpoint,
        config=config, 
        cache_dir=args.cache_dir
    ).to(args.device)
    special_tokens = BERT_SPECIAL_TOKENS if args.model_type == 'bert' else  ROBERTA_SPECIAL_TOKENS
    logger.info(f"adding special mark tokens {special_tokens} to tokenizer...")
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    assert tokenizer.additional_special_tokens == special_tokens
    model.resize_token_embeddings(len(tokenizer))
    
    if 'q' in args.prompt_type: # question style
        verbalizer = {'COREF_TOKEN': 'yes', 'NONCOREF_TOKEN': 'no'}
    else:
        verbalizer = {'COREF_TOKEN': 'same', 'NONCOREF_TOKEN': 'different'}
    logger.info(f'verbalizer: {verbalizer} ...')
    # Training
    if args.do_train:
        if args.train_data_type == 'normal':
            train_dataset = KBPCoref(
                args.train_file, 
                add_mark=args.model_type, 
                context_k=CONTEXT_K, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
            )
        else:
            train_dataset = KBPCorefTiny(
                args.train_file, 
                args.train_file_with_cos, 
                pos_top_k=args.pos_top_k, 
                neg_top_k=args.neg_top_k, 
                add_mark=args.model_type, 
                context_k=CONTEXT_K, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
            )
        dev_dataset = KBPCoref(
            args.dev_file, 
            add_mark=args.model_type, 
            context_k=CONTEXT_K, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
        )
        train(args, train_dataset, dev_dataset, model, tokenizer, 
            add_mark=args.model_type, collote_fn_type='normal', prompt_type=args.prompt_type, verbalizer=verbalizer
        )
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = KBPCoref(
            args.test_file, 
            add_mark=args.model_type, 
            context_k=CONTEXT_K, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
        )
        logger.info(f'loading trained weights from {args.output_dir} ...')
        test(args, test_dataset, model, tokenizer, save_weights, 
            add_mark=args.model_type, collote_fn_type='normal', prompt_type=args.prompt_type, verbalizer=verbalizer
        )
    # Predicting
    if args.do_predict:
        sent_dict = defaultdict(list) # {filename: [Sentence]}
        sent_len_dict = defaultdict(list) # {filename: [sentence length]}
        with open(args.test_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                sentences = sample['sentences']
                sentences_lengths = [len(tokenizer(sent['text']).tokens()) for sent in sentences]
                sent_dict[sample['doc_id']] = sentences
                sent_len_dict[sample['doc_id']] = sentences_lengths
        
        pred_event_file = '../../data/epoch_3_test_pred_events.json'
        
        for best_save_weight in save_weights:
            logger.info(f'loading weights from {best_save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, best_save_weight)))
            logger.info(f'predicting coref labels of {best_save_weight}...')

            results = []
            model.eval()
            with open(pred_event_file, 'rt' , encoding='utf-8') as f_in:
                for line in tqdm(f_in.readlines()):
                    sample = json.loads(line.strip())
                    events_from_file = sample['pred_label']
                    sents = sent_dict[sample['doc_id']]
                    sent_lens = sent_len_dict[sample['doc_id']]
                    preds, probs = [], []
                    for i in range(len(events_from_file) - 1):
                        for j in range(i + 1, len(events_from_file)):
                            e_i, e_j = events_from_file[i], events_from_file[j]
                            pred, prob = predict(args, model, tokenizer,
                                e_i['start'], e_i['trigger'], e_j['start'], e_j['trigger'], 
                                sents, sent_lens, CONTEXT_K, args.max_seq_length - PROMPT_LENGTH[args.prompt_type], 
                                args.model_type, args.prompt_type, verbalizer
                            )
                            # print(e_i['start'], e_i['trigger'], e_j['start'], e_j['trigger'])
                            # print(pred, prob)
                            preds.append(pred)
                            probs.append(prob)
                    results.append({
                        "doc_id": sample['doc_id'], 
                        "document": sample['document'], 
                        "sentences": sents, 
                        "sentence_lens": sent_lens, 
                        "events": [
                            {
                                'start': e['start'], 
                                'end': e['start'] + len(e['trigger']) - 1, 
                                'trigger': e['trigger']
                            } for e in events_from_file
                        ], 
                        "pred_label": preds, 
                        "pred_prob": probs
                    })
            save_name = f'_{args.model_type}_{args.prompt_type}_test_pred_corefs.json'
            with open(os.path.join(args.output_dir, best_save_weight + save_name), 'wt', encoding='utf-8') as f:
                for example_result in results:
                    f.write(json.dumps(example_result) + '\n')
