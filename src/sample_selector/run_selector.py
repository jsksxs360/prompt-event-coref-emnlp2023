import os
import re
import json
import logging
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.sample_selector.args import parse_args
from src.sample_selector.data import KBPCoref, get_dataLoader
from src.sample_selector.modeling import LongformerSelector

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k in ['batch_events', 'batch_event_cluster_ids']:
            new_batch_data[k] = v
        elif k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        else:
            raise ValueError(f'Unknown batch data key: {k}')
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

        if loss:
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            lr_scheduler.step()

        total_loss += loss.item() if loss else 0.
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, model):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            _, event_1_reps, event_2_reps, masks, labels = outputs
            norms_1 = (event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
            event_1_reps = event_1_reps / norms_1
            norms_2 = (event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
            event_2_reps = event_2_reps / norms_2
            event_cos = torch.sum(event_1_reps * event_2_reps, dim=-1).cpu().numpy() 
            predictions = (event_cos >= 0.5).astype(int) # [batch, event_pair_num]
            y = labels.cpu().numpy()
            lens = np.sum(masks.cpu().numpy(), axis=-1)
            true_labels += [
                int(l) for label, seq_len in zip(y, lens) for idx, l in enumerate(label) if idx < seq_len
            ]
            true_predictions += [
                int(p) for pred, seq_len in zip(predictions, lens) for idx, p in enumerate(pred) if idx < seq_len
            ]
    return classification_report(true_labels, true_predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False)
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
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, model)
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

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        dev_p, dev_r, dev_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
        logger.info(f'Dev: P - {(100*dev_p):0.4f} R - {(100*dev_r):0.4f} F1 - {(100*dev_f1):0.4f}')
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def predict(args, doc_id:str, document:str, events:list, model, tokenizer):
    '''
    # Args:
        - events: [
            (event_id, e_char_start, e_char_end), ...
        ], document[e1_char_start:e1_char_end + 1] = trigger1
    '''
    inputs = tokenizer(
        document, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt"
    )
    filtered_events = []
    new_events = []
    for event_id, char_start, char_end in events:
        token_start = inputs.char_to_token(char_start)
        if not token_start:
            token_start = inputs.char_to_token(char_start + 1)
        token_end = inputs.char_to_token(char_end)
        if not token_start or not token_end:
            print('\n' + '='*10)
            print(token_start, token_end)
            print(doc_id, char_start, char_end)
            print('[' + document[char_start:char_end+1] + ']')
            print(document[char_start-5 if char_start>= 5 else 0:char_start] + \
                '[' + document[char_start:char_end+1] + ']' + \
                document[char_end+1:char_end+6 if char_end+6 <= len(document) else len(document)])
            continue
        filtered_events.append([token_start, token_end])
        new_events.append(event_id)
    if not new_events:
        return [], [], []
    inputs = {
        'batch_inputs': inputs, 
        'batch_events': [filtered_events]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        _, event_1_reps, event_2_reps, _, _ = outputs
        norms_1 = (event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2
    event_pair_cos = torch.squeeze(torch.sum(event_1_reps * event_2_reps, dim=-1), dim=0).cpu().numpy().tolist()
    if len(new_events) > 1:
        assert len(event_pair_cos) == len(new_events) * (len(new_events) - 1) / 2
    event_id_pairs = []
    for i in range(len(new_events) - 1):
        for j in range(i + 1, len(new_events)):
            event_id_pairs.append(f'{new_events[i]}###{new_events[j]}')
    return new_events, event_id_pairs, event_pair_cos

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    model = LongformerSelector.from_pretrained(
        args.model_checkpoint,
        config=config,
        cache_dir=args.cache_dir, 
        args=args
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = KBPCoref(args.train_file)
        dev_dataset = KBPCoref(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = KBPCoref(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        best_save_weight = 'epoch_20_dev_f1_73.3697_weights.bin'
        logger.info(f'loading weights from {best_save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, best_save_weight)))
        logger.info(f'calculating event cosine similarity of {best_save_weight}...')
        results = []
        model.eval()
        with open(args.train_file, 'rt' , encoding='utf-8') as f_in:
            for line in tqdm(f_in.readlines()):
                sample = json.loads(line.strip())
                events = [
                    [event['event_id'], event['start'], event['start'] + len(event['trigger']) - 1] 
                    for event in sample['events']
                ]
                new_events, event_id_pairs, event_pair_cos = predict(args, sample['doc_id'], sample['document'], events, model, tokenizer)
                sample['events'] = [e for e in sample['events'] if e['event_id'] in new_events]
                cluster_list = []
                for cluster in sample['clusters']:
                    events = [e_id for e_id in cluster['events'] if e_id in new_events]
                    if len(events) > 0:
                        cluster_list.append({
                            'hopper_id': cluster['hopper_id'], 
                            'events': events
                        })
                sample['clusters'] = cluster_list
                sample['event_pairs_id'] = event_id_pairs
                sample['event_pairs_cos'] = event_pair_cos
                results.append(sample)
        save_name = re.sub('\.json', '', os.path.split(args.train_file)[1]) + '_with_cos.json'
        with open(os.path.join(args.output_dir, save_name), 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result) + '\n')
