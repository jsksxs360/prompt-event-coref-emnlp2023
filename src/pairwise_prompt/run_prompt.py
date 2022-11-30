import os
import logging
import json
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
from src.pairwise_prompt.data import COREF_TOKEN, NONCOREF_TOKEN
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
    'pb_d': 30,    # prompt_base + document
    'k_pb_d': 60   # knowledge + prompt_base + document
}

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
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

def test_loop(args, dataloader, dataset, model, tokenizer):
    results = []
    pos_id = tokenizer.convert_tokens_to_ids(COREF_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NONCOREF_TOKEN)
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
        results = np.asarray(results)
        predictions = np.asarray(results).argmax(axis=-1).tolist()
    return classification_report(true_labels, predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer, add_mark, collote_fn_type, prompt_type):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, add_mark=add_mark, collote_fn_type=collote_fn_type, prompt_type=prompt_type, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, add_mark=add_mark, collote_fn_type=collote_fn_type, prompt_type=prompt_type, shuffle=False)
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
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + "-" * 30)
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, dev_dataset, model, tokenizer)
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

def test(args, test_dataset, model, tokenizer, save_weights:list, add_mark, collote_fn_type, prompt_type):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, add_mark=add_mark, collote_fn_type=collote_fn_type, prompt_type=prompt_type, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, test_dataset, model, tokenizer)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

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
    special_start_end_tokens = BERT_SPECIAL_TOKENS if args.model_type == 'bert' else  ROBERTA_SPECIAL_TOKENS
    logger.info(f"adding special mark tokens {special_start_end_tokens} to tokenizer...")
    special_tokens_dict = {'additional_special_tokens': special_start_end_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    assert tokenizer.additional_special_tokens == special_start_end_tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Training
    if args.do_train:
        if args.train_data_type == 'normal':
            train_dataset = KBPCoref(
                args.train_file, 
                add_mark=args.model_type, 
                context_k=5 if args.model_type == 'longformer' else 1, 
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
                context_k=5 if args.model_type == 'longformer' else 1, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
            )
        dev_dataset = KBPCoref(
            args.dev_file, 
            add_mark=args.model_type, 
            context_k=5 if args.model_type == 'longformer' else 1, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
        )
        train(args, train_dataset, dev_dataset, model, tokenizer, 
            add_mark=args.model_type, collote_fn_type='normal', prompt_type=args.prompt_type
        )
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = KBPCoref(
            args.test_file, 
            add_mark=args.model_type, 
            context_k=5 if args.model_type == 'longformer' else 1, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length - PROMPT_LENGTH[args.prompt_type]
        )
        logger.info(f'loading trained weights from {args.output_dir} ...')
        test(args, test_dataset, model, tokenizer, save_weights, 
            add_mark=args.model_type, collote_fn_type='normal', prompt_type=args.prompt_type
        )