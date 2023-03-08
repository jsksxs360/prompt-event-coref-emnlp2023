import os
import logging
import json
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.coref_prompt.arg import parse_args
from src.coref_prompt.data import KBPCoref, KBPCorefTiny, get_dataLoader
from src.coref_prompt.modeling import BertForBasePrompt, RobertaForBasePrompt, LongformerForBasePrompt
from src.coref_prompt.prompt import create_prompt
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

BASE_PROMPT_MODELS = {
    'bert': BertForBasePrompt,
    'roberta': RobertaForBasePrompt, 
    'longformer': LongformerForBasePrompt
}

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k in ['batch_event_idx', 'label_word_id']:
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

def test_loop(args, dataloader, model):
    true_labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            true_labels += batch_data['labels']
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]
            predictions += logits.argmax(dim=-1).cpu().numpy().tolist()
    return classification_report(true_labels, predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer, prompt_type, verbalizer):
    """ Train the model """
    # Set seed
    seed_everything(args.seed)
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, prompt_type, verbalizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, prompt_type, verbalizer, shuffle=False)
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

def test(args, test_dataset, model, tokenizer, save_weights:list, prompt_type, verbalizer):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, prompt_type=prompt_type, verbalizer=verbalizer, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def predict(
    args, model, tokenizer, 
    e1_global_offset:int, e1_trigger:str, e1_args:list, 
    e2_global_offset:int, e2_trigger:str, e2_args:list, 
    sentences:list, sentences_lengths:list, 
    prompt_type, verbalizer
    ):

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

    e1_sent_idx, e1_sent_start = find_event_sent(e1_global_offset, e1_trigger, sentences)
    e2_sent_idx, e2_sent_start = find_event_sent(e2_global_offset, e2_trigger, sentences)
    prompt_data = create_prompt(
        e1_sent_idx, e1_sent_start, e1_trigger, e1_args, 
        e2_sent_idx, e2_sent_start, e2_trigger, e2_args, 
        sentences, sentences_lengths, 
        prompt_type, args.model_type, tokenizer, args.max_seq_length
    )
    pos_id, neg_id = verbalizer['coref']['id'], verbalizer['non-coref']['id']
    prompt_text = prompt_data['prompt']
    # convert char offsets to token idxs
    encoding = tokenizer(prompt_text)
    mask_idx = encoding.char_to_token(prompt_data['mask_offset'])
    e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
        encoding.char_to_token(prompt_data['e1s_offset']), 
        encoding.char_to_token(prompt_data['e1e_offset']), 
        encoding.char_to_token(prompt_data['e2s_offset']), 
        encoding.char_to_token(prompt_data['e2e_offset'])
    )
    assert None not in [mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx]
    event_idx = [e1s_idx, e1e_idx, e2s_idx, e2e_idx]
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
        'batch_event_idx': [event_idx], 
        'label_word_id': [neg_id, pos_id]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
        prob = torch.nn.functional.softmax(logits, dim=-1)[0]
    return {
        "pred": logits.argmax(dim=-1)[0].item(), 
        "prob": {'non-coref': prob[0].item(), 'coref': prob[1].item()}
    }

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
    model = BASE_PROMPT_MODELS[args.model_type].from_pretrained(
        args.model_checkpoint,
        config=config, 
        args=args, 
        cache_dir=args.cache_dir
    ).to(args.device)
    base_sp_tokens = [
        '[E1_START]', '[E1_END]', '[E2_START]', '[E2_END]', '[L1]', '[L2]', '[L3]', '[L4]', '[L5]', '[L6]'
    ] if args.model_type == 'bert' else [
        '<e1_start>', '<e1_end>', '<e2_start>', '<e2_end>', '<l1>', '<l2>', '<l3>', '<l4>', '<l5>', '<l6>'
    ]
    connect_tokens = [
        '[REFER_TO]', '[NOT_REFER_TO]'
    ] if args.model_type == 'bert' else [
        '<refer_to>', '<not_refer_to>'
    ]
    sp_tokens = base_sp_tokens + connect_tokens if 'c' in args.prompt_type else base_sp_tokens
    logger.info(f"adding special mark tokens {sp_tokens} to tokenizer...")
    tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
    model.resize_token_embeddings(len(tokenizer))
    # build verbalizer
    verbalizer = {
        'coref': {
            'token': '[REFER_TO]' if args.model_type == 'bert' else '<refer_to>', 
            'id': tokenizer.convert_tokens_to_ids('[REFER_TO]' if args.model_type == 'bert' else '<refer_to>'), 
            'description': 'refer to'
        } if 'c' in args.prompt_type else {
            'token': 'yes', 'id': tokenizer.convert_tokens_to_ids('yes')
        } if 'q' in args.prompt_type else {
            'token': 'same', 'id': tokenizer.convert_tokens_to_ids('same')
        } , 
        'non-coref': {
            'token': '[NOT_REFER_TO]' if args.model_type == 'bert' else '<not_refer_to>', 
            'id': tokenizer.convert_tokens_to_ids('[NOT_REFER_TO]' if args.model_type == 'bert' else '<not_refer_to>'), 
            'description': 'not refer to'
        } if 'c' in args.prompt_type else {
            'token': 'no', 'id': tokenizer.convert_tokens_to_ids('no')
        } if 'q' in args.prompt_type else {
            'token': 'different', 'id': tokenizer.convert_tokens_to_ids('different')
        }
    }
    logger.info(f"verbalizer: {verbalizer}")
    if 'c' in args.prompt_type:
        with torch.no_grad():
            refer_tokenized = tokenizer.tokenize(verbalizer['coref']['description'])
            refer_tokenized_ids = tokenizer.convert_tokens_to_ids(refer_tokenized)
            norefer_tokenized = tokenizer.tokenize(verbalizer['non-coref']['description'])
            norefer_tokenized_ids = tokenizer.convert_tokens_to_ids(norefer_tokenized)
            if args.model_type == 'bert':
                new_embedding = model.bert.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[-2, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.bert.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[-1, :] = new_embedding.clone().detach().requires_grad_(True)
            elif args.model_type == 'roberta':
                new_embedding = model.roberta.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[-2, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.roberta.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[-1, :] = new_embedding.clone().detach().requires_grad_(True)
            else:
                new_embedding = model.longformer.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[-2, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.longformer.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[-1, :] = new_embedding.clone().detach().requires_grad_(True)
    # Training
    if args.do_train:
        if args.train_data_type == 'normal':
            train_dataset = KBPCoref(
                args.train_file, 
                args.argument_file, 
                prompt_type=args.prompt_type, 
                model_type=args.model_type, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length
            )
        else:
            train_dataset = KBPCorefTiny(
                args.train_file, 
                args.train_file_with_cos, 
                args.argument_file, 
                neg_top_k=args.neg_top_k, 
                prompt_type=args.prompt_type, 
                model_type=args.model_type, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length
            )
        labels = [train_dataset[s_idx]['label'] for s_idx in range(len(train_dataset))]
        logger.info(f"[Train] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        dev_dataset = KBPCoref(
            args.dev_file, 
            args.argument_file, 
            prompt_type=args.prompt_type, 
            model_type=args.model_type, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length
        )
        labels = [dev_dataset[s_idx]['label'] for s_idx in range(len(dev_dataset))]
        logger.info(f"[Dev] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        train(args, train_dataset, dev_dataset, model, tokenizer, prompt_type=args.prompt_type, verbalizer=verbalizer)
    # Testing
    