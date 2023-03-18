import os
import logging
import json
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.coref_prompt.arg import parse_args
from src.coref_prompt.data import KBPCoref, KBPCorefTiny, get_dataLoader
from src.coref_prompt.data import get_pred_related_info
from src.coref_prompt.modeling import BertForBasePrompt, RobertaForBasePrompt, LongformerForBasePrompt
from src.coref_prompt.prompt import create_prompt, create_verbalizer, get_special_tokens

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
        if k in ['batch_inputs', 'batch_mask_inputs']:
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
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, prompt_type, verbalizer, with_mask=False, shuffle=False)
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
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, prompt_type=prompt_type, verbalizer=verbalizer, with_mask=False, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def predict(
    args, model, tokenizer, 
    e1_global_offset:int, e1_trigger:str, e1_related_info:dict, 
    e2_global_offset:int, e2_trigger:str, e2_related_info:dict, 
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
        e1_sent_idx, e1_sent_start, e1_trigger, e1_related_info, 
        e2_sent_idx, e2_sent_start, e2_trigger, e2_related_info, 
        sentences, sentences_lengths, 
        prompt_type, args.model_type, tokenizer, args.max_seq_length
    )
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
        'label_word_id': [verbalizer['non-coref']['id'], verbalizer['coref']['id']]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
        prob = torch.nn.functional.softmax(logits, dim=-1)[0]
    pred = logits.argmax(dim=-1)[0].item()
    prob = prob[pred].item()
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
    model = BASE_PROMPT_MODELS[args.model_type].from_pretrained(
        args.model_checkpoint,
        config=config, 
        args=args, 
        cache_dir=args.cache_dir
    ).to(args.device)
    base_sp_tokens = get_special_tokens(args.model_type, 'base')
    connect_tokens = get_special_tokens(args.model_type, 'connect')
    sp_tokens = base_sp_tokens + connect_tokens if 'c' in args.prompt_type else base_sp_tokens
    logger.info(f"adding special mark tokens {sp_tokens} to tokenizer...")
    tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
    model.resize_token_embeddings(len(tokenizer))
    # build verbalizer
    verbalizer = create_verbalizer(tokenizer, args.model_type, args.prompt_type)
    logger.info(f"verbalizer: {verbalizer}")
    if 'c' in args.prompt_type:
        logger.info(f"initialize embeddings for {verbalizer['coref']['token']} and {verbalizer['non-coref']['token']}...")
        refer_idx, norefer_idx = -2, -1
        with torch.no_grad():
            refer_tokenized = tokenizer.tokenize(verbalizer['coref']['description'])
            refer_tokenized_ids = tokenizer.convert_tokens_to_ids(refer_tokenized)
            norefer_tokenized = tokenizer.tokenize(verbalizer['non-coref']['description'])
            norefer_tokenized_ids = tokenizer.convert_tokens_to_ids(norefer_tokenized)
            if args.model_type == 'bert':
                new_embedding = model.bert.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.bert.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            elif args.model_type == 'roberta':
                new_embedding = model.roberta.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.roberta.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            else: # longformer
                new_embedding = model.longformer.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.longformer.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
    # Training
    if args.do_train:
        if args.train_data_type == 'normal':
            train_dataset = KBPCoref(
                args.train_file, 
                args.train_simi_file, 
                prompt_type=args.prompt_type, 
                model_type=args.model_type, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length
            )
        else:
            train_dataset = KBPCorefTiny(
                args.train_file, 
                args.train_file_with_cos, 
                args.train_simi_file, 
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
            args.dev_simi_file, 
            prompt_type=args.prompt_type, 
            model_type=args.model_type, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length
        )
        labels = [dev_dataset[s_idx]['label'] for s_idx in range(len(dev_dataset))]
        logger.info(f"[Dev] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        train(args, train_dataset, dev_dataset, model, tokenizer, prompt_type=args.prompt_type, verbalizer=verbalizer)
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    # Testing
    if args.do_test:
        test_dataset = KBPCoref(
            args.test_file, 
            args.test_simi_file, 
            prompt_type=args.prompt_type, 
            model_type=args.model_type, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length
        )
        labels = [test_dataset[s_idx]['label'] for s_idx in range(len(test_dataset))]
        logger.info(f"[Test] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        logger.info(f'loading trained weights from {args.output_dir} ...')
        test(args, test_dataset, model, tokenizer, save_weights, prompt_type=args.prompt_type, verbalizer=verbalizer)
    # Predicting
    if args.do_predict:
        sentence_dict = defaultdict(list) # {filename: [Sentence]}
        sentence_len_dict = defaultdict(list) # {filename: [sentence length]}
        related_dict = get_pred_related_info(args.pred_test_simi_file)
        with open(args.test_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                sentences = sample['sentences']
                sentences_lengths = [len(tokenizer.tokenize(sent['text'])) for sent in sentences]
                sentence_dict[sample['doc_id']] = sentences
                sentence_len_dict[sample['doc_id']] = sentences_lengths
        
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
                    sentences = sentence_dict[sample['doc_id']]
                    sentence_lengths = sentence_len_dict[sample['doc_id']]
                    doc_related_info = related_dict[sample['doc_id']]
                    preds, probs = [], []
                    for i in range(len(events_from_file) - 1):
                        for j in range(i + 1, len(events_from_file)):
                            e_i, e_j = events_from_file[i], events_from_file[j]
                            pred, prob = predict(
                                args, model, tokenizer,
                                e_i['start'], e_i['trigger'], doc_related_info[e_i['start']], 
                                e_j['start'], e_j['trigger'], doc_related_info[e_j['start']], 
                                sentences, sentence_lengths, 
                                args.prompt_type, verbalizer
                            )
                            preds.append(pred)
                            probs.append(prob)
                    results.append({
                        "doc_id": sample['doc_id'], 
                        "document": sample['document'], 
                        "sentences": sentences, 
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
