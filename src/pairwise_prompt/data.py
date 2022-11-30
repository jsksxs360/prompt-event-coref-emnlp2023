from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import json

# special tokens
BERT_SPECIAL_TOKENS= [
    '[EVENT1_START]', '[EVENT1_END]', '[EVENT2_START]', '[EVENT2_END]', 
    '[L_TOKEN1]', '[L_TOKEN2]', '[L_TOKEN3]', '[L_TOKEN4]', '[L_TOKEN5]', '[L_TOKEN6]'
]
ROBERTA_SPECIAL_TOKENS = [
    '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>', 
    '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
]
COREF_TOKEN, NONCOREF_TOKEN = 'same', 'different'
ADD_MARK_TYPE = ['bert', 'roberta', 'longformer']

def create_new_event_sent(
    sent1_idx:int, e1_sent_start:int, e1_trigger:str, 
    sent2_idx:int, e2_sent_start:int, e2_trigger:str, 
    sents:list, sents_lens:list, 
    add_mark:str, context_k:int, max_length:int
    ):
    context_before =  ' '.join([
        sent['text'] for sent in sents[sent1_idx - context_k if sent1_idx >= context_k else 0 : sent1_idx]
    ]).strip()
    context_before_length = sum([
        sent_len for sent_len in sents_lens[sent1_idx - context_k if sent1_idx >= context_k else 0 : sent1_idx]
    ])
    context_next = ' '.join([
        sent['text'] for sent in sents[sent2_idx + 1 : sent2_idx + context_k + 1 if sent2_idx + context_k < len(sents) else len(sents)]
    ]).strip()
    context_next_length = sum([
        sent_len for sent_len in sents_lens[sent2_idx + 1 : sent2_idx + context_k + 1 if sent2_idx + context_k < len(sents) else len(sents)]
    ])
    if sent1_idx == sent2_idx: # two events in the same sentence
        sent_text = sents[sent1_idx]['text']
        assert e1_sent_start < e2_sent_start
        before, middle, next = sent_text[:e1_sent_start], sent_text[e1_sent_start + len(e1_trigger):e2_sent_start], sent_text[e2_sent_start + len(e2_trigger):]
        new_sent = context_before + (' ' if len(context_before) > 0 else '') + before + ('[EVENT1_START] ' if add_mark=='bert' else '<event1_start> ')
        e1_new_sent_start, e1_new_sent_end = len(new_sent), len(new_sent) + len(e1_trigger) - 1
        new_sent += (
            e1_trigger + (' [EVENT1_END]' if add_mark=='bert' else ' <event1_end>') + ('' if middle.startswith(' ') else ' ') + middle + \
            ('[EVENT2_START] ' if add_mark=='bert' else '<event2_start> ')
        )
        e2_new_sent_start, e2_new_sent_end = len(new_sent), len(new_sent) + len(e2_trigger) - 1
        new_sent += (e2_trigger + (' [EVENT2_END]' if add_mark=='bert' else ' <event2_end>') + ('' if next.startswith(' ') else ' ') + next + ' ' + context_next)
        assert new_sent[e1_new_sent_start:e1_new_sent_end+1] == e1_trigger
        assert new_sent[e2_new_sent_start:e2_new_sent_end+1] == e2_trigger
        return {
            'new_sent': new_sent, 
            'e1_sent_start': e1_new_sent_start, 
            'e1_sent_end': e1_new_sent_end, 
            'e1_trigger': e1_trigger, 
            'e2_sent_start': e2_new_sent_start, 
            'e2_sent_end': e2_new_sent_end, 
            'e2_trigger': e2_trigger, 
        }
    else: # events in different sentence
        # create new sentence for event 1
        before_1, next_1 = sents[sent1_idx]['text'][:e1_sent_start], sents[sent1_idx]['text'][e1_sent_start + len(e1_trigger):]
        new_sent1 = context_before + (' ' if len(context_before) > 0 else '') + before_1 + ('[EVENT1_START] ' if add_mark=='bert' else '<event1_start> ')
        e1_new_sent_start, e1_new_sent_end = len(new_sent1), len(new_sent1) + len(e1_trigger) - 1
        new_sent1 += (e1_trigger + (' [EVENT1_END]' if add_mark=='bert' else ' <event1_end>') + ('' if next_1.startswith(' ') else ' ') + next_1)
        # create new sentence for event 2
        before_2, next_2 = sents[sent2_idx]['text'][:e2_sent_start], sents[sent2_idx]['text'][e2_sent_start + len(e2_trigger):]
        new_sent2 = before_2 + ('[EVENT2_START] ' if add_mark=='bert' else '<event2_start> ')
        e2_new_sent_start, e2_new_sent_end = len(new_sent2), len(new_sent2) + len(e2_trigger) - 1
        new_sent2 += (e2_trigger + (' [EVENT2_END]' if add_mark=='bert' else ' <event2_end>') + ('' if next_2.startswith(' ') else ' ') + next_2 + ' ' + context_next)
        length = context_before_length + sents_lens[sent1_idx] + sents_lens[sent2_idx] + context_next_length + 8
        p, q = sent1_idx + 1, sent2_idx - 1
        while p <= q:
            if p == q:
                if length + sents_lens[p] <= max_length:
                    new_sent1 += (' ' + sents[p]['text'])
                final_new_sent = new_sent1 + ' ' + new_sent2
                e2_new_sent_start, e2_new_sent_end = e2_new_sent_start + len(new_sent1) + 1, e2_new_sent_end + len(new_sent1) + 1
                assert final_new_sent[e1_new_sent_start:e1_new_sent_end+1] == e1_trigger
                assert final_new_sent[e2_new_sent_start:e2_new_sent_end+1] == e2_trigger
                return {
                    'new_sent': final_new_sent, 
                    'e1_sent_start': e1_new_sent_start, 
                    'e1_sent_end': e1_new_sent_end, 
                    'e1_trigger': e1_trigger, 
                    'e2_sent_start': e2_new_sent_start, 
                    'e2_sent_end': e2_new_sent_end, 
                    'e2_trigger': e2_trigger
                }
            if length + sents_lens[p] > max_length:
                break
            else:
                length += sents_lens[p]
                new_sent1 += (' ' + sents[p]['text'])
            if length + sents_lens[q] > max_length:
                break
            else:
                length += sents_lens[q]
                new_sent2 = sents[q]['text'] + ' ' + new_sent2
                e2_new_sent_start, e2_new_sent_end = e2_new_sent_start + len(sents[q]['text']) + 1, e2_new_sent_end + len(sents[q]['text']) + 1
            p += 1
            q -= 1
        final_new_sent = new_sent1 + ' ' + new_sent2
        e2_new_sent_start, e2_new_sent_end = e2_new_sent_start + len(new_sent1) + 1, e2_new_sent_end + len(new_sent1) + 1
        assert final_new_sent[e1_new_sent_start:e1_new_sent_end+1] == e1_trigger
        assert final_new_sent[e2_new_sent_start:e2_new_sent_end+1] == e2_trigger
        return {
            'new_sent': final_new_sent, 
            'e1_sent_start': e1_new_sent_start, 
            'e1_sent_end': e1_new_sent_end, 
            'e1_trigger': e1_trigger, 
            'e2_sent_start': e2_new_sent_start, 
            'e2_sent_end': e2_new_sent_end, 
            'e2_trigger': e2_trigger
        }

class KBPCoref(Dataset):
    def __init__(self, data_file:str, add_mark:str, context_k:int, tokenizer, max_length:int):
        assert add_mark in ADD_MARK_TYPE and context_k > 0
        self.tokenizer = tokenizer
        self.data = self.load_data(data_file, add_mark, context_k, max_length)
        
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        raise ValueError(f'Unknown event_id: {event_id}')

    def load_data(self, data_file, add_mark:str, context_k:int, max_length:int):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                events = sample['events']
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1, event_2 = events[i], events[j]
                        event_1_cluster_id = self._get_event_cluster_id(event_1['event_id'], clusters)
                        event_2_cluster_id = self._get_event_cluster_id(event_2['event_id'], clusters)
                        new_event_sent = create_new_event_sent(
                            event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                            event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                            sentences, sentences_lengths, add_mark, context_k, max_length
                        )
                        Data.append({
                            'id': sample['doc_id'], 
                            'sent': new_event_sent['new_sent'], 
                            'e1_offset': event_1['start'], 
                            'e1_start': new_event_sent['e1_sent_start'], 
                            'e1_end': new_event_sent['e1_sent_end'], 
                            'e1_trigger': new_event_sent['e1_trigger'], 
                            'e2_offset': event_2['start'], 
                            'e2_start': new_event_sent['e2_sent_start'], 
                            'e2_end': new_event_sent['e2_sent_end'], 
                            'e2_trigger': new_event_sent['e2_trigger'], 
                            'label': 1 if event_1_cluster_id == event_2_cluster_id else 0
                        })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class KBPCorefTiny(Dataset):
    
    def __init__(self, data_file:str, data_file_with_cos:str, pos_top_k:int, neg_top_k:int, add_mark:str, context_k:int, tokenizer, max_length:int):
        '''
        - data_file: source train data file
        - data_file_with_cos: train data file with event similarities
        '''
        assert pos_top_k >= 0 and neg_top_k > 0
        assert add_mark in ADD_MARK_TYPE and context_k > 0
        self.tokenizer = tokenizer
        self.data = self.load_data(data_file, data_file_with_cos, pos_top_k, neg_top_k, add_mark, context_k, max_length)
    
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        raise ValueError(f'Unknown event_id: {event_id}')

    def load_data(self, data_file, data_file_with_cos, pos_top_k:int, neg_top_k:int, add_mark:str, context_k, max_length:int):

        def create_event_simi_dict(event_pairs_id, event_pairs_cos, clusters):
            simi_dict = defaultdict(list)
            for id_pair, cos in zip(event_pairs_id, event_pairs_cos):
                e1_id, e2_id = id_pair.split('###')
                coref = 1 if self._get_event_cluster_id(e1_id, clusters) == self._get_event_cluster_id(e2_id, clusters) else 0
                simi_dict[e1_id].append({'id': e2_id, 'cos': cos, 'coref': coref})
                simi_dict[e2_id].append({'id': e1_id, 'cos': cos, 'coref': coref})
            for simi_list in simi_dict.values():
                simi_list.sort(key=lambda x:x['cos'], reverse=True)
            return simi_dict
        
        def get_noncoref_ids(simi_list, top_k):
            noncoref_ids = []
            for simi in simi_list:
                if simi['coref'] == 0:
                    noncoref_ids.append(simi['id'])
                    if len(noncoref_ids) >= top_k:
                        break
            return noncoref_ids

        def get_coref_ids(simi_list, top_k):
            coref_ids = []
            for simi in simi_list[-1::-1]:
                if simi['coref'] == 1:
                    coref_ids.append(simi['id'])
                    if len(coref_ids) >= top_k:
                        break
            return coref_ids

        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f, open(data_file_with_cos, 'rt', encoding='utf-8') as f_cos:
            if pos_top_k == 0: # normal positive samples
                for line in f: # coref pairs
                    sample = json.loads(line.strip())
                    clusters = sample['clusters']
                    sentences = sample['sentences']
                    sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                    events_list, events_dict = sample['events'], {e['event_id']:e for e in sample['events']}
                    for i in range(len(events_list) - 1):
                        for j in range(i + 1, len(events_list)):
                            event_1, event_2 = events_list[i], events_list[j]
                            event_1_cluster_id = self._get_event_cluster_id(event_1['event_id'], clusters)
                            event_2_cluster_id = self._get_event_cluster_id(event_2['event_id'], clusters)
                            if event_1_cluster_id == event_2_cluster_id:
                                new_event_sent = create_new_event_sent(
                                    event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                                    event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                                    sentences, sentences_lengths, add_mark, context_k, max_length
                                )
                                Data.append({
                                    'id': sample['doc_id'], 
                                    'sent': new_event_sent['new_sent'], 
                                    'e1_offset': event_1['start'], 
                                    'e1_start': new_event_sent['e1_sent_start'], 
                                    'e1_end': new_event_sent['e1_sent_end'], 
                                    'e1_trigger': new_event_sent['e1_trigger'], 
                                    'e2_offset': event_2['start'], 
                                    'e2_start': new_event_sent['e2_sent_start'], 
                                    'e2_end': new_event_sent['e2_sent_end'], 
                                    'e2_trigger': new_event_sent['e2_trigger'], 
                                    'label': 1
                                })
            for line in f_cos:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                event_simi_dict = create_event_simi_dict(sample['event_pairs_id'], sample['event_pairs_cos'], clusters)
                events_list, events_dict = sample['events'], {e['event_id']:e for e in sample['events']}
                if pos_top_k > 0: # select hard positive samples
                    for i in range(len(events_list)):
                        event_1 = events_list[i]
                        coref_event_ids = get_coref_ids(event_simi_dict[event_1['event_id']], top_k=pos_top_k)
                        for e_id in coref_event_ids: # coref
                            event_2 = events_dict[e_id]
                            if event_1['start'] < event_2['start']:
                                new_event_sent = create_new_event_sent(
                                    event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                                    event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                                    sentences, sentences_lengths, add_mark, context_k, max_length
                                )
                                Data.append({
                                    'id': sample['doc_id'], 
                                    'sent': new_event_sent['new_sent'], 
                                    'e1_offset': event_1['start'], 
                                    'e1_start': new_event_sent['e1_sent_start'], 
                                    'e1_end': new_event_sent['e1_sent_end'], 
                                    'e1_trigger': new_event_sent['e1_trigger'], 
                                    'e2_offset': event_2['start'], 
                                    'e2_start': new_event_sent['e2_sent_start'], 
                                    'e2_end': new_event_sent['e2_sent_end'], 
                                    'e2_trigger': new_event_sent['e2_trigger'], 
                                    'label': 1
                                })
                for i in range(len(events_list)):
                    event_1 = events_list[i]
                    noncoref_event_ids = get_noncoref_ids(event_simi_dict[event_1['event_id']], top_k=neg_top_k)
                    for e_id in noncoref_event_ids: # non-coref
                        event_2 = events_dict[e_id]
                        if event_1['start'] < event_2['start']:
                            new_event_sent = create_new_event_sent(
                                event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                                event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                                sentences, sentences_lengths, add_mark, context_k, max_length
                            )
                            Data.append({
                                'id': sample['doc_id'], 
                                'sent': new_event_sent['new_sent'], 
                                'e1_offset': event_1['start'], 
                                'e1_start': new_event_sent['e1_sent_start'], 
                                'e1_end': new_event_sent['e1_sent_end'], 
                                'e1_trigger': new_event_sent['e1_trigger'], 
                                'e2_offset': event_2['start'], 
                                'e2_start': new_event_sent['e2_sent_start'], 
                                'e2_end': new_event_sent['e2_sent_end'], 
                                'e2_trigger': new_event_sent['e2_trigger'], 
                                'label': 0
                            })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, add_mark:str, collote_fn_type:str, prompt_type:str, batch_size:int=None, shuffle:bool=False):

    assert add_mark in ADD_MARK_TYPE
    assert collote_fn_type in ['normal']
    assert prompt_type in [
        'pb_d',    # prompt_base + document
        'k_pb_d'   # knowledge + prompt_base + document
    ]

    special_start_end_tokens = BERT_SPECIAL_TOKENS if add_mark == 'bert' else  ROBERTA_SPECIAL_TOKENS
    assert tokenizer.additional_special_tokens == special_start_end_tokens

    pos_id = tokenizer.convert_tokens_to_ids(COREF_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NONCOREF_TOKEN)

    def get_prompt(
        source_sent, 
        e1_trigger, e1_start, 
        e2_trigger, e2_start, 
        add_mark
        ):
        l_token1, l_token2, l_token3, l_token4, l_token5, l_token6 = (
            '[L_TOKEN1]', '[L_TOKEN2]', '[L_TOKEN3]', '[L_TOKEN4]', '[L_TOKEN5]', '[L_TOKEN6]'
        ) if add_mark=='bert' else (
            '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
        )
        mask_token = '[MASK]' if add_mark=='bert' else '<mask>'
        if prompt_type == 'pb_d': # prompt_base + document
            prompt = f'In this document, {l_token1} {e1_trigger} {l_token2} {l_token3} {e2_trigger} {l_token4} {l_token5} {mask_token} {l_token6}. '
            encoding = tokenizer(prompt, max_length=args.max_seq_length, truncation=True)
            mask_idx = encoding.char_to_token(prompt.find(mask_token))
            assert mask_idx is not None
            e1_start += len(prompt)
            e2_start += len(prompt)
            prompt += source_sent
        elif prompt_type == 'k_pb_d': # knowledge + prompt_base + document
            knowledge = 'Events that correspond to the same event usually have related trigger words (predicates) and compatible entities (participants, time, location). '
            prompt = knowledge + f'In this document, {l_token1} {e1_trigger} {l_token2} {l_token3} {e2_trigger} {l_token4} {l_token5} {mask_token} {l_token6}. '
            encoding = tokenizer(prompt, max_length=args.max_seq_length, truncation=True)
            mask_idx = encoding.char_to_token(prompt.find(mask_token))
            assert mask_idx is not None
            e1_start += len(prompt)
            e2_start += len(prompt)
            prompt += source_sent
        return {
            'prompt': prompt, 
            'mask_idx': mask_idx, 
            'e1_start': e1_start, 
            'e2_start': e2_start
        }

    def collote_fn(batch_samples):
        batch_sen, batch_mask_idx, batch_coref  = [], [], []
        for sample in batch_samples:
            prompt_data = get_prompt(
                sample['sent'], 
                sample['e1_trigger'], sample['e1_start'], 
                sample['e2_trigger'], sample['e2_start'], 
                add_mark
            )
            batch_sen.append(prompt_data['prompt'])
            batch_mask_idx.append(prompt_data['mask_idx'])
            batch_coref.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_label = [pos_id if coref == 1 else neg_id  for coref in batch_coref]
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'labels': batch_label
        }
    
    if collote_fn_type == 'normal':
        select_collote_fn = collote_fn
    
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=select_collote_fn
    )

if __name__ == '__main__':

    def print_data_statistic(data_file):
        doc_list = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                doc_list.append(json.loads(line.strip()))
        doc_num = len(doc_list)
        event_num = sum([len(doc['events']) for doc in doc_list])
        cluster_num = sum([len(doc['clusters']) for doc in doc_list])
        singleton_num = sum([1 if len(cluster['events']) == 1  else 0 
                                    for doc in doc_list for cluster in doc['clusters']])
        print(f"Doc: {doc_num} | Event: {event_num} | Cluster: {cluster_num} | Singleton: {singleton_num}")
    
    from transformers import AutoTokenizer
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 4
    args.max_seq_length = 4096
    args.model_type = 'longformer'
    args.model_checkpoint = '../../PT_MODELS/allenai/longformer-large-4096'

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    special_start_end_tokens = BERT_SPECIAL_TOKENS if args.model_type == 'bert' else  ROBERTA_SPECIAL_TOKENS
    special_tokens_dict = {'additional_special_tokens': special_start_end_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    assert tokenizer.additional_special_tokens == special_start_end_tokens

    # train_data = KBPCoref('../../data/train_filtered.json', add_mark='longformer', context_k=1, tokenizer=tokenizer, max_length=480)
    # print_data_statistic('../../data/train_filtered.json')
    # print(len(train_data))
    # labels = [train_data[s_idx]['label'] for s_idx in range(len(train_data))]
    # print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    # train_data = iter(train_data)
    # for _ in range(5):
    #     print(next(train_data))

    train_small_data = KBPCorefTiny(
        '../../data/train_filtered.json', '../../data/train_filtered_with_cos.json', 
        pos_top_k=0, neg_top_k=10, add_mark='longformer', context_k=2, tokenizer=tokenizer, max_length=482
    )
    print_data_statistic('../../data/train_filtered_with_cos.json')
    print(len(train_small_data))
    labels = [train_small_data[s_idx]['label'] for s_idx in range(len(train_small_data))]
    print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    train_data = iter(train_small_data)
    for _ in range(100):
        print('='*30)
        print(next(train_data))
    
    train_dataloader = get_dataLoader(args, train_small_data, tokenizer, add_mark='longformer', collote_fn_type='normal', prompt_type='pb_d', shuffle=True)
    batch_data = next(iter(train_dataloader))
    batch_X, batch_y = batch_data['batch_inputs'], batch_data['labels']
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', len(batch_y))
    print(batch_X)
    print(batch_y)
    print(tokenizer.decode(batch_X['input_ids'][0]))