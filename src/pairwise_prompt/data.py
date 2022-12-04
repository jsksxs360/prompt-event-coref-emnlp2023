from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm.auto import tqdm

SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
id2subtype = {idx: c for idx, c in enumerate(SUBTYPES, start=1)}
subtype2id = {v: k for k, v in id2subtype.items()}

# special tokens
BERT_SPECIAL_TOKENS= [
    '[EVENT1_START]', '[EVENT1_END]', '[EVENT2_START]', '[EVENT2_END]', 
    '[L_TOKEN1]', '[L_TOKEN2]', '[L_TOKEN3]', '[L_TOKEN4]', '[L_TOKEN5]', '[L_TOKEN6]'
]
ROBERTA_SPECIAL_TOKENS = [
    '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>', 
    '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
]
ADD_MARK_TYPE = ['bert', 'roberta', 'longformer']

def create_new_event_sent(
    sent1_idx:int, e1_sent_start:int, e1_trigger:str, 
    sent2_idx:int, e2_sent_start:int, e2_trigger:str, 
    sents:list, sents_lens:list, 
    add_mark:str, context_k:int, max_length:int, tokenizer
    ):

    '''
    create segment contains two event mentions

    segment format: xxx E1_START trigger1 E1_END xxx E2_START trigger2 E2_END xxx

    Return:
    {
        'new_sent': new segment contains two event mentions, 
        'e1_sent_start': trigger1 position, 
        'e1_trigger': trigger1, 
        'e1s_sent_start': E1_START position, 
        'e1e_sent_start': E1_END position, 
        'e2_sent_start': trigger2 position, 
        'e2_trigger': trigger2, 
        'e2s_sent_start': E2_START position, 
        'e2e_sent_start': E2_END position
    }
    '''

    context_before_list =  [sent['text'] for sent in sents[sent1_idx - context_k if sent1_idx >= context_k else 0 : sent1_idx]]
    context_before_lengths = [sent_len for sent_len in sents_lens[sent1_idx - context_k if sent1_idx >= context_k else 0 : sent1_idx]]
    context_next_list = [sent['text'] for sent in sents[sent2_idx + 1 : sent2_idx + context_k + 1 if sent2_idx + context_k < len(sents) else len(sents)]]
    context_next_lengths = [sent_len for sent_len in sents_lens[sent2_idx + 1 : sent2_idx + context_k + 1 if sent2_idx + context_k < len(sents) else len(sents)]]

    e1s, e1e, e2s, e2e = ( # bert
        '[EVENT1_START]', '[EVENT1_END]', '[EVENT2_START]', '[EVENT2_END]'
    ) if add_mark=='bert' else ( # roberta & longformer
        '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>'
    )

    if sent1_idx == sent2_idx: # two events in the same sentence
        assert e1_sent_start < e2_sent_start
        sent_text = sents[sent1_idx]['text']
        total_length = sents_lens[sent1_idx] + 8
        before, middle, next = sent_text[:e1_sent_start], sent_text[e1_sent_start + len(e1_trigger):e2_sent_start], sent_text[e2_sent_start + len(e2_trigger):]
        new_sent_before, new_sent_middle, new_sent_next = f'{before}{e1s}', f'{e1e}{middle}{e2s}', f'{e2e}{next}'
        for before_sen, sen_len in zip(context_before_list[-1::-1], context_before_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent_before = before_sen + ' ' + new_sent_before
        for next_sen, sen_len in zip(context_next_list[-1::-1], context_next_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent_next += ' ' + next_sen
        new_sent = f'{new_sent_before}{e1_trigger}{new_sent_middle}{e2_trigger}{new_sent_next}'
        
        e1_new_sent_start = len(new_sent_before)
        e1s_new_sent_start = e1_new_sent_start - len(e1s)
        e1e_new_sent_start = e1_new_sent_start + len(e1_trigger)
        e2_new_sent_start = len(new_sent_before) + len(e1_trigger) + len(new_sent_middle)
        e2s_new_sent_start = e2_new_sent_start - len(e2s)
        e2e_new_sent_start = e2_new_sent_start + len(e2_trigger)
        
        assert new_sent[e1_new_sent_start:e1_new_sent_start + len(e1_trigger)] == e1_trigger
        assert new_sent[e1s_new_sent_start:e1s_new_sent_start + len(e1s)] == e1s
        assert new_sent[e1e_new_sent_start:e1e_new_sent_start + len(e1e)] == e1e
        assert new_sent[e2_new_sent_start:e2_new_sent_start + len(e2_trigger)] == e2_trigger
        assert new_sent[e2s_new_sent_start:e2s_new_sent_start + len(e2s)] == e2s
        assert new_sent[e2e_new_sent_start:e2e_new_sent_start + len(e2e)] == e2e
        return {
            'new_sent': new_sent, 
            'e1_trigger': e1_trigger, 
            'e1_sent_start': e1_new_sent_start, 
            'e1s_sent_start': e1s_new_sent_start, 
            'e1e_sent_start': e1e_new_sent_start, 
            'e2_trigger': e2_trigger, 
            'e2_sent_start': e2_new_sent_start, 
            'e2s_sent_start': e2s_new_sent_start, 
            'e2e_sent_start': e2e_new_sent_start
        }
    else: # events in different sentence
        before_1, next_1 = sents[sent1_idx]['text'][:e1_sent_start], sents[sent1_idx]['text'][e1_sent_start + len(e1_trigger):]
        before_2, next_2 = sents[sent2_idx]['text'][:e2_sent_start], sents[sent2_idx]['text'][e2_sent_start + len(e2_trigger):]
        total_length = sents_lens[sent1_idx] + sents_lens[sent2_idx] + 8
        if total_length > max_length:
            span_length = (max_length - 80) // 4
            before_1 = ' '.join([c for c in before_1.split(' ') if c != ''][-span_length:]).strip()
            next_1 = ' '.join([c for c in next_1.split(' ') if c != ''][:span_length]).strip()
            before_2 = ' '.join([c for c in before_2.split(' ') if c != ''][-span_length:]).strip()
            next_2 = ' '.join([c for c in next_2.split(' ') if c != ''][:span_length]).strip()
            total_length = len(tokenizer(f'{before_1} {e1_trigger} {next_1} {before_2} {e2_trigger} {next_2}').tokens()) + 8
        new_sent1_before, new_sent1_next = f'{before_1}{e1s}', f'{e1e}{next_1}'
        new_sent2_before, new_sent2_next = f'{before_2}{e2s}', f'{e2e}{next_2}'
        for before_sen, sen_len in zip(context_before_list[-1::-1], context_before_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent1_before = before_sen + ' ' + new_sent1_before
        for next_sen, sen_len in zip(context_next_list[-1::-1], context_next_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent2_next += ' ' + next_sen
        new_sent1 = f'{new_sent1_before}{e1_trigger}{new_sent1_next}'
        new_sent2 = f'{new_sent2_before}{e2_trigger}{new_sent2_next}'
        
        e1_new_sent_start = len(new_sent1_before)
        e1s_new_sent_start = e1_new_sent_start - len(e1s)
        e1e_new_sent_start = e1_new_sent_start + len(e1_trigger)
        e2_new_sent_start = len(new_sent2_before)
        e2s_new_sent_start = e2_new_sent_start - len(e2s)
        e2e_new_sent_start = e2_new_sent_start + len(e2_trigger)
        # add sentences between new_sent1 and new_sent2
        p, q = sent1_idx + 1, sent2_idx - 1
        while p <= q:
            if p == q:
                if total_length + sents_lens[p] <= max_length:
                    new_sent1 += (' ' + sents[p]['text'])
                final_new_sent = new_sent1 + ' ' + new_sent2
                e2_new_sent_start += len(new_sent1) + 1
                e2s_new_sent_start += len(new_sent1) + 1
                e2e_new_sent_start += len(new_sent1) + 1
                
                assert final_new_sent[e1_new_sent_start:e1_new_sent_start + len(e1_trigger)] == e1_trigger
                assert final_new_sent[e1s_new_sent_start:e1s_new_sent_start + len(e1s)] == e1s
                assert final_new_sent[e1e_new_sent_start:e1e_new_sent_start + len(e1e)] == e1e
                assert final_new_sent[e2_new_sent_start:e2_new_sent_start + len(e2_trigger)] == e2_trigger
                assert final_new_sent[e2s_new_sent_start:e2s_new_sent_start + len(e2s)] == e2s
                assert final_new_sent[e2e_new_sent_start:e2e_new_sent_start + len(e2e)] == e2e
                return {
                    'new_sent': final_new_sent, 
                    'e1_trigger': e1_trigger, 
                    'e1_sent_start': e1_new_sent_start, 
                    'e1s_sent_start': e1s_new_sent_start, 
                    'e1e_sent_start': e1e_new_sent_start, 
                    'e2_trigger': e2_trigger, 
                    'e2_sent_start': e2_new_sent_start, 
                    'e2s_sent_start': e2s_new_sent_start, 
                    'e2e_sent_start': e2e_new_sent_start
                }
            if total_length + sents_lens[p] > max_length:
                break
            else:
                total_length += sents_lens[p]
                new_sent1 += (' ' + sents[p]['text'])
            if total_length + sents_lens[q] > max_length:
                break
            else:
                total_length += sents_lens[q]
                new_sent2 = sents[q]['text'] + ' ' + new_sent2
                e2_new_sent_start += len(sents[q]['text']) + 1
                e2s_new_sent_start += len(sents[q]['text']) + 1
                e2e_new_sent_start += len(sents[q]['text']) + 1
            p += 1
            q -= 1
        final_new_sent = new_sent1 + ' ' + new_sent2
        e2_new_sent_start += len(new_sent1) + 1
        e2s_new_sent_start += len(new_sent1) + 1
        e2e_new_sent_start += len(new_sent1) + 1
        assert final_new_sent[e1s_new_sent_start:e1s_new_sent_start + len(e1s)] == e1s
        assert final_new_sent[e1e_new_sent_start:e1e_new_sent_start + len(e1e)] == e1e
        assert final_new_sent[e2s_new_sent_start:e2s_new_sent_start + len(e2s)] == e2s
        assert final_new_sent[e2e_new_sent_start:e2e_new_sent_start + len(e2e)] == e2e
        assert final_new_sent[e1_new_sent_start:e1_new_sent_start + len(e1_trigger)] == e1_trigger
        assert final_new_sent[e2_new_sent_start:e2_new_sent_start + len(e2_trigger)] == e2_trigger
        return {
            'new_sent': final_new_sent, 
            'e1_trigger': e1_trigger, 
            'e1_sent_start': e1_new_sent_start, 
            'e1s_sent_start': e1s_new_sent_start, 
            'e1e_sent_start': e1e_new_sent_start, 
            'e2_trigger': e2_trigger, 
            'e2_sent_start': e2_new_sent_start, 
            'e2s_sent_start': e2s_new_sent_start, 
            'e2e_sent_start': e2e_new_sent_start
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
                            sentences, sentences_lengths, add_mark, context_k, max_length, tokenizer
                        )
                        Data.append({
                            'id': sample['doc_id'], 
                            'sent': new_event_sent['new_sent'], 
                            'e1_offset': event_1['start'], # event1
                            'e1_trigger': new_event_sent['e1_trigger'], 
                            'e1_subtype': event_1['subtype'], 
                            'e1_start': new_event_sent['e1_sent_start'], 
                            'e1s_start': new_event_sent['e1s_sent_start'], 
                            'e1e_start': new_event_sent['e1e_sent_start'], 
                            'e2_offset': event_2['start'], # event2
                            'e2_trigger': new_event_sent['e2_trigger'], 
                            'e2_subtype': event_2['subtype'], 
                            'e2_start': new_event_sent['e2_sent_start'], 
                            'e2s_start': new_event_sent['e2s_sent_start'], 
                            'e2e_start': new_event_sent['e2e_sent_start'], 
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
                                    sentences, sentences_lengths, add_mark, context_k, max_length, tokenizer
                                )
                                Data.append({
                                    'id': sample['doc_id'], 
                                    'sent': new_event_sent['new_sent'], 
                                    'e1_offset': event_1['start'], # event1
                                    'e1_trigger': new_event_sent['e1_trigger'], 
                                    'e1_subtype': event_1['subtype'], 
                                    'e1_start': new_event_sent['e1_sent_start'], 
                                    'e1s_start': new_event_sent['e1s_sent_start'], 
                                    'e1e_start': new_event_sent['e1e_sent_start'], 
                                    'e2_offset': event_2['start'], # event2
                                    'e2_trigger': new_event_sent['e2_trigger'], 
                                    'e2_subtype': event_2['subtype'], 
                                    'e2_start': new_event_sent['e2_sent_start'], 
                                    'e2s_start': new_event_sent['e2s_sent_start'], 
                                    'e2e_start': new_event_sent['e2e_sent_start'], 
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
                                    sentences, sentences_lengths, add_mark, context_k, max_length, tokenizer
                                )
                                Data.append({
                                    'id': sample['doc_id'], 
                                    'sent': new_event_sent['new_sent'], 
                                    'e1_offset': event_1['start'], # event1
                                    'e1_trigger': new_event_sent['e1_trigger'], 
                                    'e1_subtype': event_1['subtype'], 
                                    'e1_start': new_event_sent['e1_sent_start'], 
                                    'e1s_start': new_event_sent['e1s_sent_start'], 
                                    'e1e_start': new_event_sent['e1e_sent_start'], 
                                    'e2_offset': event_2['start'], # event2
                                    'e2_trigger': new_event_sent['e2_trigger'], 
                                    'e2_subtype': event_2['subtype'], 
                                    'e2_start': new_event_sent['e2_sent_start'], 
                                    'e2s_start': new_event_sent['e2s_sent_start'], 
                                    'e2e_start': new_event_sent['e2e_sent_start'], 
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
                                sentences, sentences_lengths, add_mark, context_k, max_length, tokenizer
                            )
                            Data.append({
                                'id': sample['doc_id'], 
                                'sent': new_event_sent['new_sent'], 
                                'e1_offset': event_1['start'], # event1
                                'e1_trigger': new_event_sent['e1_trigger'], 
                                'e1_subtype': event_1['subtype'], 
                                'e1_start': new_event_sent['e1_sent_start'], 
                                'e1s_start': new_event_sent['e1s_sent_start'], 
                                'e1e_start': new_event_sent['e1e_sent_start'], 
                                'e2_offset': event_2['start'], # event2
                                'e2_trigger': new_event_sent['e2_trigger'], 
                                'e2_subtype': event_2['subtype'], 
                                'e2_start': new_event_sent['e2_sent_start'], 
                                'e2s_start': new_event_sent['e2s_sent_start'], 
                                'e2e_start': new_event_sent['e2e_sent_start'], 
                                'label': 0
                            })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, add_mark:str, collote_fn_type:str, prompt_type:str, verbalizer:dict, batch_size:int=None, shuffle:bool=False):

    assert add_mark in ADD_MARK_TYPE
    assert collote_fn_type in ['normal']
    assert prompt_type in [
        'pmb_d',   # prompt_manual_base + document
        'pmq_d',   # prompt_manual_question + document
        'pb_d',    # prompt_base + document
        'k_pb_d'   # knowledge + prompt_base + document
    ]

    if add_mark == 'bert':
        special_start_end_tokens = BERT_SPECIAL_TOKENS
        e1s_token, e1e_token, e2s_token, e2e_token = '[EVENT1_START]', '[EVENT1_END]', '[EVENT2_START]', '[EVENT2_END]'
        l_token1, l_token2, l_token3, l_token4, l_token5, l_token6 = '[L_TOKEN1]', '[L_TOKEN2]', '[L_TOKEN3]', '[L_TOKEN4]', '[L_TOKEN5]', '[L_TOKEN6]'
        mask_token = '[MASK]'
    else:
        special_start_end_tokens = ROBERTA_SPECIAL_TOKENS
        e1s_token, e1e_token, e2s_token, e2e_token = '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>'
        l_token1, l_token2, l_token3, l_token4, l_token5, l_token6 = '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
        mask_token = '<mask>'
    assert tokenizer.additional_special_tokens == special_start_end_tokens

    pos_id = tokenizer.convert_tokens_to_ids(verbalizer['COREF_TOKEN'])
    neg_id = tokenizer.convert_tokens_to_ids(verbalizer['NONCOREF_TOKEN'])

    def get_prompt(
        source_sent, 
        e1_trigger, e1_start, e1s_start, e1e_start, 
        e2_trigger, e2_start, e2s_start, e2e_start
        ):
        knowledge = 'Events that correspond to the same event usually have related trigger words (predicates) and compatible entities (participants, time, location). '
        if 'pmb_d' in prompt_type: # manual prompt
            prompt = f'In this document, the {e1s_token} {e1_trigger} {e1e_token} event and the {e2s_token} {e2_trigger} {e2e_token} event refer to {mask_token} event. '
            if prompt_type == 'k_pmb_d': 
                prompt = knowledge + prompt
        elif 'pmq_d' in prompt_type: # manual question-style prompt
            prompt = f'In this document, the {e1s_token} {e1_trigger} {e1e_token} event and the {e2s_token} {e2_trigger} {e2e_token} event refer to the same event? {mask_token}. '
            if prompt_type == 'k_pmq_d':
                prompt = knowledge + prompt
        elif 'pb_d' in prompt_type: # learnable prompt
            prompt = f'In this document, {l_token1} {e1s_token} {e1_trigger} {e1e_token} {l_token2} {l_token3} {e2s_token} {e2_trigger} {e2e_token} {l_token4} {l_token5} {mask_token} {l_token6}. '
            if prompt_type == 'k_pb_d': 
                prompt = knowledge + prompt
        
        e1_start, e1s_start, e1e_start = np.asarray([e1_start, e1s_start, e1e_start]) + np.full((3,), len(prompt))
        e2_start, e2s_start, e2e_start = np.asarray([e2_start, e2s_start, e2e_start]) + np.full((3,), len(prompt))
        prompt += source_sent
        assert prompt[e1_start:e1_start + len(e1_trigger)] == e1_trigger
        assert prompt[e1s_start:e1s_start + len(e1s_token)] == e1s_token
        assert prompt[e1e_start:e1e_start + len(e1e_token)] == e1e_token
        assert prompt[e2_start:e2_start + len(e2_trigger)] == e2_trigger
        assert prompt[e2s_start:e2s_start + len(e2s_token)] == e2s_token
        assert prompt[e2e_start:e2e_start + len(e2e_token)] == e2e_token
        # convert char offsets to token idxs
        encoding = tokenizer(prompt, max_length=args.max_seq_length, truncation=True)
        mask_idx = encoding.char_to_token(prompt.find(mask_token))
        e1_idx, e1s_idx, e1e_idx = encoding.char_to_token(e1_start), encoding.char_to_token(e1s_start), encoding.char_to_token(e1e_start)
        e2_idx, e2s_idx, e2e_idx = encoding.char_to_token(e2_start), encoding.char_to_token(e2s_start), encoding.char_to_token(e2e_start)
        assert None not in [mask_idx, e1_idx, e1s_idx, e1e_idx, e2_idx, e2s_idx, e2e_idx]
        
        return {
            'prompt': prompt, 
            'mask_idx': mask_idx, 
            'e1_idx': e1_idx, 
            'e1s_idx': e1s_idx, 
            'e1e_idx': e1e_idx, 
            'e2_start': e2_idx, 
            'e2s_idx': e2s_idx,
            'e2e_idx': e2e_idx
        }

    def collote_fn(batch_samples):
        batch_sen, batch_mask_idx, batch_coref = [], [], []
        for sample in batch_samples:
            prompt_data = get_prompt(
                sample['sent'], 
                sample['e1_trigger'], sample['e1_start'], sample['e1s_start'], sample['e1e_start'], 
                sample['e2_trigger'], sample['e2_start'], sample['e2s_start'], sample['e2e_start']
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
    
    def collote_fn_longformer(batch_samples):
        batch_sen, batch_mask_idx, batch_event_idx, batch_coref = [], [], [], []
        for sample in batch_samples:
            prompt_data = get_prompt(
                sample['sent'], 
                sample['e1_trigger'], sample['e1_start'], sample['e1s_start'], sample['e1e_start'], 
                sample['e2_trigger'], sample['e2_start'], sample['e2s_start'], sample['e2e_start']
            )
            batch_sen.append(prompt_data['prompt'])
            batch_mask_idx.append(prompt_data['mask_idx'])
            batch_event_idx.append([
                prompt_data['e1s_idx'], prompt_data['e1e_idx'], prompt_data['e2s_idx'], prompt_data['e2e_idx']
            ])
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
            'batch_event_idx': batch_event_idx, 
            'labels': batch_label
        }
    
    if collote_fn_type == 'normal':
        select_collote_fn = collote_fn_longformer if add_mark == 'longformer' else collote_fn
    
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
    args.max_seq_length = 512
    args.model_type = 'longformer'
    args.model_checkpoint = '../../PT_MODELS/allenai/longformer-large-4096'

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    special_start_end_tokens = BERT_SPECIAL_TOKENS if args.model_type == 'bert' else  ROBERTA_SPECIAL_TOKENS
    special_tokens_dict = {'additional_special_tokens': special_start_end_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    assert tokenizer.additional_special_tokens == special_start_end_tokens

    # train_data = KBPCoref('../../data/train_filtered.json', add_mark='longformer', context_k=2, tokenizer=tokenizer, max_length=482)
    # print_data_statistic('../../data/train_filtered.json')
    # print(len(train_data))
    # labels = [train_data[s_idx]['label'] for s_idx in range(len(train_data))]
    # print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    # train_data = iter(train_data)
    # for _ in range(5):
    #     print(next(train_data))

    train_small_data = KBPCorefTiny(
        '../../data/train_filtered.json', '../../data/train_filtered_with_cos.json', 
        pos_top_k=0, neg_top_k=10, add_mark='longformer', context_k=2, tokenizer=tokenizer, max_length=512-40
    )
    print_data_statistic('../../data/train_filtered_with_cos.json')
    print(len(train_small_data))
    labels = [train_small_data[s_idx]['label'] for s_idx in range(len(train_small_data))]
    print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    print(train_small_data[0])
    print('Testing dataset...')
    for _ in tqdm(train_small_data):
        pass
    
    verbalizer = {'COREF_TOKEN': 'same', 'NONCOREF_TOKEN': 'different'}
    train_dataloader = get_dataLoader(args, train_small_data, tokenizer, add_mark='longformer', collote_fn_type='normal', prompt_type='pb_d', verbalizer=verbalizer, shuffle=True)
    batch_data = next(iter(train_dataloader))
    batch_X, batch_y = batch_data['batch_inputs'], batch_data['labels']
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', len(batch_y))
    print(batch_X)
    print(batch_y)
    print(tokenizer.decode(batch_X['input_ids'][0]))
    print('Testing dataloader...')
    batch_datas = iter(train_dataloader)
    for step in tqdm(range(len(train_dataloader))):
        next(batch_datas)