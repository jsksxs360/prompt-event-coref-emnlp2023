from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import json
from collections import Counter, defaultdict

MAX_LOOP_NUM = 1000
SPECIAL_TOKENS = [
    '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>', 
    '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
]
np.random.seed(42)

def get_sen_with_events(
    sentence:str,
    cluster1_events, c1_start_token:str, c1_end_token:str, 
    cluster2_events, c2_start_token:str, c2_end_token:str
    ):
    all_events = []
    all_events += [{'offset': event['sent_start'], 'trigger': event['trigger'], 'cluster': 1} for event in cluster1_events]
    all_events += [{'offset': event['sent_start'], 'trigger': event['trigger'], 'cluster': 2} for event in cluster2_events]
    all_events.sort(key=lambda x:x['offset'])
    new_sen, start_p = '', 0
    new_event_offsets = []
    for event in all_events:
        new_sen += sentence[start_p:event['offset']]
        new_event_offsets.append([
            len(new_sen), len(new_sen) + (len(c1_start_token) if event['cluster'] == 1 else len(c2_start_token)) + len(event['trigger'])
        ])
        new_sen += (c1_start_token + event['trigger'] + c1_end_token) if event['cluster'] == 1 else (c2_start_token + event['trigger'] + c2_end_token)
        start_p = event['offset'] + len(event['trigger'])
    new_sen += sentence[start_p:]
    return new_sen, new_event_offsets

get_all_events_in_cluster = lambda event_list, cluster: [event for event in event_list if event['event_id'] in cluster]

def choose_sent_idxs(cluster1_events:list, cluster2_events:list, sent_lengths:list, max_approx_length:int) -> set:
    c1_sent_idxs, c2_sent_idxs = set([e['sent_idx'] for e in cluster1_events]), set([e['sent_idx'] for e in cluster2_events])
    c1_and_c2_sent_idxs = c1_sent_idxs & c2_sent_idxs
    c1_sent_idxs, c2_sent_idxs = sorted(list(c1_sent_idxs - c1_and_c2_sent_idxs)), sorted(list(c2_sent_idxs - c1_and_c2_sent_idxs))
    chosen_sent_idx = set()
    length = 0
    check_c1 = check_c2 = False
    for sent_idx in c1_and_c2_sent_idxs:
        if length + sent_lengths[sent_idx] + 8 > max_approx_length:
            break
        chosen_sent_idx.add(sent_idx)
        length += sent_lengths[sent_idx] + 8
        check_c1 = check_c2 = True
        
    p1 = p2 = 0
    while p1 < len(c1_sent_idxs) and p2 < len(c2_sent_idxs):
        if length + sent_lengths[c1_sent_idxs[p1]] + 4 > max_approx_length:
            break
        chosen_sent_idx.add(c1_sent_idxs[p1])
        length += sent_lengths[c1_sent_idxs[p1]] + 4
        check_c1 = True
        p1 += 1
        if length + sent_lengths[c2_sent_idxs[p2]] + 4 > max_approx_length:
            break
        chosen_sent_idx.add(c2_sent_idxs[p2])
        length += sent_lengths[c2_sent_idxs[p2]] + 4
        check_c2 = True
        p2 += 1
    for sent_idx in c1_sent_idxs[p1:len(c1_sent_idxs)]:
        if length + sent_lengths[sent_idx] + 4 > max_approx_length:
                break
        chosen_sent_idx.add(sent_idx)
        length += sent_lengths[sent_idx] + 4
        check_c1 = True
    for sent_idx in c2_sent_idxs[p2:len(c2_sent_idxs)]:
        if length + sent_lengths[sent_idx] + 4 > max_approx_length:
                break
        chosen_sent_idx.add(sent_idx)
        length += sent_lengths[sent_idx] + 4
        check_c2 = True
    # assert check_c1 and check_c2
    if not (check_c1 and check_c2):
        return None
    chosen_event_num = sum([1 for e in cluster1_events + cluster2_events if e['sent_idx'] in chosen_sent_idx])
    if chosen_event_num < 3:
        return None
    # assert chosen_event_num >= 3, f'At least 3 events should be chosen, now is {chosen_event_num}'
    return chosen_sent_idx

def get_event_cluster_id(event_id:str, clusters:list) -> str:
    for cluster in clusters:
        if event_id in cluster['events']:
            return cluster['hopper_id']
    raise ValueError(f'Unknown event_id: {event_id}')

def create_event_simi_dict(event_pairs_id, event_pairs_cos, clusters):
    simi_dict = defaultdict(list)
    for id_pair, cos in zip(event_pairs_id, event_pairs_cos):
        e1_id, e2_id = id_pair.split('###')
        coref = 1 if get_event_cluster_id(e1_id, clusters) == get_event_cluster_id(e2_id, clusters) else 0
        simi_dict[e1_id].append({'id': e2_id, 'cos': cos, 'coref': coref})
        simi_dict[e2_id].append({'id': e1_id, 'cos': cos, 'coref': coref})
    for simi_list in simi_dict.values():
        simi_list.sort(key=lambda x:x['cos'], reverse=True)
    return simi_dict

def get_noncoref_ids(simi_list):
    noncoref_ids = []
    for simi in simi_list:
        if simi['coref'] == 0:
            noncoref_ids.append(simi['id'])
    return noncoref_ids

def create_fake_cluster(cluster_events, simi_dict, events_dict, fake_cluster_k):
    noncoref_event_ids = {
        e['event_id']:get_noncoref_ids(simi_dict[e['event_id']]) for e in cluster_events
    }
    fake_clusters = []
    for k in range(fake_cluster_k):
        fake_cluster = []
        for e in cluster_events:
            noncoref_events = noncoref_event_ids[e['event_id']]
            if k >= len(noncoref_events):
                break
            for e_id in noncoref_events[k:]:
                if e_id not in fake_cluster:
                    fake_cluster.append(e_id)
                    break
        if len(fake_cluster) == len(cluster_events):
            fake_clusters.append([events_dict[e_id] for e_id in fake_cluster])
    return fake_clusters

class KBPCorefTiny(Dataset):
    
    def __init__(self, data_file:str, data_file_with_cos:str, pos_r:float, neg_r:float, tokenizer, max_length:int, fake_cluster_k=2):
        '''
        - data_file: source train data file
        - data_file_with_cos: train data file with event similarities
        '''
        assert 0. < pos_r <= 1. and neg_r > 0. and max_length > 0
        self.tokenizer = tokenizer
        self.fake_cluster_k = fake_cluster_k
        self.pos_r = pos_r
        self.neg_r = neg_r
        self.max_length = max_length
        self.data = self.load_data(data_file, data_file_with_cos)

    def _get_my_sample(self, cluster1_events, cluster2_events, sentences, sentences_lengths, max_length):
        # choose event sentences to control the total length
        chosen_sent_idxs = choose_sent_idxs(cluster1_events, cluster2_events, sentences_lengths, max_length)
        if not chosen_sent_idxs:
            return None
        sentence_event = {}
        trigger1, trigger2 = [], []
        for e in cluster1_events:
            if e['sent_idx'] not in chosen_sent_idxs:
                continue
            trigger1.append(e['trigger'])
            if e['sent_idx'] not in sentence_event:
                sentence_event[e['sent_idx']] = {
                    'text': sentences[e['sent_idx']]['text'], 
                    'cluster1_events': [e], 
                    'cluster2_events': []
                }
            else:
                sentence_event[e['sent_idx']]['cluster1_events'].append(e)
        for e in cluster2_events:
            if e['sent_idx'] not in chosen_sent_idxs:
                continue
            trigger2.append(e['trigger'])
            if e['sent_idx'] not in sentence_event:
                sentence_event[e['sent_idx']] = {
                    'text': sentences[e['sent_idx']]['text'], 
                    'cluster1_events': [], 
                    'cluster2_events': [e]
                }
            else:
                sentence_event[e['sent_idx']]['cluster2_events'].append(e)
        # select word with the largest number in the cluster as representative trigger
        trigger1, trigger2 = Counter(trigger1).most_common()[0][0], Counter(trigger2).most_common()[0][0]
        sentence_event = sorted(sentence_event.items(), key=lambda x:x[0])
        document, event_s_e_offsets = '', []
        for _, s_e in sentence_event:
            new_sen, new_event_offsets = get_sen_with_events(
                s_e['text'], 
                s_e['cluster1_events'], '<event1_start>', '<event1_end>', 
                s_e['cluster2_events'], '<event2_start>', '<event2_end>'
            )
            event_s_e_offsets += [[s+len(document), e+len(document)] for s, e in new_event_offsets]
            document += new_sen + ' '
        
        document_length = len(self.tokenizer(document).tokens())
        assert document_length <= max_length, f'max length value should >= {document_length}. Now is {max_length}.'
        return {
            'sent': document, 
            'event_s_e_offset': event_s_e_offsets, 
            'cluster1_trigger': trigger1, 
            'cluster2_trigger': trigger2
        }
    
    def load_data(self, data_file, data_file_with_cos):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f, open(data_file_with_cos, 'rt', encoding='utf-8') as f_cos:
            num_file = sum([1 for _ in open(data_file, 'rt', encoding='utf-8')])
            for line in tqdm(f, total=num_file): # coref cluster pairs
                sample = json.loads(line.strip())
                clusters, sentences, events_list = sample['clusters'], sample['sentences'], sample['events']
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                for cluster in clusters:
                    cluster_size = len(cluster['events'])
                    if cluster_size < 3:
                        continue
                    cluster_events = get_all_events_in_cluster(events_list, cluster['events'])
                    for c1_size in range(1, cluster_size // 2 + 1):
                        sample_num = loop_num = 0
                        while True:
                            if sample_num >= cluster_size * self.pos_r or loop_num > MAX_LOOP_NUM:
                                break
                            c1_indexs = np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False) # random, random, random!
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            c2_events = [event for idx, event in enumerate(cluster_events) if idx not in c1_indexs]
                            my_sample = self._get_my_sample(c1_events, c2_events, sentences, sentences_lengths, self.max_length)
                            loop_num += 1
                            if not my_sample:
                                continue
                            my_sample['id'], my_sample['label'] = sample['doc_id'], 1
                            Data.append(my_sample)
                            sample_num += 1
            num_file = sum([1 for _ in open(data_file_with_cos, 'rt', encoding='utf-8')])
            for line in tqdm(f_cos, total=num_file): # non-coref cluster pairs
                sample = json.loads(line.strip())
                clusters, sentences, events_list = sample['clusters'], sample['sentences'], sample['events']
                event_simi_dict = create_event_simi_dict(sample['event_pairs_id'], sample['event_pairs_cos'], clusters)
                events_dict = {e['event_id']:e for e in events_list}
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                cluster_sizes = [len(cluster['events']) for cluster in clusters]
                clusters = [get_all_events_in_cluster(events_list, cluster['events']) for cluster in clusters]
                for c_idx, cluster_events in enumerate(clusters):
                    cluster_size = cluster_sizes[c_idx]
                    if cluster_size < 3 or len(events_list) < 2 * cluster_size:
                        continue
                    for c1_size in range(1, cluster_size // 2 + 1):
                        sample_num = loop_num = 0
                        for other_c_idx, other_cluster_events in enumerate(clusters): # other cluster
                            if sample_num >= cluster_size * self.neg_r or loop_num > MAX_LOOP_NUM:
                                break
                            if other_c_idx == c_idx or cluster_sizes[other_c_idx] < 2 or cluster_sizes[other_c_idx] < c1_size:
                                continue
                            c1_indexs = np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False) # random, random, random!
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            my_sample = self._get_my_sample(c1_events, other_cluster_events, sentences, sentences_lengths, self.max_length)
                            loop_num += 1
                            if not my_sample:
                                continue
                            my_sample['id'], my_sample['label'] = sample['doc_id'], 0
                            Data.append(my_sample)
                            sample_num += 1
                        while True: # fake cluster 2
                            if sample_num >= cluster_size * self.neg_r or loop_num > MAX_LOOP_NUM:
                                break
                            c1_indexs = np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False) # random, random, random!
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            c2_events = [event for idx, event in enumerate(cluster_events) if idx not in c1_indexs]
                            loop_num += 1
                            fake_clusters = create_fake_cluster(c2_events, event_simi_dict, events_dict, self.fake_cluster_k)
                            for fake_c2_events in fake_clusters:
                                my_sample = self._get_my_sample(c1_events, fake_c2_events, sentences, sentences_lengths, self.max_length)
                                if not my_sample:
                                    continue
                                my_sample['id'], my_sample['label'] = sample['doc_id'], 0
                                Data.append(my_sample)
                                sample_num += 1
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, collote_fn_type:str, prompt_type:str, verbalizer:dict, batch_size:int=None, shuffle:bool=False):

    assert collote_fn_type in ['normal']
    assert prompt_type in [
        'pmb_d', 'k_pmb_d',  # prompt_manual_base + document (knowledge)
        'pmq_d', 'k_pmq_d',  # prompt_manual_question + document (knowledge)
        'pb_d', 'k_pb_d'    # prompt_base + document (knowledge)
    ]
    assert tokenizer.additional_special_tokens == SPECIAL_TOKENS
    
    e1s_token, e1e_token, e2s_token, e2e_token = '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>'
    l_token1, l_token2, l_token3, l_token4, l_token5, l_token6 = '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
    mask_token = '<mask>'

    pos_id = tokenizer.convert_tokens_to_ids(verbalizer['COREF_TOKEN'])
    neg_id = tokenizer.convert_tokens_to_ids(verbalizer['NONCOREF_TOKEN'])

    def get_prompt(source_sent, e1_trigger, e2_trigger, event_s_e_offset):
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
        
        event_s_e_offset = [[e_s + len(prompt), e_e + len(prompt)] for e_s, e_e in event_s_e_offset]
        prompt += source_sent
        for e_s, e_e in event_s_e_offset: # check offset
            assert prompt[e_s:e_s + len(e1s_token)] == e1s_token or prompt[e_s:e_s + len(e2s_token)] == e2s_token
            assert prompt[e_e:e_e + len(e1e_token)] == e1e_token or prompt[e_e:e_e + len(e2e_token)] == e2e_token
        # convert char offsets to token idxs
        encoding = tokenizer(prompt, max_length=args.max_seq_length, truncation=True)
        mask_idx = encoding.char_to_token(prompt.find(mask_token))
        assert mask_idx is not None
        event_s_e_idxs = []
        for e_s, e_e in event_s_e_offset:
            e_s_idx, e_e_idx = encoding.char_to_token(e_s), encoding.char_to_token(e_e)
            assert e_s_idx is not None and e_e_idx is not None
            event_s_e_idxs.append([e_s_idx, e_e_idx])
        
        return {
            'prompt': prompt, 
            'mask_idx': mask_idx, 
            'event_idx': event_s_e_idxs
        }
    
    def collote_fn(batch_samples):
        batch_sen, batch_mask_idx, batch_event_idx, batch_coref = [], [], [], []
        for sample in batch_samples:
            prompt_data = get_prompt(sample['sent'], sample['cluster1_trigger'], sample['cluster2_trigger'], sample['event_s_e_offset'])
            batch_sen.append(prompt_data['prompt'])
            batch_mask_idx.append(prompt_data['mask_idx'])
            batch_event_idx.append(prompt_data['event_idx'])
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
    args.max_seq_length = 512
    args.model_type = 'longformer'
    args.model_checkpoint = '../../PT_MODELS/allenai/longformer-large-4096'

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    assert tokenizer.additional_special_tokens == SPECIAL_TOKENS

    train_small_data = KBPCorefTiny(
        '../../data/train_filtered.json', '../../data/train_filtered_with_cos.json', 
        pos_r=1., neg_r=1.5, tokenizer=tokenizer, max_length=512-40
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
    train_dataloader = get_dataLoader(args, train_small_data, tokenizer, collote_fn_type='normal', prompt_type='pb_d', verbalizer=verbalizer, shuffle=True)
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
