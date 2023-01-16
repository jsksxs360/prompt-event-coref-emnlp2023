from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import json
from collections import Counter
from utils import create_fake_cluster, create_event_simi_dict
from utils import create_new_sent, get_prompt
from utils import get_all_events_in_cluster

MAX_LOOP_NUM = 1000
RANDOM_SEED = 42

PROMPT_TYPE = [
    'hb_d', 'd_hb',  # hard base template
    'hq_d', 'd_hq',  # hard question-style template
    'sb_d', 'd_sb'   # soft base template
]
BERT_SPECIAL_TOKENS= [
    '[EVENT1_START]', '[EVENT1_END]', '[EVENT2_START]', '[EVENT2_END]', 
    '[L_TOKEN1]', '[L_TOKEN2]', '[L_TOKEN3]', '[L_TOKEN4]', '[L_TOKEN5]', '[L_TOKEN6]'
]
ROBERTA_SPECIAL_TOKENS = [
    '<event1_start>', '<event1_end>', '<event2_start>', '<event2_end>', 
    '<l_token1>', '<l_token2>', '<l_token3>', '<l_token4>', '<l_token5>', '<l_token6>'
]
BERT_SPECIAL_TOKEN_DICT = {
    'e1s_token': '[EVENT1_START]', 'e1e_token': '[EVENT1_END]', 
    'e2s_token': '[EVENT2_START]', 'e2e_token': '[EVENT2_END]', 
    'l_token1': '[L_TOKEN1]', 'l_token2': '[L_TOKEN2]', 'l_token3': '[L_TOKEN3]', 
    'l_token4': '[L_TOKEN4]', 'l_token5': '[L_TOKEN5]', 'l_token6': '[L_TOKEN6]', 
    'mask_token': '[MASK]'
}
ROBERTA_SPECIAL_TOKEN_DICT = {
    'e1s_token': '<event1_start>', 'e1e_token': '<event1_end>', 
    'e2s_token': '<event2_start>', 'e2e_token': '<event2_end>', 
    'l_token1': '<l_token1>', 'l_token2': '<l_token2>', 'l_token3': '<l_token3>', 
    'l_token4': '<l_token4>', 'l_token5': '<l_token5>', 'l_token6': '<l_token6>', 
    'mask_token': '<mask>'
}
ADD_MARK_TYPE = ['bert', 'roberta', 'longformer']

np.random.seed(RANDOM_SEED)

class KBPCorefTiny(Dataset):
    
    def __init__(self, data_file:str, data_file_with_cos:str, pos_r:float, neg_r:float, add_mark:str, tokenizer, max_length:int, fake_cluster_k=2):
        '''
        - data_file: source train data file
        - data_file_with_cos: train data file with event similarities
        '''
        assert 0. < pos_r <= 1. and neg_r > 0. and max_length > 0
        assert add_mark in ADD_MARK_TYPE
        self.tokenizer = tokenizer
        self.data = self.load_data(data_file, data_file_with_cos, add_mark, pos_r, neg_r, max_length, fake_cluster_k)
    
    def load_data(self, data_file, data_file_with_cos, add_mark, pos_r, neg_r, max_length, fake_cluster_k):
        Data = []
        special_token_dict = BERT_SPECIAL_TOKEN_DICT if add_mark=='bert' else ROBERTA_SPECIAL_TOKEN_DICT
        with open(data_file, 'rt', encoding='utf-8') as f, open(data_file_with_cos, 'rt', encoding='utf-8') as f_cos:
            ##################### coref cluster pairs (positive samples) #####################
            for line in tqdm(f.readlines()): 
                sample = json.loads(line.strip())
                clusters, sentences, events_list = sample['clusters'], sample['sentences'], sample['events']
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                for cluster in clusters:
                    cluster_size = len(cluster['events'])
                    if cluster_size < 3:
                        continue
                    cluster_events = get_all_events_in_cluster(events_list, cluster['events'])
                    for c1_size in range(1, cluster_size // 2 + 1):
                        sample_num, loop_num = 0, 0
                        sampled_c1_indexs = []
                        while True:
                            loop_num += 1
                            if sample_num >= cluster_size * pos_r or loop_num > MAX_LOOP_NUM:
                                break
                            c1_indexs = set(np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False)) # random, random, random!
                            if c1_indexs in sampled_c1_indexs: # filter same sampled c1
                                continue
                            sampled_c1_indexs.append(c1_indexs)
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            c2_size = np.random.randint(c1_size, cluster_size - c1_size + 1) # sample cluster 2
                            c2_indexs = set(np.random.choice(list(set(range(cluster_size)) - c1_indexs), c2_size, replace=False))
                            c2_events = [event for idx, event in enumerate(cluster_events) if idx in c2_indexs]
                            my_sample = create_new_sent(
                                c1_events, c2_events, sentences, sentences_lengths, 
                                special_token_dict, tokenizer, max_length
                            )
                            if not my_sample:
                                continue
                            my_sample['id'], my_sample['label'], my_sample['type'] = sample['doc_id'], 1, '1'
                            Data.append(my_sample)
                            sample_num += 1
            ##################### non-coref cluster pairs (negtive samples) #####################
            for line in tqdm(f_cos.readlines()): 
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
                        sample_num, loop_num = 0, 0
                        for other_c_idx, other_cluster_events in enumerate(clusters): # other cluster
                            if sample_num >= cluster_size * neg_r or loop_num > MAX_LOOP_NUM:
                                break
                            other_cluster_size = cluster_sizes[other_c_idx]
                            if other_c_idx == c_idx or other_cluster_size < 2 or other_cluster_size < c1_size:
                                continue
                            loop_num += 1
                            c1_indexs = set(np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False)) # random, random, random!
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            c2_size = np.random.randint(c1_size, other_cluster_size + 1)
                            c2_indexs = set(np.random.choice(np.random.permutation(other_cluster_size), c2_size, replace=False))
                            c2_events = [event for idx, event in enumerate(other_cluster_events) if idx in c2_indexs]
                            my_sample = create_new_sent(
                                c1_events, c2_events, sentences, sentences_lengths, 
                                special_token_dict, tokenizer, max_length
                            )
                            if not my_sample:
                                continue
                            my_sample['id'], my_sample['label'], my_sample['type'] = sample['doc_id'], 0, '0-1'
                            Data.append(my_sample)
                            sample_num += 1
                        sampled_c1_indexs = []
                        while True: # fake cluster 2
                            loop_num += 1
                            if sample_num >= cluster_size * neg_r or loop_num > MAX_LOOP_NUM:
                                break
                            c1_indexs = set(np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False)) # random, random, random!
                            if c1_indexs in sampled_c1_indexs: # filter same sampled c1 (c2)
                                continue
                            sampled_c1_indexs.append(c1_indexs)
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            c2_size = np.random.randint(c1_size, cluster_size - c1_size + 1) # sample cluster 2
                            c2_indexs = set(np.random.choice(list(set(range(cluster_size)) - c1_indexs), c2_size, replace=False))
                            c2_events = [event for idx, event in enumerate(cluster_events) if idx in c2_indexs]
                            fake_clusters = create_fake_cluster(c2_events, event_simi_dict, events_dict, fake_cluster_k)
                            for fake_c2_events in fake_clusters:
                                my_sample = create_new_sent(
                                    c1_events, fake_c2_events, sentences, sentences_lengths, 
                                    special_token_dict, tokenizer, max_length
                                )
                                if not my_sample:
                                    continue
                                my_sample['id'], my_sample['label'], my_sample['type'] = sample['doc_id'], 0, '0-2'
                                Data.append(my_sample)
                                sample_num += 1
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, add_mark:str, collote_fn_type:str, prompt_type:str, verbalizer:dict, batch_size:int=None, shuffle:bool=False):

    assert collote_fn_type in ['normal']
    assert prompt_type in PROMPT_TYPE

    special_token_dict = BERT_SPECIAL_TOKEN_DICT if add_mark=='bert' else ROBERTA_SPECIAL_TOKEN_DICT

    pos_id = tokenizer.convert_tokens_to_ids(verbalizer['COREF_TOKEN'])
    neg_id = tokenizer.convert_tokens_to_ids(verbalizer['NONCOREF_TOKEN'])

    def collote_fn(batch_samples):
        batch_sen, batch_mask_idx, batch_coref = [], [], []
        for sample in batch_samples:
            prompt_data = get_prompt(
                prompt_type, special_token_dict, sample['sent'], 
                sample['cluster1_trigger'], sample['cluster2_trigger'], sample['event_s_e_offset'], 
                tokenizer
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
        batch_label = [pos_id if coref == 1 else neg_id for coref in batch_coref]
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'labels': batch_label
        }

    def collote_fn_longformer(batch_samples):
        batch_sen, batch_mask_idx, batch_event_idx, batch_coref = [], [], [], []
        for sample in batch_samples:
            prompt_data = get_prompt(
                prompt_type, special_token_dict, sample['sent'], 
                sample['cluster1_trigger'], sample['cluster2_trigger'], sample['event_s_e_offset'], 
                tokenizer
            )
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
        batch_label = [pos_id if coref == 1 else neg_id for coref in batch_coref]
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

    train_small_data = KBPCorefTiny(
        '../../data/train_filtered.json', '../../data/train_filtered_with_cos.json', 
        pos_r=1., neg_r=1.5, add_mark=args.model_type, tokenizer=tokenizer, max_length=512-40
    )
    print_data_statistic('../../data/train_filtered_with_cos.json')
    print(len(train_small_data))
    labels = [train_small_data[s_idx]['label'] for s_idx in range(len(train_small_data))]
    print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    types = Counter([train_small_data[s_idx]['type'] for s_idx in range(len(train_small_data))])
    print(types.most_common())
    print(train_small_data[0])
    print('Testing dataset...')
    for _ in tqdm(train_small_data):
        pass

    verbalizer = {'COREF_TOKEN': 'same', 'NONCOREF_TOKEN': 'different'}
    train_dataloader = get_dataLoader(args, train_small_data, tokenizer, add_mark=args.model_type, collote_fn_type='normal', prompt_type='sb_d', verbalizer=verbalizer, shuffle=True)
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
