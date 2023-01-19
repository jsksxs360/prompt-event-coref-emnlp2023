from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import json
from utils import create_event_simi_dict, cal_cluster_simi
from utils import create_new_sent, get_prompt
from utils import get_all_events_in_cluster

MAX_LOOP_NUM = 100
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
    
    def __init__(self, data_file:str, data_file_with_cos:str, pos_k:int, neg_k:int, add_mark:str, tokenizer, max_length:int):
        '''
        - data_file: source train data file
        - data_file_with_cos: train data file with event similarities
        '''
        assert pos_k > 0 and neg_k > 0 and max_length > 0
        assert add_mark in ADD_MARK_TYPE
        self.tokenizer = tokenizer
        self.pos_k = pos_k
        self.neg_k = neg_k
        self.max_length = max_length
        self.special_token_dict = BERT_SPECIAL_TOKEN_DICT if add_mark=='bert' else ROBERTA_SPECIAL_TOKEN_DICT
        self.data = self.load_data(data_file, data_file_with_cos)
    
    def load_data(self, data_file, data_file_with_cos):
        Data = []
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
                            if sample_num >= self.pos_k or loop_num > MAX_LOOP_NUM:
                                break
                            # randomly select c1_size events to create cluster 1
                            c1_indexs = set(np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False))
                            if c1_indexs in sampled_c1_indexs: # filter same cluster 1
                                continue
                            sampled_c1_indexs.append(c1_indexs)
                            c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                            # randomly select more than c1_size events to create cluster 2
                            c2_size = np.random.randint(c1_size, cluster_size - c1_size + 1)
                            c2_indexs = set(np.random.choice(list(set(range(cluster_size)) - c1_indexs), c2_size, replace=False))
                            c2_events = [event for idx, event in enumerate(cluster_events) if idx in c2_indexs]
                            # create segment contains events from two clusters
                            my_sample = create_new_sent(
                                c1_events, c2_events, sentences, sentences_lengths, 
                                self.special_token_dict, self.tokenizer, self.max_length
                            )
                            if not my_sample:
                                continue
                            my_sample['id'], my_sample['label'] = sample['doc_id'], 1
                            Data.append(my_sample)
                            sample_num += 1
            ##################### non-coref cluster pairs (negtive samples) #####################
            for line in tqdm(f_cos.readlines()): 
                sample = json.loads(line.strip())
                clusters, sentences, events_list = sample['clusters'], sample['sentences'], sample['events']
                event_simi_dict = create_event_simi_dict(sample['event_pairs_id'], sample['event_pairs_cos'], clusters)
                # events_dict = {e['event_id']: e for e in events_list}
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                cluster_sizes = [len(cluster['events']) for cluster in clusters]
                clusters = [get_all_events_in_cluster(events_list, cluster['events']) for cluster in clusters]
                for c_idx, cluster_events in enumerate(clusters):
                    cluster_size = cluster_sizes[c_idx]
                    if cluster_size < 3:
                        continue
                    for c1_size in range(1, cluster_size // 2 + 1):
                        # sample other clusters as cluster 2
                        for other_c_idx, other_cluster_events in enumerate(clusters): 
                            other_cluster_size = cluster_sizes[other_c_idx]
                            if other_c_idx == c_idx or other_cluster_size < 2 or other_cluster_size < c1_size:
                                continue
                            cluster_pair_list = []
                            sample_num, loop_num = 0, 0
                            sampled_c2_indexs = []
                            while True:
                                loop_num += 1
                                if sample_num >= self.neg_k * 2 or loop_num > MAX_LOOP_NUM:
                                    break
                                # randomly select c1_size events to create cluster 1
                                c1_indexs = set(np.random.choice(np.random.permutation(cluster_size), c1_size, replace=False))
                                c1_events = [event for idx, event in enumerate(cluster_events) if idx in c1_indexs]
                                # randomly select more than c1_size events to create cluster 2
                                c2_size = np.random.randint(c1_size, other_cluster_size + 1)
                                c2_indexs = set(np.random.choice(np.random.permutation(other_cluster_size), c2_size, replace=False))
                                if c2_indexs in sampled_c2_indexs: # filter same cluster 2
                                    continue
                                sampled_c2_indexs.append(c2_indexs)
                                c2_events = [event for idx, event in enumerate(other_cluster_events) if idx in c2_indexs]
                                cluster_pair_list.append([c1_events, c2_events, cal_cluster_simi(c1_events, c2_events, event_simi_dict)])
                                sample_num += 1
                            cluster_pair_list.sort(key=lambda x:x[2], reverse=True)
                            for cluster_pair in cluster_pair_list[:self.neg_k]:
                                # create segment contains events from two clusters
                                my_sample = create_new_sent(
                                    cluster_pair[0], cluster_pair[1], sentences, sentences_lengths, 
                                    self.special_token_dict, self.tokenizer, self.max_length
                                )
                                if not my_sample:
                                    continue
                                my_sample['id'], my_sample['label'] = sample['doc_id'], 0
                                Data.append(my_sample)
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
        pos_k=10, neg_k=5, add_mark=args.model_type, tokenizer=tokenizer, max_length=512-40
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
