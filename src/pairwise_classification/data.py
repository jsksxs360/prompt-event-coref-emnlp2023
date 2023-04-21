from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm
from collections import defaultdict
from utils import create_sample

# special tokens
BERT_SPECIAL_TOKENS= ['[START]', '[END]']
ROBERTA_SPECIAL_TOKENS = ['<start>', '<end>']

def get_event_cluster_id(event_id:str, clusters:list) -> str:
    for cluster in clusters:
        if event_id in cluster['events']:
            return cluster['hopper_id']
    raise ValueError(f'Unknown event_id: {event_id}')

class KBPCorefPair(Dataset):
    def __init__(self, data_file:str, add_mark:bool, model_type:str, tokenizer, max_length:int):
        assert model_type in ['bert', 'roberta', 'longformer']
        self.add_mark = add_mark
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                sentences_lengths = [len(self.tokenizer.tokenize(sent['text'])) for sent in sentences]
                events = sample['events']
                # create event pairs
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1, event_2 = events[i], events[j]
                        event_1_cluster_id = get_event_cluster_id(event_1['event_id'], clusters)
                        event_2_cluster_id = get_event_cluster_id(event_2['event_id'], clusters)
                        sample_data = create_sample(
                            self.add_mark, 
                            event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                            event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                            sentences, sentences_lengths, 
                            self.model_type, self.tokenizer, self.max_length
                        )
                        Data.append({
                            'id': sample['doc_id'], 
                            'text': sample_data['text'], 
                            'e1_id': event_1['start'], # event1
                            'e1_trigger': event_1['trigger'], 
                            'e1s_offset': sample_data['e1s_offset'], 
                            'e1e_offset': sample_data['e1e_offset'], 
                            'e2_id': event_2['start'], # event2
                            'e2_trigger': event_2['trigger'], 
                            'e2s_offset': sample_data['e2s_offset'], 
                            'e2e_offset': sample_data['e2e_offset'], 
                            'label': 1 if event_1_cluster_id == event_2_cluster_id else 0
                        })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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

def get_noncoref_ids(simi_list, top_k):
    noncoref_ids = []
    for simi in simi_list:
        if simi['coref'] == 0:
            noncoref_ids.append(simi['id'])
            if len(noncoref_ids) >= top_k:
                break
    return noncoref_ids

class KBPCorefPairTiny(Dataset):
    def __init__(self, data_file:str, data_file_with_cos:str, add_mark:bool, neg_top_k:int, model_type:str, tokenizer, max_length:int):
        '''
        - data_file: source train data file
        - data_file_with_cos: train data file with event similarities
        '''
        assert neg_top_k > 0
        assert model_type in ['bert', 'roberta', 'longformer']
        self.add_mark = add_mark
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_file, data_file_with_cos, neg_top_k)

    def load_data(self, data_file, data_file_with_cos, neg_top_k:int):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f, open(data_file_with_cos, 'rt', encoding='utf-8') as f_cos:
            # positive samples (coref pairs)
            for line in tqdm(f.readlines()): 
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                sentences_lengths = [len(self.tokenizer.tokenize(sent['text'])) for sent in sentences]
                events = sample['events']
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1, event_2 = events[i], events[j]
                        event_1_cluster_id = get_event_cluster_id(event_1['event_id'], clusters)
                        event_2_cluster_id = get_event_cluster_id(event_2['event_id'], clusters)
                        if event_1_cluster_id == event_2_cluster_id:
                            sample_data = create_sample(
                                self.add_mark, 
                                event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                                event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                                sentences, sentences_lengths, 
                                self.model_type, self.tokenizer, self.max_length
                            )
                            Data.append({
                                'id': sample['doc_id'], 
                                'text': sample_data['text'], 
                                'e1_id': event_1['start'], # event1
                                'e1_trigger': event_1['trigger'], 
                                'e1s_offset': sample_data['e1s_offset'], 
                                'e1e_offset': sample_data['e1e_offset'], 
                                'e2_id': event_2['start'], # event2
                                'e2_trigger': event_2['trigger'], 
                                'e2s_offset': sample_data['e2s_offset'], 
                                'e2e_offset': sample_data['e2e_offset'], 
                                'label': 1 if event_1_cluster_id == event_2_cluster_id else 0
                            })
            # negtive samples (non-coref pairs)
            for line in tqdm(f_cos.readlines()):
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                sentences_lengths = [len(self.tokenizer.tokenize(sent['text'])) for sent in sentences]
                event_simi_dict = create_event_simi_dict(sample['event_pairs_id'], sample['event_pairs_cos'], clusters)
                events_list, events_dict = sample['events'], {e['event_id']:e for e in sample['events']}
                for i in range(len(events_list)):
                    event_1 = events_list[i]
                    noncoref_event_ids = get_noncoref_ids(event_simi_dict[event_1['event_id']], top_k=neg_top_k)
                    for e_id in noncoref_event_ids: # non-coref
                        event_2 = events_dict[e_id]
                        if event_1['start'] < event_2['start']:
                            sample_data = create_sample(
                                self.add_mark, 
                                event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                                event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                                sentences, sentences_lengths, 
                                self.model_type, self.tokenizer, self.max_length
                            )
                            Data.append({
                                'id': sample['doc_id'], 
                                'text': sample_data['text'], 
                                'e1_id': event_1['start'], # event1
                                'e1_trigger': event_1['trigger'], 
                                'e1s_offset': sample_data['e1s_offset'], 
                                'e1e_offset': sample_data['e1e_offset'], 
                                'e2_id': event_2['start'], # event2
                                'e2_trigger': event_2['trigger'], 
                                'e2s_offset': sample_data['e2s_offset'], 
                                'e2e_offset': sample_data['e2e_offset'], 
                                'label': 0
                            })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):

    def collote_fn(batch_samples):
        batch_sen, batch_e1_idx, batch_e2_idx, batch_labels = [], [], [], []
        for sample in batch_samples:
            batch_sen.append(sample['text'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['text'])
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            assert None not in [e1s_idx, e1e_idx, e2s_idx, e2e_idx]
            batch_e1_idx.append([[e1s_idx, e1e_idx]])
            batch_e2_idx.append([[e2s_idx, e2e_idx]])
            batch_labels.append(int(sample['label']))
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            'batch_inputs': batch_inputs, 
            'batch_e1_idx': batch_e1_idx, 
            'batch_e2_idx': batch_e2_idx, 
            'labels': batch_labels
        }

    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=collote_fn
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
    args.data_include_mark = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    special_tokens_dict = {
        'additional_special_tokens': ['<e1_start>', '<e1_end>', '<e2_start>', '<e2_end>']
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # train_data = KBPCorefPair(
    #     '../../data/train_filtered.json', add_mark=args.data_include_mark, 
    #     model_type=args.model_type, tokenizer=tokenizer, max_length=args.max_seq_length
    # )
    # print_data_statistic('../../data/train_filtered.json')
    # print(len(train_data))
    # labels = [train_data[s_idx]['label'] for s_idx in range(len(train_data))]
    # print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    # for i in range(5):
    #     print(train_data[i])

    train_small_data = KBPCorefPairTiny(
        '../../data/train_filtered.json', '../../data/train_filtered_with_cos.json', 
        add_mark=args.data_include_mark, neg_top_k=3, 
        model_type=args.model_type, tokenizer=tokenizer, max_length=args.max_seq_length
    )
    print(len(train_small_data))
    labels = [train_small_data[s_idx]['label'] for s_idx in range(len(train_small_data))]
    print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    for i in range(5):
        print(train_small_data[i])

    train_dataloader = get_dataLoader(args, train_small_data, tokenizer, args.batch_size, shuffle=True)
    batch_data = next(iter(train_dataloader))
    print('batch_inputs shape:', {k: v.shape for k, v in batch_data['batch_inputs'].items()})
    print('batch_inputs: ', batch_data['batch_inputs'])
    print('batch_e1_idx:', batch_data['batch_e1_idx'])
    print('batch_e2_idx:', batch_data['batch_e2_idx'])
    print('labels:', batch_data['labels'])
    print(tokenizer.decode(batch_data['batch_inputs']['input_ids'][0]))
    print('Testing dataloader...')
    batch_datas = iter(train_dataloader)
    for step in tqdm(range(len(train_dataloader))):
        next(batch_datas)
