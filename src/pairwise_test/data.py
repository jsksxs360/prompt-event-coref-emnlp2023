from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import json

CONTEXT_SENT_NUM = 5

class KBPCorefPair(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        return None

    def load_data(self, data_file, context_k=5):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                events = []
                for e in sample['events']:
                    before =  ' '.join([
                        sent['text'] for sent in sentences[e['sent_idx'] - context_k if e['sent_idx'] >= context_k else 0 : e['sent_idx']]
                    ]).strip()
                    after = ' '.join([
                        sent['text'] for sent in sentences[e['sent_idx'] + 1 : e['sent_idx'] + context_k + 1 if e['sent_idx'] + context_k < len(sentences) else len(sentences)]
                    ]).strip()
                    new_event_sent = before + (' ' if len(before) > 0 else '') + sentences[e['sent_idx']]['text'] + ' ' + after
                    sent_start = e['sent_start'] + (len(before) + 1 if len(before) > 0 else 0)
                    sent_end = sent_start + len(e['trigger']) - 1
                    assert new_event_sent[sent_start:sent_end+1] == e['trigger']
                    events.append({
                        'start': e['start'], 
                        'sent_start': sent_start, 
                        'sent_end': sent_end, 
                        'sent_text': new_event_sent,  
                        'cluster_id': self._get_event_cluster_id(e['event_id'], clusters)
                    })
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1, event_2 = events[i], events[j]
                        Data.append({
                            'id': sample['doc_id'], 
                            'e1_offset': event_1['start'], 
                            'e1_sen': event_1['sent_text'], 
                            'e1_start': event_1['sent_start'], 
                            'e1_end': event_1['sent_end'], 
                            'e2_offset': event_2['start'], 
                            'e2_sen': event_2['sent_text'], 
                            'e2_start': event_2['sent_start'], 
                            'e2_end': event_2['sent_end'], 
                            'label': 1 if event_1['cluster_id'] == event_2['cluster_id'] else 0
                        })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class KBPCorefPairSmall(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        return None

    def load_data(self, data_file, context_k=CONTEXT_SENT_NUM):

        def create_event_simi_dict(event_pairs_id, event_pairs_cos, clusters):
            simi_dict = defaultdict(list)
            for id_pair, cos in zip(event_pairs_id, event_pairs_cos):
                e1_id, e2_id = id_pair.split('###')
                e1_cluster_id, e2_cluster_id = self._get_event_cluster_id(e1_id, clusters), self._get_event_cluster_id(e2_id, clusters)
                simi_dict[e1_id].append({'id': e2_id, 'cos': cos, 'coref': 1 if e1_cluster_id == e2_cluster_id else 0})
                simi_dict[e2_id].append({'id': e1_id, 'cos': cos, 'coref': 1 if e1_cluster_id == e2_cluster_id else 0})
            for simi_list in simi_dict.values():
                simi_list.sort(key=lambda x:x['cos'], reverse=True)
            return simi_dict
        
        def get_noncoref_ids(simi_list, top_k=1):
            noncoref_ids = []
            for simi in simi_list:
                if simi['coref'] == 0:
                    noncoref_ids.append(simi['id'])
                    if len(noncoref_ids) >= top_k:
                        break
            assert len(noncoref_ids) <= top_k
            return noncoref_ids

        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                event_pairs_id, event_pairs_cos = sample['event_pairs_id'], sample['event_pairs_cos']
                event_simi_dict = create_event_simi_dict(event_pairs_id, event_pairs_cos, clusters)
                events_list, events_dict = [], {}
                for e in sample['events']:
                    before =  ' '.join([
                        sent['text'] for sent in sentences[e['sent_idx'] - context_k if e['sent_idx'] >= context_k else 0:e['sent_idx']]
                    ]).strip()
                    after = ' '.join([
                        sent['text'] for sent in sentences[e['sent_idx'] + 1:e['sent_idx'] + context_k + 1 if e['sent_idx'] + context_k < len(sentences) else len(sentences)]
                    ]).strip()
                    new_event_sent = before + (' ' if len(before) > 0 else '') + sentences[e['sent_idx']]['text'] + ' ' + after
                    sent_start = e['sent_start'] + (len(before) + 1 if len(before) > 0 else 0)
                    sent_end = sent_start + len(e['trigger']) - 1
                    assert new_event_sent[sent_start:sent_end+1] == e['trigger']
                    events_list.append({
                        'id': e['event_id'],
                        'start': e['start'], 
                        'sent_start': sent_start, 
                        'sent_end': sent_end, 
                        'sent_text': new_event_sent,  
                        'cluster_id': self._get_event_cluster_id(e['event_id'], clusters)
                    })
                    events_dict[e['event_id']] = {
                        'start': e['start'], 
                        'sent_start': sent_start, 
                        'sent_end': sent_end, 
                        'sent_text': new_event_sent,  
                        'cluster_id': self._get_event_cluster_id(e['event_id'], clusters)
                    }
                for i in range(len(events_list) - 1):
                    event_1 = events_list[i]
                    noncoref_event_ids = get_noncoref_ids(event_simi_dict[event_1['id']], top_k=2)
                    for e_id in noncoref_event_ids: # non-coref
                        event_2 = events_dict[e_id]
                        Data.append({
                            'id': sample['doc_id'], 
                            'e1_offset': event_1['start'], 
                            'e1_sen': event_1['sent_text'], 
                            'e1_start': event_1['sent_start'], 
                            'e1_end': event_1['sent_end'], 
                            'e2_offset': event_2['start'], 
                            'e2_sen': event_2['sent_text'], 
                            'e2_start': event_2['sent_start'], 
                            'e2_end': event_2['sent_end'], 
                            'label': 0
                        })
                    for j in range(i + 1, len(events_list)): # coref
                        event_2 = events_list[j]
                        if event_1['cluster_id'] == event_2['cluster_id']:
                            Data.append({
                                'id': sample['doc_id'], 
                                'e1_offset': event_1['start'], 
                                'e1_sen': event_1['sent_text'], 
                                'e1_start': event_1['sent_start'], 
                                'e1_end': event_1['sent_end'], 
                                'e2_offset': event_2['start'], 
                                'e2_sen': event_2['sent_text'], 
                                'e2_start': event_2['sent_start'], 
                                'e2_end': event_2['sent_end'], 
                                'label': 1
                            })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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

train_data = KBPCorefPair('../../data/train_filtered.json')
print_data_statistic('../../data/train_filtered.json')
print(len(train_data))
labels = [train_data[s_idx]['label'] for s_idx in range(len(train_data))]
print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))

train_small_data = KBPCorefPairSmall('../../data/train_filtered_with_cos.json')
print_data_statistic('../../data/train_filtered_with_cos.json')
print(len(train_small_data))
labels = [train_small_data[s_idx]['label'] for s_idx in range(len(train_small_data))]
print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False, collote_fn_type='normal'):

    def _cut_sent(sent, e_char_start, e_char_end, max_length):
        before = ' '.join([c for c in sent[:e_char_start].split(' ') if c != ''][-max_length:]).strip()
        trigger = sent[e_char_start:e_char_end+1]
        after = ' '.join([c for c in sent[e_char_end+1:].split(' ') if c != ''][:max_length]).strip()
        new_sent, new_char_start, new_char_end = before + ' ' + trigger + ' ' + after, len(before) + 1, len(before) + len(trigger)
        assert new_sent[new_char_start:new_char_end+1] == trigger
        return new_sent, new_char_start, new_char_end

    max_mention_length = (args.max_seq_length - 50) // 4

    def collote_fn(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx, batch_label  = [], [], [], []
        for sample in batch_samples:
            sen_1, e1_char_start, e1_char_end = _cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_mention_length)
            sen_2, e2_char_start, e2_char_end = _cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_mention_length)
            batch_sen_1.append(sen_1)
            batch_sen_2.append(sen_2)
            batch_event_idx.append(
                (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
            )
            batch_label.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_e1_token_idx, batch_e2_token_idx = [], []
        for sen_1, sen_2, event_idx in zip(batch_sen_1, batch_sen_2, batch_event_idx):
            e1_char_start, e1_char_end, e2_char_start, e2_char_end = event_idx
            encoding = tokenizer(sen_1, sen_2, max_length=args.max_seq_length, truncation=True)
            e1_start = encoding.char_to_token(e1_char_start, sequence_index=0)
            if not e1_start:
                e1_start = encoding.char_to_token(e1_char_start + 1, sequence_index=0)
            e1_end = encoding.char_to_token(e1_char_end, sequence_index=0)
            e2_start = encoding.char_to_token(e2_char_start, sequence_index=1)
            if not e2_start:
                e2_start = encoding.char_to_token(e2_char_start + 1, sequence_index=1)
            e2_end = encoding.char_to_token(e2_char_end, sequence_index=1)
            assert e1_start and e1_end and e2_start and e2_end
            batch_e1_token_idx.append([[e1_start, e1_end]])
            batch_e2_token_idx.append([[e2_start, e2_end]])
        return {
            'batch_inputs': batch_inputs, 
            'batch_e1_idx': batch_e1_token_idx, 
            'batch_e2_idx': batch_e2_token_idx, 
            'labels': batch_label
        }
    
#     def collote_fn_with_mask(batch_samples):
#         batch_sen_1, batch_sen_2, batch_event_idx = [], [], []
#         batch_label, batch_subtypes = [], []
#         for sample in batch_samples:
#             sen_1, e1_char_start, e1_char_end = _cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_mention_length)
#             sen_2, e2_char_start, e2_char_end = _cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_mention_length)
#             batch_sen_1.append(sen_1)
#             batch_sen_2.append(sen_2)
#             batch_event_idx.append(
#                 (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
#             )
#             batch_label.append(sample['label'])
#             batch_subtypes.append([sample['e1_subtype'], sample['e2_subtype']])
#         batch_inputs = tokenizer(
#             batch_sen_1, 
#             batch_sen_2, 
#             max_length=args.max_seq_length, 
#             padding=True, 
#             truncation=True, 
#             return_tensors="pt"
#         )
#         batch_inputs_with_mask = tokenizer(
#             batch_sen_1, 
#             batch_sen_2, 
#             max_length=args.max_seq_length, 
#             padding=True, 
#             truncation=True, 
#             return_tensors="pt"
#         )
#         batch_e1_token_idx, batch_e2_token_idx = [], []
#         for sen_1, sen_2, event_idx in zip(batch_sen_1, batch_sen_2, batch_event_idx):
#             e1_char_start, e1_char_end, e2_char_start, e2_char_end = event_idx
#             encoding = tokenizer(sen_1, sen_2, max_length=args.max_seq_length, truncation=True)
#             e1_start = encoding.char_to_token(e1_char_start, sequence_index=0)
#             if not e1_start:
#                 e1_start = encoding.char_to_token(e1_char_start + 1, sequence_index=0)
#             e1_end = encoding.char_to_token(e1_char_end, sequence_index=0)
#             e2_start = encoding.char_to_token(e2_char_start, sequence_index=1)
#             if not e2_start:
#                 e2_start = encoding.char_to_token(e2_char_start + 1, sequence_index=1)
#             e2_end = encoding.char_to_token(e2_char_end, sequence_index=1)
#             assert e1_start and e1_end and e2_start and e2_end
#             batch_e1_token_idx.append([[e1_start, e1_end]])
#             batch_e2_token_idx.append([[e2_start, e2_end]])
#         for b_idx in range(len(batch_label)):
#             e1_start, e1_end = batch_e1_token_idx[b_idx][0]
#             e2_start, e2_end = batch_e2_token_idx[b_idx][0]
#             batch_inputs_with_mask['input_ids'][b_idx][e1_start:e1_end+1] = tokenizer.mask_token_id
#             batch_inputs_with_mask['input_ids'][b_idx][e2_start:e2_end+1] = tokenizer.mask_token_id
#         return {
#             'batch_inputs': batch_inputs, 
#             'batch_inputs_with_mask': batch_inputs_with_mask, 
#             'batch_e1_idx': batch_e1_token_idx, 
#             'batch_e2_idx': batch_e2_token_idx, 
#             'labels': batch_label, 
#             'subtypes': batch_subtypes
#         }
    


    if collote_fn_type == 'normal':
        select_collote_fn = collote_fn
#     elif collote_fn_type == 'with_mask':
#         select_collote_fn = collote_fn_with_mask
#     elif collote_fn_type == 'with_dist':
#         select_collote_fn = collote_fn_with_dist
#     elif collote_fn_type == 'with_mask_dist':
#         select_collote_fn = collote_fn_with_mask_dist

    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=select_collote_fn
    )
