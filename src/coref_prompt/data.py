from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm
from collections import defaultdict
from prompt import PROMPT_TYPE, SELECT_ARG_STRATEGY, EVENT_SUBTYPES, subtype2id, id2subtype
from prompt import create_prompt

def get_pred_related_info(simi_file:str) -> dict:
    '''
    # Returns:
        - related info dictionary: {doc_id: {event_offset: {
            'arguments': [{"global_offset": 798, "mention": "We", "role": "participant"}]
            'related_triggers': ['charged'], 
            'related_arguments': [
                {'global_offset': 1408, 'mention': 'Garvina', 'role': 'participant'}, 
                {'global_offset': 1368, 'mention': 'Prosecutors', 'role': 'participant'}
            ]
        }}}
    '''
    related_info_dict = {}
    with open(simi_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            related_info_dict[sample['doc_id']] = {
                int(offset): {
                    'arguments': related_info['arguments'], 
                    'related_triggers': related_info['related_triggers'], 
                    'related_arguments': related_info['related_arguments']
                }
                for offset, related_info in sample['relate_info'].items()
            }
    return related_info_dict

def get_event_cluster_id(event_id:str, clusters:list) -> str:
    for cluster in clusters:
        if event_id in cluster['events']:
            return cluster['hopper_id']
    raise ValueError(f'Unknown event_id: {event_id}')

class KBPCoref(Dataset):
    def __init__(self, data_file:str, simi_file:str, prompt_type:str, select_arg_strategy:str, model_type:str, tokenizer, max_length:int):
        assert prompt_type in PROMPT_TYPE and select_arg_strategy in SELECT_ARG_STRATEGY and model_type in ['bert', 'roberta', 'longformer']
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.related_dict = get_pred_related_info(simi_file)
        self.data = self.load_data(data_file, prompt_type, select_arg_strategy)
    
    def load_data(self, data_file, prompt_type:str, select_arg_strategy:str):
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
                        event_1_related_info = self.related_dict[sample['doc_id']][event_1['start']]
                        event_2_related_info = self.related_dict[sample['doc_id']][event_2['start']]
                        prompt_data = create_prompt(
                            event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], event_1_related_info, 
                            event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], event_2_related_info, 
                            sentences, sentences_lengths, 
                            prompt_type, select_arg_strategy, 
                            self.model_type, self.tokenizer, self.max_length
                        )
                        Data.append({
                            'id': sample['doc_id'], 
                            'prompt': prompt_data['prompt'], 
                            'mask_offset': prompt_data['mask_offset'], 
                            'type_match_mask_offset': prompt_data['type_match_mask_offset'], 
                            'arg_match_mask_offset': prompt_data['arg_match_mask_offset'], 
                            'trigger_offsets': prompt_data['trigger_offsets'], 
                            'e1_id': event_1['start'], # event1
                            'e1_trigger': event_1['trigger'], 
                            'e1_subtype': event_1['subtype'] if event_1['subtype'] in EVENT_SUBTYPES else 'normal', 
                            'e1_subtype_id': subtype2id.get(event_1['subtype'], 0), # 0 - 'other'
                            'e1s_offset': prompt_data['e1s_offset'], 
                            'e1e_offset': prompt_data['e1e_offset'], 
                            'e1_type_mask_offset': prompt_data['e1_type_mask_offset'], 
                            'e2_id': event_2['start'], # event2
                            'e2_trigger': event_2['trigger'], 
                            'e2_subtype': event_2['subtype'] if event_2['subtype'] in EVENT_SUBTYPES else 'normal', 
                            'e2_subtype_id': subtype2id.get(event_2['subtype'], 0), # 0 - 'other'
                            'e2s_offset': prompt_data['e2s_offset'], 
                            'e2e_offset': prompt_data['e2e_offset'], 
                            'e2_type_mask_offset': prompt_data['e2_type_mask_offset'], 
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

class KBPCorefTiny(Dataset):
    def __init__(self, data_file:str, data_file_with_cos:str, simi_file:str, neg_top_k:int, 
                 prompt_type:str, select_arg_strategy:str, model_type:str, tokenizer, max_length:int):
        '''
        - data_file: source train data file
        - data_file_with_cos: train data file with event similarities
        '''
        assert prompt_type in PROMPT_TYPE and select_arg_strategy in SELECT_ARG_STRATEGY and model_type in ['bert', 'roberta', 'longformer']
        assert neg_top_k > 0
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.related_dict = get_pred_related_info(simi_file)
        self.data = self.load_data(data_file, data_file_with_cos, neg_top_k, prompt_type, select_arg_strategy)
    
    def load_data(self, data_file, data_file_with_cos, neg_top_k, prompt_type:str, select_arg_strategy:str):
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
                        event_1_related_info = self.related_dict[sample['doc_id']][event_1['start']]
                        event_2_related_info = self.related_dict[sample['doc_id']][event_2['start']]
                        if event_1_cluster_id == event_2_cluster_id:
                            prompt_data = create_prompt(
                                event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], event_1_related_info, 
                                event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], event_2_related_info, 
                                sentences, sentences_lengths, 
                                prompt_type, select_arg_strategy, 
                                self.model_type, self.tokenizer, self.max_length
                            )
                            Data.append({
                                'id': sample['doc_id'], 
                                'prompt': prompt_data['prompt'], 
                                'mask_offset': prompt_data['mask_offset'], 
                                'type_match_mask_offset': prompt_data['type_match_mask_offset'], 
                                'arg_match_mask_offset': prompt_data['arg_match_mask_offset'], 
                                'trigger_offsets': prompt_data['trigger_offsets'], 
                                'e1_id': event_1['start'], # event1
                                'e1_trigger': event_1['trigger'], 
                                'e1_subtype': event_1['subtype'] if event_1['subtype'] in EVENT_SUBTYPES else 'normal', 
                                'e1_subtype_id': subtype2id.get(event_1['subtype'], 0), # 0 - 'other'
                                'e1s_offset': prompt_data['e1s_offset'], 
                                'e1e_offset': prompt_data['e1e_offset'], 
                                'e1_type_mask_offset': prompt_data['e1_type_mask_offset'], 
                                'e2_id': event_2['start'], # event2
                                'e2_trigger': event_2['trigger'], 
                                'e2_subtype': event_2['subtype'] if event_2['subtype'] in EVENT_SUBTYPES else 'normal', 
                                'e2_subtype_id': subtype2id.get(event_2['subtype'], 0), # 0 - 'other'
                                'e2s_offset': prompt_data['e2s_offset'], 
                                'e2e_offset': prompt_data['e2e_offset'], 
                                'e2_type_mask_offset': prompt_data['e2_type_mask_offset'], 
                                'label': 1
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
                            event_1_related_info = self.related_dict[sample['doc_id']][event_1['start']]
                            event_2_related_info = self.related_dict[sample['doc_id']][event_2['start']]
                            prompt_data = create_prompt(
                                event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], event_1_related_info, 
                                event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], event_2_related_info, 
                                sentences, sentences_lengths, 
                                prompt_type, select_arg_strategy, 
                                self.model_type, self.tokenizer, self.max_length
                            )
                            Data.append({
                                'id': sample['doc_id'], 
                                'prompt': prompt_data['prompt'], 
                                'mask_offset': prompt_data['mask_offset'], 
                                'type_match_mask_offset': prompt_data['type_match_mask_offset'], 
                                'arg_match_mask_offset': prompt_data['arg_match_mask_offset'], 
                                'trigger_offsets': prompt_data['trigger_offsets'], 
                                'e1_id': event_1['start'], # event1
                                'e1_trigger': event_1['trigger'], 
                                'e1_subtype': event_1['subtype'] if event_1['subtype'] in EVENT_SUBTYPES else 'normal', 
                                'e1_subtype_id': subtype2id.get(event_1['subtype'], 0), # 0 - 'other'
                                'e1s_offset': prompt_data['e1s_offset'], 
                                'e1e_offset': prompt_data['e1e_offset'], 
                                'e1_type_mask_offset': prompt_data['e1_type_mask_offset'], 
                                'e2_id': event_2['start'], # event2
                                'e2_trigger': event_2['trigger'], 
                                'e2_subtype': event_2['subtype'] if event_2['subtype'] in EVENT_SUBTYPES else 'normal', 
                                'e2_subtype_id': subtype2id.get(event_2['subtype'], 0), # 0 - 'other'
                                'e2s_offset': prompt_data['e2s_offset'], 
                                'e2e_offset': prompt_data['e2e_offset'], 
                                'e2_type_mask_offset': prompt_data['e2_type_mask_offset'], 
                                'label': 0
                            })
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, prompt_type:str, verbalizer:dict, with_mask:bool=None, batch_size:int=None, shuffle:bool=False):
    assert prompt_type in PROMPT_TYPE
    pos_id, neg_id = verbalizer['coref']['id'], verbalizer['non-coref']['id']
    if prompt_type.startswith('m'):
        match_id, mismatch_id = verbalizer['match']['id'], verbalizer['mismatch']['id']
    if prompt_type.startswith('m') or prompt_type.startswith('t'):
        event_type_ids = {
            s_id: verbalizer[subtype]['id']
            for s_id, subtype in id2subtype.items()
        }

    def collote_fn(batch_samples):
        batch_sen, batch_mask_idx, batch_event_idx, batch_labels = [], [], [], []
        for sample in batch_samples:
            batch_sen.append(sample['prompt'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['prompt'])
            mask_idx = encoding.char_to_token(sample['mask_offset'])
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            assert None not in [mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx]
            batch_mask_idx.append(mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
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
            'batch_mask_idx': batch_mask_idx, 
            'batch_event_idx': batch_event_idx, 
            'label_word_id': [neg_id, pos_id], 
            'labels': batch_labels
        }
    
    def collote_fn_with_mask(batch_samples):
        batch_sen = []
        batch_mask_idx, batch_event_idx, batch_trigger_idx = [], [], []
        batch_labels = []
        for sample in batch_samples:
            batch_sen.append(sample['prompt'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['prompt'])
            mask_idx = encoding.char_to_token(sample['mask_offset'])
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            trigger_idxs = [
                [encoding.char_to_token(s), encoding.char_to_token(e)]
                for s, e in sample['trigger_offsets']
            ]
            assert None not in [mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx]
            for s, e in trigger_idxs:
                assert None not in [s, e]
            batch_mask_idx.append(mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
            batch_trigger_idx.append(trigger_idxs)
            batch_labels.append(int(sample['label']))
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_mask_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        for b_idx in range(len(batch_labels)):
            for s, e in batch_trigger_idx[b_idx]:
                batch_mask_inputs['input_ids'][b_idx][s:e+1] = tokenizer.mask_token_id
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_inputs': batch_mask_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'batch_event_idx': batch_event_idx, 
            'label_word_id': [neg_id, pos_id], 
            'labels': batch_labels
        }
    
    def collote_fn_subtype(batch_samples):
        batch_sen, batch_mask_idx, batch_event_idx  = [], [], []
        batch_e1_type_mask_idx, batch_e2_type_mask_idx = [], []
        batch_labels, batch_e1_type_labels, batch_e2_type_labels = [], [], []
        for sample in batch_samples:
            batch_sen.append(sample['prompt'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['prompt'])
            mask_idx = encoding.char_to_token(sample['mask_offset'])
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            e1_type_mask_idx, e2_type_mask_idx = (
                encoding.char_to_token(sample['e1_type_mask_offset']), 
                encoding.char_to_token(sample['e2_type_mask_offset'])
            )
            assert None not in [
                mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx, 
                e1_type_mask_idx, e2_type_mask_idx
            ]
            batch_mask_idx.append(mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
            batch_e1_type_mask_idx.append(e1_type_mask_idx)
            batch_e2_type_mask_idx.append(e2_type_mask_idx)
            batch_labels.append(int(sample['label']))
            batch_e1_type_labels.append(int(sample['e1_subtype_id']))
            batch_e2_type_labels.append(int(sample['e2_subtype_id']))
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'batch_event_idx': batch_event_idx, 
            'batch_t1_mask_idx': batch_e1_type_mask_idx, 
            'batch_t2_mask_idx': batch_e2_type_mask_idx, 
            'label_word_id': [neg_id, pos_id], 
            'subtype_label_word_id': [event_type_ids[i] for i in range(len(EVENT_SUBTYPES) + 1)], 
            'labels': batch_labels, 
            'e1_subtype_labels': batch_e1_type_labels, 
            'e2_subtype_labels': batch_e2_type_labels
        }
    
    def collote_fn_subtype_with_mask(batch_samples):
        batch_sen = []
        batch_mask_idx, batch_event_idx, batch_trigger_idx  = [], [], []
        batch_e1_type_mask_idx, batch_e2_type_mask_idx = [], []
        batch_labels, batch_e1_type_labels, batch_e2_type_labels = [], [], []
        for sample in batch_samples:
            batch_sen.append(sample['prompt'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['prompt'])
            mask_idx = encoding.char_to_token(sample['mask_offset'])
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            trigger_idxs = [
                [encoding.char_to_token(s), encoding.char_to_token(e)]
                for s, e in sample['trigger_offsets']
            ]
            e1_type_mask_idx, e2_type_mask_idx = (
                encoding.char_to_token(sample['e1_type_mask_offset']), 
                encoding.char_to_token(sample['e2_type_mask_offset'])
            )
            assert None not in [
                mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx, 
                e1_type_mask_idx, e2_type_mask_idx
            ]
            for s, e in trigger_idxs:
                assert None not in [s, e]
            batch_mask_idx.append(mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
            batch_trigger_idx.append(trigger_idxs)
            batch_e1_type_mask_idx.append(e1_type_mask_idx)
            batch_e2_type_mask_idx.append(e2_type_mask_idx)
            batch_labels.append(int(sample['label']))
            batch_e1_type_labels.append(int(sample['e1_subtype_id']))
            batch_e2_type_labels.append(int(sample['e2_subtype_id']))
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_mask_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        for b_idx in range(len(batch_labels)):
            for s, e in batch_trigger_idx[b_idx]:
                batch_mask_inputs['input_ids'][b_idx][s:e+1] = tokenizer.mask_token_id
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'batch_event_idx': batch_event_idx, 
            'batch_t1_mask_idx': batch_e1_type_mask_idx, 
            'batch_t2_mask_idx': batch_e2_type_mask_idx, 
            'label_word_id': [neg_id, pos_id], 
            'subtype_label_word_id': [event_type_ids[i] for i in range(len(EVENT_SUBTYPES) + 1)], 
            'labels': batch_labels, 
            'e1_subtype_labels': batch_e1_type_labels, 
            'e2_subtype_labels': batch_e2_type_labels
        }

    def collote_fn_subtype_match(batch_samples):
        batch_sen, batch_mask_idx, batch_event_idx = [], [], []
        batch_type_match_mask_idx, batch_arg_match_mask_idx = [], []
        batch_e1_type_mask_idx, batch_e2_type_mask_idx = [], []
        batch_labels = []
        batch_type_match_labels, batch_arg_match_labels = [], []
        batch_e1_type_labels, batch_e2_type_labels = [], []
        for sample in batch_samples:
            batch_sen.append(sample['prompt'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['prompt'])
            mask_idx, type_match_mask_idx, arg_match_mask_idx = (
                encoding.char_to_token(sample['mask_offset']), 
                encoding.char_to_token(sample['type_match_mask_offset']), 
                encoding.char_to_token(sample['arg_match_mask_offset']), 
            )
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            e1_type_mask_idx, e2_type_mask_idx = (
                encoding.char_to_token(sample['e1_type_mask_offset']), 
                encoding.char_to_token(sample['e2_type_mask_offset'])
            )
            assert None not in [
                mask_idx, type_match_mask_idx, arg_match_mask_idx, 
                e1s_idx, e1e_idx, e2s_idx, e2e_idx, e1_type_mask_idx, e2_type_mask_idx
            ]
            batch_mask_idx.append(mask_idx)
            batch_type_match_mask_idx.append(type_match_mask_idx)
            batch_arg_match_mask_idx.append(arg_match_mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
            batch_e1_type_mask_idx.append(e1_type_mask_idx)
            batch_e2_type_mask_idx.append(e2_type_mask_idx)
            batch_labels.append(int(sample['label']))
            batch_type_match_labels.append(int(sample['e1_subtype_id'] == sample['e2_subtype_id']))
            batch_arg_match_labels.append(int(sample['label']))
            batch_e1_type_labels.append(int(sample['e1_subtype_id']))
            batch_e2_type_labels.append(int(sample['e2_subtype_id']))
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'batch_type_match_mask_idx': batch_type_match_mask_idx, 
            'batch_arg_match_mask_idx': batch_arg_match_mask_idx, 
            'batch_event_idx': batch_event_idx, 
            'batch_t1_mask_idx': batch_e1_type_mask_idx, 
            'batch_t2_mask_idx': batch_e2_type_mask_idx, 
            'label_word_id': [neg_id, pos_id], 
            'match_label_word_id': [match_id, mismatch_id], 
            'subtype_label_word_id': [event_type_ids[i] for i in range(len(EVENT_SUBTYPES) + 1)], 
            'labels': batch_labels, 
            'subtype_match_labels': batch_type_match_labels, 
            'arg_match_labels': batch_arg_match_labels, 
            'e1_subtype_labels': batch_e1_type_labels, 
            'e2_subtype_labels': batch_e2_type_labels
        }
    
    def collote_fn_subtype_match_with_mask(batch_samples):
        batch_sen = []
        batch_mask_idx, batch_event_idx, batch_trigger_idx = [], [], []
        batch_type_match_mask_idx, batch_arg_match_mask_idx = [], []
        batch_e1_type_mask_idx, batch_e2_type_mask_idx = [], []
        batch_labels = []
        batch_type_match_labels, batch_arg_match_labels = [], []
        batch_e1_type_labels, batch_e2_type_labels = [], []
        for sample in batch_samples:
            batch_sen.append(sample['prompt'])
            # convert char offsets to token idxs
            encoding = tokenizer(sample['prompt'])
            mask_idx, type_match_mask_idx, arg_match_mask_idx = (
                encoding.char_to_token(sample['mask_offset']), 
                encoding.char_to_token(sample['type_match_mask_offset']), 
                encoding.char_to_token(sample['arg_match_mask_offset']), 
            )
            e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
                encoding.char_to_token(sample['e1s_offset']), 
                encoding.char_to_token(sample['e1e_offset']), 
                encoding.char_to_token(sample['e2s_offset']), 
                encoding.char_to_token(sample['e2e_offset'])
            )
            trigger_idxs = [
                [encoding.char_to_token(s), encoding.char_to_token(e)]
                for s, e in sample['trigger_offsets']
            ]
            e1_type_mask_idx, e2_type_mask_idx = (
                encoding.char_to_token(sample['e1_type_mask_offset']), 
                encoding.char_to_token(sample['e2_type_mask_offset'])
            )
            assert None not in [
                mask_idx, type_match_mask_idx, arg_match_mask_idx, 
                e1s_idx, e1e_idx, e2s_idx, e2e_idx, e1_type_mask_idx, e2_type_mask_idx
            ]
            for s, e in trigger_idxs:
                assert None not in [s, e]
            batch_mask_idx.append(mask_idx)
            batch_type_match_mask_idx.append(type_match_mask_idx)
            batch_arg_match_mask_idx.append(arg_match_mask_idx)
            batch_event_idx.append([e1s_idx, e1e_idx, e2s_idx, e2e_idx])
            batch_trigger_idx.append(trigger_idxs)
            batch_e1_type_mask_idx.append(e1_type_mask_idx)
            batch_e2_type_mask_idx.append(e2_type_mask_idx)
            batch_labels.append(int(sample['label']))
            batch_type_match_labels.append(int(sample['e1_subtype_id'] == sample['e2_subtype_id']))
            batch_arg_match_labels.append(int(sample['label']))
            batch_e1_type_labels.append(int(sample['e1_subtype_id']))
            batch_e2_type_labels.append(int(sample['e2_subtype_id']))
        batch_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_mask_inputs = tokenizer(
            batch_sen, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_inputs': batch_mask_inputs, 
            'batch_mask_idx': batch_mask_idx, 
            'batch_type_match_mask_idx': batch_type_match_mask_idx, 
            'batch_arg_match_mask_idx': batch_arg_match_mask_idx, 
            'batch_event_idx': batch_event_idx, 
            'batch_t1_mask_idx': batch_e1_type_mask_idx, 
            'batch_t2_mask_idx': batch_e2_type_mask_idx, 
            'label_word_id': [neg_id, pos_id], 
            'match_label_word_id': [match_id, mismatch_id], 
            'subtype_label_word_id': [event_type_ids[i] for i in range(len(EVENT_SUBTYPES) + 1)], 
            'labels': batch_labels, 
            'subtype_match_labels': batch_type_match_labels, 
            'arg_match_labels': batch_arg_match_labels, 
            'e1_subtype_labels': batch_e1_type_labels, 
            'e2_subtype_labels': batch_e2_type_labels
        }

    with_mask_input = with_mask if with_mask else args.with_mask
    if prompt_type.startswith('h') or prompt_type.startswith('s'): # base prompt
        select_collote_fn = collote_fn_with_mask if with_mask_input else collote_fn
    elif prompt_type.startswith('t'): # knowledge prompt
        select_collote_fn = collote_fn_subtype_with_mask if with_mask_input else collote_fn_subtype
    elif prompt_type.startswith('m'): # mix prompt
        select_collote_fn = collote_fn_subtype_match_with_mask if with_mask_input else collote_fn_subtype_match
    
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

    # arg_dict = get_pred_arguments('../../data/EventExtraction/omni_gold_test_pred_args.json')
    # for event_arg_dic in arg_dict.values():
    #     print(event_arg_dic)
    from transformers import AutoTokenizer
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 4
    args.max_seq_length = 512
    args.model_type = 'longformer'
    args.model_checkpoint = '../../PT_MODELS/allenai/longformer-large-4096'
    args.prompt_type = 'm_hta_hn'
    args.select_arg_strategy = 'no_filter'
    args.with_mask = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    base_sp_tokens = ['<e1_start>', '<e1_end>', '<e2_start>', '<e2_end>', '<l1>', '<l2>', '<l3>', '<l4>', '<l5>', '<l6>', '<l7>', '<l8>', '<l9>', '<l10>']
    match_sp_tokens = ['<match>', '<mismatch>']
    connect_tokens = ['<refer_to>', '<not_refer_to>']
    type_sp_tokens = [f'<st_{t_id}>' for t_id in range(len(EVENT_SUBTYPES) + 1)]
    special_tokens_dict = {
        'additional_special_tokens': base_sp_tokens + match_sp_tokens + connect_tokens + type_sp_tokens
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # train_data = KBPCoref(
    #     '../../data/train_filtered.json', '../../data/KnowledgeExtraction/simi_train_related_info_0.75.json', 
    #     prompt_type=args.select_arg_strategy, select_arg_strategy=args.select_arg_strategy, 
    #     model_type='longformer', tokenizer=tokenizer, max_length=512
    # )
    # print_data_statistic('../../data/train_filtered.json')
    # print(len(train_data))
    # labels = [train_data[s_idx]['label'] for s_idx in range(len(train_data))]
    # print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    # for i in range(5):
    #     print(train_data[i])

    train_small_data = KBPCorefTiny(
        '../../data/train_filtered.json', '../../data/train_filtered_with_cos.json', '../../data/KnowledgeExtraction/simi_files/simi_chatgpt_train_related_info_0.75.json', 
        neg_top_k=3, prompt_type=args.prompt_type, select_arg_strategy=args.select_arg_strategy, 
        model_type='longformer', tokenizer=tokenizer, max_length=512
    )
    print_data_statistic('../../data/train_filtered_with_cos.json')
    print(len(train_small_data))
    labels = [train_small_data[s_idx]['label'] for s_idx in range(len(train_small_data))]
    print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    for i in range(10):
        print(train_small_data[i])
    
    verbalizer = {
        'coref': {
            'token': '<refer_to>', 'id': tokenizer.convert_tokens_to_ids('<refer_to>'), 
            'description': 'refer to'
        } if 'c' in args.prompt_type else {
            'token': 'yes', 'id': tokenizer.convert_tokens_to_ids('yes')
        } if 'q' in args.prompt_type else {
            'token': 'same', 'id': tokenizer.convert_tokens_to_ids('same')
        } , 
        'non-coref': {
            'token': '<not_refer_to>', 'id': tokenizer.convert_tokens_to_ids('<not_refer_to>'), 
            'description': 'not refer to'
        } if 'c' in args.prompt_type else {
            'token': 'no', 'id': tokenizer.convert_tokens_to_ids('no')
        } if 'q' in args.prompt_type else {
            'token': 'different', 'id': tokenizer.convert_tokens_to_ids('different')
        }, 
        'match': {'token': '<match>', 'id': tokenizer.convert_tokens_to_ids('<match>')}, 
        'mismatch': {'token': '<mismatch>', 'id': tokenizer.convert_tokens_to_ids('<mismatch>')}
    }
    for subtype, s_id in subtype2id.items():
        verbalizer[subtype] = {'token': f'<st_{s_id}>', 'id': tokenizer.convert_tokens_to_ids(f'<st_{s_id}>')}
    # print(verbalizer)
    if args.prompt_type.startswith('h') or args.prompt_type.startswith('s'): # base prompt
        print('=' * 20, 'base prompt')
        train_dataloader = get_dataLoader(args, train_small_data, tokenizer, prompt_type=args.prompt_type, verbalizer=verbalizer, shuffle=True)
        batch_data = next(iter(train_dataloader))
        print('batch_inputs shape:', {k: v.shape for k, v in batch_data['batch_inputs'].items()})
        print('batch_inputs: ', batch_data['batch_inputs'])
        print('batch_mask_idx:', batch_data['batch_mask_idx'])
        print('batch_event_idx:', batch_data['batch_event_idx'])
        print('labels:', batch_data['labels'])
        print(tokenizer.decode(batch_data['batch_inputs']['input_ids'][0]))
        print('Testing dataloader...')
        batch_datas = iter(train_dataloader)
        for step in tqdm(range(len(train_dataloader))):
            next(batch_datas)
    elif args.prompt_type.startswith('t'): # knowledge prompt
        print('=' * 20, 'knowledge prompt')
        train_dataloader = get_dataLoader(args, train_small_data, tokenizer, prompt_type=args.prompt_type, verbalizer=verbalizer, shuffle=True)
        batch_data = next(iter(train_dataloader))
        print('batch_inputs shape:', {k: v.shape for k, v in batch_data['batch_inputs'].items()})
        print('batch_inputs: ', batch_data['batch_inputs'])
        print('batch_mask_idx:', batch_data['batch_mask_idx'])
        print('batch_event_idx:', batch_data['batch_event_idx'])
        print('batch_t1_mask_idx:', batch_data['batch_t1_mask_idx'])
        print('batch_t2_mask_idx:', batch_data['batch_t2_mask_idx'])
        print('labels:', batch_data['labels'])
        print('e1_subtype_labels:', batch_data['e1_subtype_labels'])
        print('e2_subtype_labels:', batch_data['e2_subtype_labels'])
        print(tokenizer.decode(batch_data['batch_inputs']['input_ids'][0]))
        print('Testing dataloader...')
        batch_datas = iter(train_dataloader)
        for step in tqdm(range(len(train_dataloader))):
            next(batch_datas)
    elif args.prompt_type.startswith('m'): # mix prompt
        print('=' * 20, 'mix prompt')
        train_dataloader = get_dataLoader(args, train_small_data, tokenizer, prompt_type=args.prompt_type, verbalizer=verbalizer, shuffle=True)
        batch_data = next(iter(train_dataloader))
        print('batch_inputs shape:', {k: v.shape for k, v in batch_data['batch_inputs'].items()})
        print('batch_inputs: ', batch_data['batch_inputs'])
        print('batch_mask_idx:', batch_data['batch_mask_idx'])
        print('batch_type_match_mask_idx', batch_data['batch_type_match_mask_idx'])
        print('batch_arg_match_mask_idx', batch_data['batch_arg_match_mask_idx'])
        print('batch_event_idx:', batch_data['batch_event_idx'])
        print('batch_t1_mask_idx:', batch_data['batch_t1_mask_idx'])
        print('batch_t2_mask_idx:', batch_data['batch_t2_mask_idx'])
        print('labels:', batch_data['labels'])
        print('subtype_match_labels:', batch_data['subtype_match_labels'])
        print('arg_match_labels:', batch_data['arg_match_labels'])
        print('e1_subtype_labels:', batch_data['e1_subtype_labels'])
        print('e2_subtype_labels:', batch_data['e2_subtype_labels'])
        print(tokenizer.decode(batch_data['batch_inputs']['input_ids'][0]))
        print('Testing dataloader...')
        batch_datas = iter(train_dataloader)
        for step in tqdm(range(len(train_dataloader))):
            next(batch_datas)
