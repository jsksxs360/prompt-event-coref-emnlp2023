from torch.utils.data import Dataset
import json
from tqdm.auto import tqdm
from utils import create_prompt

PROMPT_TYPE = [
    'hn', 'hm', 'hq', # base prompts 
    'sn', 'sm', 'sq', # (hard/soft normal/middle/question)
    't_hn', 'ta_hn', 't_hm', 'ta_hm', 't_hq', 'ta_hq', # knowledge enhanced prompts 
    't_sn', 'ta_sn', 't_sm', 'ta_sm', 't_sq', 'ta_sq', # (subtype/subtype-argument)
    'm_hs_hn', 'm_hs_hm', 'm_hs_hq', 'm_hsa_hn', 'm_hsa_hm', 'm_hsa_hq', # mix prompts
    'm_ss_hn', 'm_ss_hm', 'm_ss_hq', 'm_ssa_hn', 'm_ssa_hm', 'm_ssa_hq'  # (hard/soft subtype/argument/subtype-argument)
]

EVENT_SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
id2subtype = {idx: c for idx, c in enumerate(EVENT_SUBTYPES, start=1)}
subtype2id = {v: k for k, v in id2subtype.items()}

def get_pred_arguments(arg_file:str) -> dict:
    '''
    # Returns: 
        - argument dictionary: {doc_id: {event_id: [{"global_offset": 798, "mention": "We", "role": "participant"}]}}
    '''
    participant_roles = set(['defendant', 'entity', 'person', 'position', 'agent', 'attacker', 
                             'giver', 'victim', 'audience', 'recipient', 'target', 'seller', 
                             'beneficiary', 'plaintiff', 'adjudicator', 'org', 'prosecutor'])
    place_roles = set(['place', 'destination', 'origin'])
    arg_dict = {}
    with open(arg_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            arg_dict[sample['doc_id']] = {
                event['start']: [
                    {
                        'global_offset': arg['start'], 
                        'mention': arg['mention'], 
                        'role': 'participant' if arg['role'].lower() in participant_roles else 'place'
                    } for arg in event['arguments'] if arg['role'].lower() in participant_roles | place_roles
                ] 
                for event in sample['event_args']
            }
    return arg_dict

def get_event_cluster_id(event_id:str, clusters:list) -> str:
    for cluster in clusters:
        if event_id in cluster['events']:
            return cluster['hopper_id']
    raise ValueError(f'Unknown event_id: {event_id}')

class KBPCoref(Dataset):
    def __init__(self, data_file:str, arg_file:str, prompt_type:str, model_type:str, tokenizer, max_length:int):
        assert prompt_type in PROMPT_TYPE and model_type in ['bert', 'roberta', 'longformer']
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.arg_dict = get_pred_arguments(arg_file)
        self.data = self.load_data(data_file, prompt_type)
    
    def load_data(self, data_file, prompt_type:str):
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
                        event_1_args = self.arg_dict[sample['doc_id']][event_1['start']]
                        event_2_args = self.arg_dict[sample['doc_id']][event_2['start']]
                        prompt_data = create_prompt(
                            event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], event_1_args, 
                            event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], event_2_args, 
                            sentences, sentences_lengths, 
                            prompt_type, self.model_type, self.tokenizer, self.max_length
                        )
                        Data.append({
                            'id': sample['doc_id'], 
                            'prompt': prompt_data['prompt'], 
                            'mask_offset': prompt_data['mask_offset'], 
                            'type_match_mask_offset': prompt_data['type_match_mask_offset'], 
                            'arg_match_mask_offset': prompt_data['arg_match_mask_offset'], 
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    special_tokens_dict = {'additional_special_tokens': ['<e1_start>', '<e1_end>', '<e2_start>', '<e2_end>', '<l1>', '<l2>', '<l3>', '<l4>', '<l5>', '<l6>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    train_data = KBPCoref(
        '../../data/train_filtered.json', '../../data/EventExtraction/omni_train_pred_args.json', 
        prompt_type='hn', model_type='longformer', tokenizer=tokenizer, max_length=512
    )
    print_data_statistic('../../data/train_filtered.json')
    print(len(train_data))
    labels = [train_data[s_idx]['label'] for s_idx in range(len(train_data))]
    print('Coref:', labels.count(1), 'non-Coref:', labels.count(0))
    for i in range(5):
        print(train_data[i])
