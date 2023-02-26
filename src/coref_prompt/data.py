from torch.utils.data import Dataset
import json

PROMPT_TYPE = [
    'hn', 'hm', 'hq', # base prompts 
    'sn', 'sm', 'sq', # (hard/soft normal/middle/question)
    's_hn', 'a_hn', 'sa_hn', 's_hm', 'a_hm', 'sa_hm', 's_hq', 'a_hq', 'sa_hq', # knowledge enhanced prompts 
    's_sn', 'a_sn', 'sa_sn', 's_sm', 'a_sm', 'sa_sm', 's_sq', 'a_sq', 'sa_sq', # (subtype/argument/subtype-argument)
    'm_hs', 'm_ha', 'm_hsa', # mix prompts
    'm_ss', 'm_sa', 'm_ssa'  # (hard/soft subtype/argument/subtype-argument)
]

EVENT_SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
id2subtype = {idx: c for idx, c in enumerate(EVENT_SUBTYPES, start=1)}
subtype2id = {v: k for k, v in id2subtype.items()}

class KBPCoref(Dataset):
    def __init__(self, data_file:str, arg_file:str, prompt_type:str, model_type:str, tokenizer, max_length:int):
        assert prompt_type in PROMPT_TYPE and model_type in ['bert', 'roberta', 'longformer']
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.kbp_entity_dict = get_KBP_entities(entity_file)
        self.data = self.load_data(data_file, prompt_type)
        
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        raise ValueError(f'Unknown event_id: {event_id}')

    def load_data(self, data_file, prompt_type:str):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                sentences_lengths = [len(self.tokenizer(sent['text']).tokens()) for sent in sentences]
                events = sample['events']
                # create event pairs
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1, event_2 = events[i], events[j]
                        event_1_cluster_id = self._get_event_cluster_id(event_1['event_id'], clusters)
                        event_2_cluster_id = self._get_event_cluster_id(event_2['event_id'], clusters)
                        
                        
                        
                        new_event_sent = create_new_sent(
                            event_1['sent_idx'], event_1['sent_start'], event_1['trigger'], 
                            event_2['sent_idx'], event_2['sent_start'], event_2['trigger'], 
                            sentences, sentences_lengths, sentences_entities, 
                            special_token_dict, context_k, max_length, self.tokenizer
                        )
                        Data.append({
                            'id': sample['doc_id'], 
                            'sent': new_event_sent['new_sent'], 
                            'e1_offset': event_1['start'], # event1
                            'e1_trigger': new_event_sent['e1_trigger'], 
                            'e1_subtype': subtype2id.get(event_1['subtype'], 0), # 0 - 'other'
                            'e1_subtype_str': event_1['subtype'] if event_1['subtype'] in SUBTYPES else 'normal', 
                            'e1_start': new_event_sent['e1_sent_start'], 
                            'e1s_start': new_event_sent['e1s_sent_start'], 
                            'e1e_start': new_event_sent['e1e_sent_start'], 
                            'e1_entities': new_event_sent['e1_entities'], 
                            'e2_offset': event_2['start'], # event2
                            'e2_trigger': new_event_sent['e2_trigger'], 
                            'e2_subtype': subtype2id.get(event_2['subtype'], 0), # 0 - 'other'
                            'e2_subtype_str': event_2['subtype'] if event_2['subtype'] in SUBTYPES else 'normal', 
                            'e2_start': new_event_sent['e2_sent_start'], 
                            'e2s_start': new_event_sent['e2s_sent_start'], 
                            'e2e_start': new_event_sent['e2e_sent_start'], 
                            'e2_entities': new_event_sent['e2_entities'], 
                            'label': 1 if event_1_cluster_id == event_2_cluster_id else 0
                        })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
