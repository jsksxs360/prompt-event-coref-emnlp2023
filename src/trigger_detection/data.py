from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import torch

CATEGORIES = [
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]

class KBPTrigger(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                tags = [{
                    'start': event['start'], 
                    'end': event['start'] + len(event['trigger']) - 1, 
                    'trigger': event['trigger'], 
                    'subtype': event['subtype']
                } for event in sample['events'] if event['subtype'] in CATEGORIES]
                Data.append({
                    'id': sample['doc_id'], 
                    'document': sample['document'], 
                    'tags': tags
                })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    
    def collote_fn(batch_samples):
        batch_sentence, batch_tags  = [], []
        for sample in batch_samples:
            batch_sentence.append(sample['document'])
            batch_tags.append(sample['tags'])
        batch_inputs = tokenizer(
            batch_sentence, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
        for s_idx, sentence in enumerate(batch_sentence):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            for tag in batch_tags[s_idx]:
                assert sentence[tag['start']:tag['end']+1] == tag['trigger']
                token_start = encoding.char_to_token(tag['start'])
                token_end = encoding.char_to_token(tag['end'])
                if not token_start or not token_end:
                    continue
                batch_label[s_idx][token_start] = args.label2id[f"B-{tag['subtype']}"]
                batch_label[s_idx][token_start + 1:token_end + 1] = args.label2id[f"I-{tag['subtype']}"]
        return {
            'batch_inputs': batch_inputs, 
            'labels': batch_label
        }
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn)
