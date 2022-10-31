from torch.utils.data import Dataset, DataLoader
import json

class KBPCoref(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        return None

    def load_data(self, data_file):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                events = [
                    {
                        'event_id': e['event_id'], 
                        'char_start': e['start'], 
                        'char_end': e['start'] + len(e['trigger']) - 1, 
                        'trigger': e['trigger'], 
                        'cluster_id': self._get_event_cluster_id(e['event_id'], clusters)
                    }
                    for e in sample['events']
                ]
                Data.append({
                    'id': sample['doc_id'], 
                    'document': sample['document'], 
                    'events': events
                })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):

    def collote_fn(batch_samples):
        batch_sentences, batch_events  = [], []
        for sample in batch_samples:
            batch_sentences.append(sample['document'])
            batch_events.append(sample['events'])
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_filtered_events = []
        batch_filtered_event_cluster_id = []
        for sentence, events in zip(batch_sentences, batch_events):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_event_cluster_id = []
            for e in events:
                token_start = encoding.char_to_token(e['char_start'])
                if not token_start:
                    token_start = encoding.char_to_token(e['char_start'] + 1)
                token_end = encoding.char_to_token(e['char_end'])
                if not token_start or not token_end:
                    continue
                filtered_events.append([token_start, token_end])
                filtered_event_cluster_id.append(e['cluster_id'])
            batch_filtered_events.append(filtered_events)
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)
        return {
            'batch_inputs': batch_inputs, 
            'batch_events': batch_filtered_events, 
            'batch_event_cluster_ids': batch_filtered_event_cluster_id
        }
    
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=collote_fn
    )