import os
import json
import torch.cuda
import re 
import torch 
from collections import defaultdict
from OmniEvent.infer_module.io_format import Result, Event
from OmniEvent.model.model import get_model_cls
from OmniEvent.utils import check_web_and_convert_path
from transformers import (
    BertTokenizerFast,
    RobertaTokenizerFast,
    T5TokenizerFast,
    MT5TokenizerFast,
    BartTokenizerFast
)

split_word = ":"

def get_words(text, language):
    if language == "English":
        words = text.split()
    elif language == "Chinese":
        words = list(text)
    else:
        raise NotImplementedError
    return words

class EDProcessor():
    def __init__(self, tokenizer, max_seq_length=160):
        self.tokenizer = tokenizer 
        self.max_seq_length = max_seq_length

    def tokenize_per_instance(self, text, schema):
        if schema in ["<duee>", "<fewfc>", "<leven>"]:
            words = get_words(schema+text, "Chinese")
        else:
            words = get_words(schema+text, "English")
        input_context = self.tokenizer(words,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.max_seq_length,
                                       is_split_into_words=True)
        return dict(
            input_ids=torch.tensor(input_context["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(input_context["attention_mask"], dtype=torch.float32)
        )

    def tokenize(self, texts, schemas):
        batch = []
        for text, schema in zip(texts, schemas):
            batch.append(self.tokenize_per_instance(text, schema))
        # output batch features 
        output_batch = defaultdict(list)
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        # truncate
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key not in output_batch:
                continue
            output_batch[key] = output_batch[key][:, :input_length].cuda()
        return output_batch

class EAEProcessor():
    def __init__(self, tokenizer, max_seq_length=160):
        self.tokenizer = tokenizer 
        self.max_seq_length = max_seq_length

    def insert_marker(self, text, trigger_pos, whitespace=True):
        space = " " if whitespace else ""
        markered_text = ""
        tokens = text.split()
        char_pos = 0
        for i, token in enumerate(tokens):
            if char_pos == trigger_pos[0]:
                markered_text += "<event>" + space
            char_pos += len(token) + len(space)
            markered_text += token + space
            if char_pos == trigger_pos[1] + len(space):
                markered_text += "</event>" + space
        markered_text = markered_text.strip()
        return markered_text

    def tokenize_per_instance(self, text, trigger, schema):
        if schema in ["<duee>", "<fewfc>", "<leven>"]:
            language = "Chinese"
        else:
            language = "English"
        whitespace = True if language == "English" else False
        text = self.insert_marker(text, trigger["offset"], whitespace)
        words = get_words(text, language)
        input_context = self.tokenizer(words,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=self.max_seq_length,
                                       is_split_into_words=True)
        return dict(
            input_ids=torch.tensor(input_context["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(input_context["attention_mask"], dtype=torch.float32)
        )

    def tokenize(self, instances):
        batch = []
        for i, instance in enumerate(instances):
            for trigger in instance["triggers"]:
                batch.append(self.tokenize_per_instance(instance["text"], trigger, instance["schema"]))
        # output batch features 
        output_batch = defaultdict(list)
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        # truncate
        input_length = int(output_batch["attention_mask"].sum(-1).max())
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key not in output_batch:
                continue
            output_batch[key] = output_batch[key][:, :input_length].cuda()
        return output_batch

def find_position(mention, text):
    char_start = text.index(mention)
    char_end = char_start + len(mention)
    return [char_start, char_end]

def get_ed_result(texts, triggers):
    results = []
    for i, text in enumerate(texts):
        triggers_in_text = [trigger for trigger in triggers if trigger[0]==i]
        result = Result()
        events = []
        for trigger in triggers_in_text:
            type = trigger[1]
            mention = trigger[2]
            if mention not in text:
                continue
            offset = find_position(mention, text)
            event = {
                "type": type,
                "trigger": mention,
                "offset": offset
            }
            events.append(event)
        results.append({
            "text": text,
            "events": events 
        })
    return results

def get_eae_result(instances, arguments):
    results = []
    for i, instance in enumerate(instances):
        result = Result()
        events = []
        for trigger, argus_in_trigger in zip(instance["triggers"], arguments):
            event = Event()
            event_arguments = []
            for argu in argus_in_trigger:
                role = argu[1]
                mention = argu[2]
                if mention not in instance["text"]:
                    continue
                offset = find_position(mention, instance["text"])
                argument = {
                    "mention": mention,
                    "offset": offset,
                    "role": role
                }
                event_arguments.append(argument)
            events.append({
                "type": trigger["type"] if "type" in trigger else "NA",
                "offset": trigger["offset"], 
                "trigger": trigger["mention"],
                "arguments": event_arguments
            })
        results.append({
            "text": instance["text"],
            "events": events
        })
    return results

def prepare_for_eae_from_input(texts, all_triggers, schemas):
    instances = []
    for text, triggers, schema in zip(texts, all_triggers, schemas):
        instance = {
            "text": text,
            "schema": schema,
            "triggers": []
        }
        for trigger in triggers:
            instance["triggers"].append({
                "mention": trigger[0],
                "offset": [trigger[1], trigger[2]]
            })
        instances.append(instance)
    return instances

def prepare_for_eae_from_pred(texts, triggers, schemas):
    instances = []
    for i, text in enumerate(texts):
        triggers_in_text = [trigger for trigger in triggers if trigger[0]==i]
        instance = {
            "text": text,
            "schema": schemas[i],
            "triggers": []
        }
        for trigger in triggers_in_text:
            type = trigger[1]
            mention = trigger[2]
            if mention not in text:
                continue
            offset = find_position(mention, text)
            instance["triggers"].append({
                "type": type, 
                "mention": mention,
                "offset": offset
            })
        instances.append(instance)
    return instances 

def extract_argument(raw_text, instance_id, template=re.compile(r"<|>")):
    arguments = []
    for span in template.split(raw_text):
        if span.strip() == "":
            continue
        words = span.strip().split(split_word)
        if len(words) != 2:
            continue
        role = words[0].strip().replace(" ", "")
        value = words[1].strip().replace(" ", "")
        if role != "" and value != "":
            arguments.append((instance_id, role, value))
    arguments = list(set(arguments))
    return arguments

def generate(model, tokenizer, inputs):
    gen_kwargs = {
        "max_length": 128,
        "num_beams": 4,
        "synced_gpus": False,
        "prefix_allowed_tokens_fn": None 
    }

    if "attention_mask" in inputs:
        gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

    generation_inputs = inputs["input_ids"]

    generated_tokens = model.generate(
        generation_inputs,
        **gen_kwargs,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

def do_event_detection(model, tokenizer, texts, schemas):
    data_processor = EDProcessor(tokenizer, max_seq_length=512)
    inputs = data_processor.tokenize(texts, schemas)
    decoded_preds = generate(model, tokenizer, inputs)
    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    pred_triggers = []
    for i, pred in enumerate(decoded_preds):
        pred = clean_str(pred)
        pred_triggers.extend(extract_argument(pred, i))
    return pred_triggers

def do_event_argument_extraction(model, tokenizer, instances):
    data_processor = EAEProcessor(tokenizer, max_seq_length=512)
    inputs = data_processor.tokenize(instances)
    decoded_preds = generate(model, tokenizer, inputs)
    def clean_str(x_str):
        for to_remove_token in [tokenizer.eos_token, tokenizer.pad_token]:
            x_str = x_str.replace(to_remove_token, '')
        return x_str.strip()
    pred_triggers = []
    for i, pred in enumerate(decoded_preds):
        pred = clean_str(pred)
        pred_triggers.append(extract_argument(pred, i))
    return pred_triggers

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

TOKENIZER_NAME_TO_CLS = {
    "BertTokenizer": BertTokenizerFast,
    "RobertaTokenizer": RobertaTokenizerFast,
    "T5Tokenizer": T5TokenizerFast,
    "MT5Tokenizer": MT5TokenizerFast,
    "BartTokenizer": BartTokenizerFast
}

def get_tokenizer(path):
    path = check_web_and_convert_path(path, "tokenizer")
    tokenizer_config = json.load(open(os.path.join(path, "tokenizer_config.json")))
    tokenizer_cls = TOKENIZER_NAME_TO_CLS[tokenizer_config["tokenizer_class"]]
    tokenizer = tokenizer_cls.from_pretrained(path)
    return tokenizer

def get_model(model_args, path):
    # path = check_web_and_convert_path(path, "model")
    model = get_model_cls(model_args).from_pretrained(path)
    return model

def get_pretrained(path, device):
    # config 
    # parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.from_pretrained(model_name_or_path)
    # model
    model_args = AttrDict({
        "paradigm": "seq2seq",
        "model_type": "mt5"
    })
    model = get_model(model_args, path)
    model = model.to(device)
    # model.cuda()
    # tokenizer 
    tokenizer = get_tokenizer(path)

    return model, tokenizer

if __name__ == "__main__":

    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    CATEGORIES = [
        'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
        'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
        'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
    ]

    def event_detection(data_file, save_event_file, create_segment:bool=False):
        tokenizer = AutoTokenizer.from_pretrained('../../../PT_MODELS/google/mt5-base/')
        ed_model, ed_tokenizer = get_pretrained("../../../PT_MODELS/OmniEvent/s2s-mt5-ed/", 'cuda')
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                sentences = sample['sentences']
                sentences_lengths = [len(tokenizer(sent['text']).tokens()) for sent in sentences]
                pred_events = []
                for sent_idx in range(len(sentences)):
                    if create_segment:
                        new_segment = ''
                        segment_mapping = []
                        total_length = 0
                        if sent_idx > 0 and sentences_lengths[sent_idx - 1] + sentences_lengths[sent_idx] < 512:
                            segment_mapping.append({'new_start': 0, 'new_end': len(sentences[sent_idx - 1]['text']), 'start': sentences[sent_idx - 1]['start']})
                            new_segment = sentences[sent_idx - 1]['text'] + ' '
                            total_length += sentences_lengths[sent_idx - 1]
                        segment_mapping.append({'new_start': len(new_segment), 'new_end': len(new_segment) + len(sentences[sent_idx]['text']), 'start': sentences[sent_idx]['start']})
                        new_segment += (sentences[sent_idx]['text'] + ' ')
                        total_length += sentences_lengths[sent_idx]
                        if sent_idx < len(sentences) - 1 and total_length + sentences_lengths[sent_idx + 1] < 512:
                            segment_mapping.append({'new_start': len(new_segment), 'new_end': len(new_segment) + len(sentences[sent_idx + 1]['text']), 'start': sentences[sent_idx + 1]['start']})
                            new_segment += sentences[sent_idx + 1]['text']
                        ed_events = do_event_detection(ed_model, ed_tokenizer, [new_segment], ['kbp'])
                        results = get_ed_result([new_segment], ed_events)[0]["events"]
                        for event in results:
                            if event['type'] not in CATEGORIES:
                                continue
                            sent_start, start = -1, -1
                            for mapping in segment_mapping:
                                if event['offset'][0] >= mapping['new_start'] and event['offset'][1] <= mapping['new_end']:
                                    sent_start = event['offset'][0] - mapping['new_start']
                                    start = mapping['start']
                                    break
                            assert start != -1
                            pred_events.append({
                                'start': sent_start + start, 
                                'end': sent_start + start + len(event['trigger']) - 1, 
                                'trigger': event['trigger'], 
                                'subtype': event['type']
                            })
                        new_pred_events = []
                        for event in pred_events:
                            should_add = True
                            for e in new_pred_events:
                                if e['start'] <= event['start'] <= e['end'] or e['start'] <= event['end'] <= e['end']:
                                    should_add = False
                                    break
                            if should_add:
                                new_pred_events.append(event)
                        pred_events = sorted(new_pred_events, key=lambda x:x['start'])
                    else:
                        ed_events = do_event_detection(ed_model, ed_tokenizer, [sentences[sent_idx]['text']], ['kbp'])
                        results = get_ed_result([sentences[sent_idx]['text']], ed_events)[0]["events"]
                        for event in results:
                            if event['type'] not in CATEGORIES:
                                continue
                            pred_events.append({
                                'start': event['offset'][0] + sentences[sent_idx]['start'], 
                                'end': event['offset'][1] + sentences[sent_idx]['start'] - 1, 
                                'trigger': event['trigger'], 
                                'subtype': event['type']
                            })
                for event in pred_events:
                    assert sample['document'][event['start']:event['end']+1] == event['trigger']
                print(pred_events)
                Data.append({
                    'doc_id': sample['doc_id'], 
                    'document': sample['document'], 
                    'pred_label': pred_events, 
                    'true_label': [{
                        'start': event['start'], 
                        'end': event['start'] + len(event['trigger']) - 1, 
                        'trigger': event['trigger'], 
                        'subtype': event['subtype']
                    } for event in sample['events'] if event['subtype'] in CATEGORIES]
                })
        Data.sort(key=lambda x:x['doc_id'])
        with open(save_event_file, 'wt', encoding='utf-8') as f:
            for doc in Data:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    # event_detection('../test_filtered.json', 'omni_test_pred_events.json')

    def evalute_ed(data_file):

        def event_in_list(event, event_list):
            for e in event_list:
                if e['start'] == event['start'] and e['end'] == event['end'] and e['subtype'] == event['subtype']:
                    return True
            return False

        subtype_metrics = {subtype: {'TP': 0, 'FP': 0, 'FN': 0} for subtype in CATEGORIES}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                labels, preds = sample['true_label'], sample['pred_label']
                for e_y in labels:
                    if event_in_list(e_y, preds):
                        subtype_metrics[e_y['subtype']]['TP'] += 1
                    else:
                        subtype_metrics[e_y['subtype']]['FN'] += 1
                for e_p in preds:
                    if not event_in_list(e_p, labels):
                        subtype_metrics[e_p['subtype']]['FP'] += 1
        tp_avg, fp_avg, fn_avg = 0., 0., 0.
        for metrics in subtype_metrics.values():
            tp_avg += metrics['TP']
            fp_avg += metrics['FP']
            fn_avg += metrics['FN']
        tp_avg /= len(subtype_metrics)
        fp_avg /= len(subtype_metrics)
        fn_avg /= len(subtype_metrics)
        micro_p, micro_r = tp_avg / (tp_avg + fp_avg), tp_avg / (tp_avg + fn_avg)
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r)
        print(micro_f1)
    
    # evalute_ed('../omni_test_pred_events.json')

    def event_argument_extraction(event_file, save_argument_file):
        eae_model, eae_tokenizer = get_pretrained("../../../PT_MODELS/OmniEvent/s2s-mt5-eae/", 'cuda')
        Data = []
        with open(event_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                sentences = sample['sentences']
                pred_event_args = []
                for sent in sentences:
                    sent['event'] = []
                for event in sample['events']:
                    sentences[event['sent_idx']]['event'].append((event['trigger'], event['sent_start'], event['sent_start'] + len(event['trigger'])))
                for sent in sentences:
                    if not sent['event']:
                        continue
                    instances = prepare_for_eae_from_input([sent['text']], [sent['event']], ['kbp'])
                    arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances)
                    results = get_eae_result(instances, arguments)[0]["events"]
                    for event in results:
                        pred_event_args.append({
                            'start': event['offset'][0] + sent['start'], 
                            'end': event['offset'][1] + sent['start'] - 1, 
                            'trigger': event['trigger'], 
                            'subtype': event['type'], 
                            'arguments': [
                                {
                                    'start': arg['offset'][0] + sent['start'], 
                                    'end': arg['offset'][1] + sent['start'] - 1, 
                                    'mention': arg['mention'], 
                                    'role': arg['role']
                                } for arg in event['arguments']
                            ]
                        })
                for event in pred_event_args:
                    assert sample['document'][event['start']:event['end']+1] == event['trigger']
                    for arg in event['arguments']:
                        assert sample['document'][arg['start']:arg['end']+1] == arg['mention']
                print(pred_event_args)
                Data.append({
                    'doc_id': sample['doc_id'], 
                    'document': sample['document'], 
                    'event_args': pred_event_args
                })
        Data.sort(key=lambda x:x['doc_id'])
        with open(save_argument_file, 'wt', encoding='utf-8') as f:
            for doc in Data:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    def event_argument_extraction_for_testfile(gold_test_file, pred_event_file, save_argument_file):
        
        def get_event_sent_idx(trigger_start, trigger_end, sentences):
            for idx, sent in enumerate(sentences):
                if trigger_start >= sent['start'] and trigger_end < sent['start'] + len(sent['text']):
                    return idx, trigger_start - sent['start']
            raise ValueError('can\'t find sentence idx for event', trigger_start, trigger_end)
        
        eae_model, eae_tokenizer = get_pretrained("../../../PT_MODELS/OmniEvent/s2s-mt5-eae/", 'cuda')
        doc_sent_dict = {}
        with open(gold_test_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                doc_sent_dict[sample['doc_id']] = sample['sentences']
        Data = []
        with open(pred_event_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                sentences = doc_sent_dict[sample['doc_id']]
                pred_event_args = []
                for sent in sentences:
                    sent['event'] = []
                for event in sample['pred_label']:
                    sent_idx, sent_start = get_event_sent_idx(event['start'], event['start'] + len(event['trigger']) - 1, sentences)
                    sentences[sent_idx]['event'].append((event['trigger'], sent_start, sent_start + len(event['trigger'])))
                for sent in sentences:
                    if not sent['event']:
                        continue
                    instances = prepare_for_eae_from_input([sent['text']], [sent['event']], ['kbp'])
                    arguments = do_event_argument_extraction(eae_model, eae_tokenizer, instances)
                    results = get_eae_result(instances, arguments)[0]["events"]
                    for event in results:
                        pred_event_args.append({
                            'start': event['offset'][0] + sent['start'], 
                            'end': event['offset'][1] + sent['start'] - 1, 
                            'trigger': event['trigger'], 
                            'subtype': event['type'], 
                            'arguments': [
                                {
                                    'start': arg['offset'][0] + sent['start'], 
                                    'end': arg['offset'][1] + sent['start'] - 1, 
                                    'mention': arg['mention'], 
                                    'role': arg['role']
                                } for arg in event['arguments']
                            ]
                        })
                for event in pred_event_args:
                    assert sample['document'][event['start']:event['end']+1] == event['trigger']
                    for arg in event['arguments']:
                        assert sample['document'][arg['start']:arg['end']+1] == arg['mention']
                print(pred_event_args)
                Data.append({
                    'doc_id': sample['doc_id'], 
                    'document': sample['document'], 
                    'event_args': pred_event_args
                })
        Data.sort(key=lambda x:x['doc_id'])
        with open(save_argument_file, 'wt', encoding='utf-8') as f:
            for doc in Data:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    # event_argument_extraction('../train_filtered.json', 'omni_train_pred_args.json')
    # event_argument_extraction('../dev_filtered.json', 'omni_dev_pred_args.json')
    # event_argument_extraction_for_testfile('../test_filtered.json', '../epoch_3_test_pred_events.json', 'omni_epoch_3_test_pred_args.json')

    def role_statistics(argument_file):
        role_dict = {}
        event_num, arg_num, has_arg = 0, 0, 0
        with open(argument_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                event_args = sample['event_args']
                for event in event_args:
                    event_num += 1
                    if event['arguments']: 
                        has_arg += 1
                    for arg in event['arguments']:
                        role = arg['role'].lower()
                        if role in role_dict:
                            role_dict[role]['num'] += 1
                            role_dict[role]['mentions'].append(arg['mention'])
                        else:
                            role_dict[role] = {'num': 1, 'mentions': [arg['mention']]}
                        arg_num += 1
        print(event_num, arg_num, has_arg)
        from collections import Counter
        for role, info in role_dict.items():
            counter = Counter(info['mentions'])
            print(role, '\tnum:', info['num'])
            print(counter.most_common()[:5])
    
    role_statistics('./omni_train_pred_args.json')
