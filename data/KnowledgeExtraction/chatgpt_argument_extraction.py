import openai
from tqdm.auto import tqdm
import json
import os
import time

openai.api_key = "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"
START_TAG, END_TAG = '[EVENT]', '[/EVENT]'
WINDOWS = 5

def get_argument_info(pretty_event_mention, trigger, start_tag, end_tag):
    time.sleep(1)
    event = f'{start_tag} {trigger} {end_tag}'
    assert event in pretty_event_mention
    prefix = f'In the following document, please focus on the event {event}: '
    instruction = (
        f'Please extract the participants and locations of the event {event} '
        'in the above document. Respond in the form of a list:'
    )
    prompt = prefix + pretty_event_mention + '\n' + instruction
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0
    )
    res = response['choices'][0]['message']['content'].strip()
    res_items = res.split('\n')
    flag, participants, locations, unknow = '', [], [], []
    for line in res_items:
        line = line[:line.find('(')].strip() if line.find('(') != -1 else line.strip()
        if not line or 'none' in line.lower() or \
            'not' in line.lower() or 'event:' in line.lower() or 'events:' in line.lower():
            continue
        if not line.startswith('-'):
            if 'participant' in line.lower():
                flag = 'part'
            elif 'location' in line.lower():
                flag = 'loc'
            else:
                flag = 'unk'
        else:
            if line.startswith('- Participant:') or line.startswith('- Participants:'):
                flag = 'part'
                p = line[line.find(':')+1:].strip()
                if p:
                    participants.append(p)
            elif line.startswith('- Location:') or line.startswith('- Locations:'):
                flag = 'loc'
                l = line[line.find(':')+1:].strip()
                if l:
                    locations.append(l)
            else:
                (participants if flag == 'part' else locations if flag == 'loc' else unknow).append(
                    line[1:].strip()
                )
    if unknow:
        print('[UNKNOW]', unknow)
        print(pretty_event_mention, trigger)
    return {
        'text': res, 
        'participants': participants, 
        'locations': locations, 
        'unknow': unknow
    }

def pretty_event_mention(
    sentences, sent_idx, sent_offset, trigger, 
    start_tag='[EVENT]', end_tag='[/EVENT]', windows=5
    ):
    before_sentence, after_sentence = '', ''
    for i in range(1,1+windows):
        if sent_idx - i >= 0:
            before_sentence = sentences[sent_idx - i]['text'] + ' ' + before_sentence
        if sent_idx + i < len(sentences):
            after_sentence += ' ' + sentences[sent_idx + i]['text']
    sentence = sentences[sent_idx]['text']
    assert sentence[sent_offset:sent_offset + len(trigger)] == trigger
    mention = "{}{}{} {} {}{}{}".format(
        before_sentence, 
        sentence[:sent_offset], 
        start_tag, 
        trigger, 
        end_tag, 
        sentence[sent_offset + len(trigger):], 
        after_sentence
    )
    return mention

def event_argument_extraction(event_file, save_argument_file):
    finish_doc_ids = set()
    if os.path.exists(save_argument_file):
        with open(save_argument_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                finish_doc_ids.add(sample['doc_id'])
    with open(event_file, 'rt', encoding='utf-8') as f_in, \
         open(save_argument_file, 'at', encoding='utf-8') as f_out:
        for line in tqdm(f_in.readlines()):
            sample = json.loads(line.strip())
            if sample['doc_id'] in finish_doc_ids:
                continue
            sentences = sample['sentences']
            pred_event_args = []
            for event in sample['events']:
                em = pretty_event_mention(
                    sentences, event['sent_idx'], event['sent_start'], event['trigger'], 
                    start_tag=START_TAG, end_tag=END_TAG, windows=WINDOWS
                )
                pred_res = get_argument_info(em, event['trigger'], start_tag=START_TAG, end_tag=END_TAG)
                pred_event_args.append({
                    'start': event['start'], 
                    'end': event['start'] + len(event['trigger']) - 1, 
                    'trigger': event['trigger'], 
                    'subtype': event['subtype'], 
                    'participants': pred_res['participants'], 
                    'locations': pred_res['locations'], 
                    'unknow': pred_res['unknow'], 
                    'chatgpt_res': pred_res['text']
                })
            print(pred_event_args)
            doc = {
                'doc_id': sample['doc_id'], 
                'document': sample['document'], 
                'event_args': pred_event_args
            }
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return True

def event_argument_extraction_for_testfile(gold_test_file, pred_event_file, save_argument_file):

    def get_event_sent_idx(trigger_start, trigger_end, sentences):
        for idx, sent in enumerate(sentences):
            if trigger_start >= sent['start'] and trigger_end < sent['start'] + len(sent['text']):
                return idx, trigger_start - sent['start']
        raise ValueError('can\'t find sentence idx for event', trigger_start, trigger_end)
    
    doc_sent_dict = {}
    with open(gold_test_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line.strip())
            doc_sent_dict[sample['doc_id']] = sample['sentences']
    finish_doc_ids = set()
    if os.path.exists(save_argument_file):
        with open(save_argument_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                sample = json.loads(line.strip())
                finish_doc_ids.add(sample['doc_id'])
    with open(pred_event_file, 'rt', encoding='utf-8') as f_in, \
         open(save_argument_file, 'at', encoding='utf-8') as f_out:
        for line in tqdm(f_in.readlines()):
            sample = json.loads(line.strip())
            if sample['doc_id'] in finish_doc_ids:
                continue
            sentences = doc_sent_dict[sample['doc_id']]
            pred_event_args = []
            for event in sample['pred_label']:
                sent_idx, sent_start = get_event_sent_idx(event['start'], event['start'] + len(event['trigger']) - 1, sentences)
                em = pretty_event_mention(
                    sentences, sent_idx, sent_start, event['trigger'], 
                    start_tag=START_TAG, end_tag=END_TAG, windows=WINDOWS
                )
                pred_res = get_argument_info(em, event['trigger'], start_tag=START_TAG, end_tag=END_TAG)
                pred_event_args.append({
                    'start': event['start'], 
                    'end': event['start'] + len(event['trigger']) - 1, 
                    'trigger': event['trigger'], 
                    'subtype': event['subtype'], 
                    'participants': pred_res['participants'], 
                    'locations': pred_res['locations'], 
                    'unknow': pred_res['unknow'], 
                    'chatgpt_res': pred_res['text']
                })
            print(pred_event_args)
            doc = {
                'doc_id': sample['doc_id'], 
                'document': sample['document'], 
                'event_args': pred_event_args
            }
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return True

# while True:
#     try:
#         status = event_argument_extraction('../train_filtered.json', 'chatgpt_train_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)
# while True:
#     try:
#         status = event_argument_extraction('../dev_filtered.json', 'chatgpt_dev_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)
# while True:
#     try:
#         status = event_argument_extraction_for_testfile('../test_filtered.json', '../epoch_3_test_pred_events.json', 'chatgpt_epoch_3_test_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)

def fix_parsing_problems(chatgpt_result_file, start_tag, end_tag):

    def find_new_args(unknow_list):
        new_part, new_loc, new_arg = [], [], []
        for item in unknow_list:
            if start_tag in item or end_tag in item:
                continue
            if item.lower().startswith('participant') or item.lower().startswith('[participant'):
                p = item[item.find(':')+1:].strip()
                if p:
                    new_part.append(p)
            elif item.lower().startswith('location') or item.lower().startswith('[location'):
                l = item[item.find(':')+1:].strip()
                if l:
                    new_loc.append(l)
            else:
                new_arg.append(item)
        return new_part, new_loc, new_arg
    
    def filter_args(arg_list):
        arg_list = filter(
            lambda x: '[' not in x and ']' not in x and '@' not in x and\
                'no' not in x.lower() and 'not' not in x.lower() and \
                'none' not in x.lower() and 'unidentified' not in x.lower() and\
                len(x.split()) <= 10, 
            arg_list
        )
        res = []
        for arg in arg_list:
            if ':' in arg:
                arg = arg[:arg.find(':')].strip()
            if ',' in arg and len(arg.split()) > 5:
                arg = arg[:arg.find(',')].strip()
            if 'who' in arg and len(arg.split()) > 5:
                arg = arg[:arg.find('who')].strip()
            if arg:
                res.append(arg)
        return res

    Data = []
    with open(chatgpt_result_file, 'rt') as f_in:
        for line in f_in.readlines():
            Data.append(json.loads(line.strip()))
    for data in Data:
        event_args = []
        for event in data['event_args']:
            if event['unknow']:
                new_part, new_loc, new_arg = find_new_args(event['unknow'])
                event_args.append({
                    'start': event['start'], 
                    'end': event['end'], 
                    'trigger': event['trigger'], 
                    'subtype': event['subtype'], 
                    'participants': filter_args(event['participants'] + new_part), 
                    'locations': filter_args(event['locations'] + new_loc), 
                    'unknow': filter_args(new_arg), 
                    'chatgpt_res': event['chatgpt_res']
                })
            else:
                event_args.append({
                    'start': event['start'], 
                    'end': event['end'], 
                    'trigger': event['trigger'], 
                    'subtype': event['subtype'], 
                    'participants': filter_args(event['participants']), 
                    'locations': filter_args(event['locations']), 
                    'unknow': [], 
                    'chatgpt_res': event['chatgpt_res']
                })
        data['event_args'] = event_args
    os.remove(chatgpt_result_file)
    with open(chatgpt_result_file, 'wt') as f_out:
        for doc in Data:
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')

# fix_parsing_problems('chatgpt_dev_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)
# fix_parsing_problems('chatgpt_epoch_3_test_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)
# fix_parsing_problems('chatgpt_train_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)

# with open('chatgpt_epoch_3_test_pred_args.json', 'rt') as f_in:
#     for line in tqdm(f_in.readlines()):
#         data = json.loads(line.strip())
#         for event in data['event_args']:
#             if event['unknow']:
#                 print('-'*20, '\n', event['unknow'])
#                 print('part:', event['participants'])
#                 print('loc:', event['locations'])
