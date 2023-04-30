import openai
from tqdm.auto import tqdm
import json
import os
import time
import re

openai.api_key = "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"
START_TAG, END_TAG = '<EVENT>', '</EVENT>'
WINDOWS = 5

def get_argument_info(pretty_event_mention, trigger, start_tag, end_tag):
    time.sleep(0.05)
    event = f'{start_tag} {trigger} {end_tag}'
    assert event in pretty_event_mention
    prefix = (
        f'In the given document, please focus on the possible event labeled as {event} and extract its corresponding participants and locations. '
        f"All extracted items must be directly related to {event}, all the other entities should be excluded. "
        f"If {event} is not an event, answer: none. The given document is: "
    )
    instruction = (
        f"Please extract the directly related participants and locations of {event} from the above document without rephrasing. "
        "Respond in the form of a list: - Participants: 'participant 1', 'participant 2'...\n-Locations: 'location 1', 'location 2'..."
        f"\nIf participants or locations of {event} do not exist in the above document, the corresponding list item is none. "
        "Do not interpret the extracted information. Keep the answer short and concise."   
    )
    prompt = prefix + pretty_event_mention + '\n' + instruction
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in event information extraction."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    res = response['choices'][0]['message']['content'].strip()
    return res

def pretty_event_mention(
    sentences, sent_idx, sent_offset, trigger, 
    start_tag, end_tag, windows
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
                    'chatgpt_res': pred_res
                })
                print(pred_res)
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
                    'chatgpt_res': pred_res
                })
                print(pred_res)
            doc = {
                'doc_id': sample['doc_id'], 
                'document': sample['document'], 
                'event_args': pred_event_args
            }
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return True

# while True:
#     try:
#         status = event_argument_extraction('../train_filtered.json', './argument_files/chatgpt3.5_train_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)
# while True:
#     try:
#         status = event_argument_extraction('../dev_filtered.json', './argument_files/chatgpt3.5_dev_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)
# while True:
#     try:
#         status = event_argument_extraction('../test_filtered.json', './argument_files/chatgpt3.5_gold_test_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)
# while True:
#     try:
#         status = event_argument_extraction_for_testfile('../test_filtered.json', '../epoch_3_test_pred_events.json', './argument_files/chatgpt3.5_epoch_3_test_pred_args.json')
#         if status:
#             break
#     except Exception as ex:
#         print(f"Caught exception {ex}")
#         time.sleep(30)

def parsing(chatgpt_res, start_tag, end_tag, max_length):

    def check_arg(arg, max_length):
        if arg and len(arg.split()) <= max_length:
            return True, arg
        if '(' in arg and ')' in arg:
            arg = re.sub(u"\\(.*?\\)", "", arg).strip()
        if len(arg.split()) > max_length and ':' in arg:
            arg = arg[:arg.find(':')].strip()
        if len(arg.split()) > max_length and ',' in arg:
            arg = arg[:arg.find(',')].strip()
        if len(arg.split()) > max_length and 'who' in arg:
            arg = arg[:arg.find('who')].strip()
        if arg and len(arg.split()) <= max_length:
            return True, arg
        else:
            return False, ''
    
    res_lines = chatgpt_res.split('\n')
    flag, participants, locations = '', [], []
    filter_words = set([
        'no', 'not', 'none', 'none.', 'unidentified', 'unidentified.', 
        'unknown', 'unknown.', 'unknow', 'unknow.', 'event:', 'events:'
    ])
    filter_spans = set([start_tag.lower(), end_tag.lower(), '@', '.com', '.net', '.org'])
    for line in res_lines:
        line = line.strip()
        core_line = re.sub(u"\\(.*?\\)", "", line).strip().lower()
        core_line_words = core_line.split()
        if not line or \
            True in [(fw in core_line_words) for fw in filter_words] or \
            True in [(fs in core_line) for fs in filter_spans]:
            if 'implied to be in ' in core_line:
                item = line[line.find('implied to be in ') + len('implied to be in '):].strip()
                if item and 'participant' in line.lower():
                    should_add, new_p = check_arg(item, max_length)
                    if should_add:
                        participants.append(item)
                elif item and 'location' in line.lower():
                    should_add, new_l = check_arg(item, max_length)
                    if should_add:
                        locations.append(new_l)
            elif 'implied to be ' in core_line:
                item = line[line.find('implied to be ') + len('implied to be '):].strip()
                if item and 'participant' in line.lower():
                    should_add, new_p = check_arg(item, max_length)
                    if should_add:
                        participants.append(item)
                elif item and 'location' in line.lower():
                    should_add, new_l = check_arg(item, max_length)
                    if should_add:
                        locations.append(new_l)
            continue
        if not line.startswith('-'): # list title
            if 'participant' in line.lower():
                flag = 'part'
                p = line[line.find(':')+1:].strip()
                should_add, new_p = check_arg(p, max_length)
                if should_add:
                    participants.append(new_p)
            elif 'location' in line.lower():
                flag = 'loc'
                l = line[line.find(':')+1:].strip()
                should_add, new_l = check_arg(l, max_length)
                if should_add:
                    locations.append(new_l)
            else:
                flag = 'unk'
        else: # list item
            if line.lower().startswith('- participant'):
                flag = 'part'
                p = line[line.find(':')+1:].strip()
                should_add, new_p = check_arg(p, max_length)
                if should_add:
                    participants.append(new_p)
            elif line.lower().startswith('- location'):
                flag = 'loc'
                l = line[line.find(':')+1:].strip()
                should_add, new_l = check_arg(l, max_length)
                if should_add:
                    locations.append(new_l)
            else:
                item = line[2:]
                should_add, new_item = check_arg(item, max_length)
                if should_add:
                    if flag == 'part':
                        participants.append(new_item)
                    elif flag == 'loc':
                        locations.append(new_item)
    return participants, locations

def parse_arg_file(chatgpt_result_file, start_tag, end_tag, max_length=15):
    Data = []
    with open(chatgpt_result_file, 'rt') as f_in:
        for line in f_in.readlines():
            Data.append(json.loads(line.strip()))
    for data in Data:
        for event in data['event_args']:
            participants, locations = parsing(
                event['chatgpt_res'], start_tag, end_tag, max_length
            )
            event['participants'] = participants
            event['locations'] = locations
    os.remove(chatgpt_result_file)
    with open(chatgpt_result_file, 'wt') as f_out:
        for doc in Data:
            f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')

# parse_arg_file('./argument_files/chatgpt3.5_train_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)
# parse_arg_file('./argument_files/chatgpt3.5_dev_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)
# parse_arg_file('./argument_files/chatgpt3.5_gold_test_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)
# parse_arg_file('./argument_files/chatgpt3.5_epoch_3_test_pred_args.json', start_tag=START_TAG, end_tag=END_TAG)
