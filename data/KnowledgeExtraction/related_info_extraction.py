from collections import defaultdict
import json

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

def create_event_simi_dict(simi_file, cosine_threshold):
    doc_simi_dict = {}
    with open(simi_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            event_pairs_id, event_pairs_cos = sample['event_pairs_id'], sample['event_pairs_cos']
            simi_dict = defaultdict(list)
            for id_pair, cos in zip(event_pairs_id, event_pairs_cos):
                if cos < cosine_threshold:
                    continue
                e1_id, e2_id = id_pair.split('###')
                simi_dict[e1_id].append({'id': e2_id, 'cos': cos})
                simi_dict[e2_id].append({'id': e1_id, 'cos': cos})
            for simi_list in simi_dict.values():
                simi_list.sort(key=lambda x:x['cos'], reverse=True)
            doc_simi_dict[sample['doc_id']] = simi_dict
        return doc_simi_dict

def get_event_by_id(event_id, events):
    for e in events:
        if e['event_id'] == event_id:
            return e
    return None

def create_simi_event_file(data_file, arg_file, simi_file, save_file, cosine_threshold):
    doc_arg_dict = get_pred_arguments(arg_file)
    doc_simi_dict = create_event_simi_dict(simi_file, cosine_threshold)
    Results = []
    with open(data_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            simi_info = {}
            arg_dict, simi_dict = doc_arg_dict[sample['doc_id']], doc_simi_dict[sample['doc_id']]
            events = sample['events']
            for e in events:
                if e['event_id'] in simi_dict:
                    triggers, trigger_offsets, args = set(), [], []
                    for related_e in simi_dict[e['event_id']]:
                        r_e = get_event_by_id(related_e['id'], events)
                        triggers.add(r_e['trigger'])
                        trigger_offsets.append(r_e['start'])
                        args += arg_dict[r_e['start']]
                    simi_info[e['start']] = {
                        'trigger': e['trigger'], 
                        'arguments': arg_dict[e['start']], 
                        'related_triggers': list(triggers), 
                        'related_trigger_offsets': trigger_offsets, 
                        'related_arguments': args
                    }
                else:
                    simi_info[e['start']] = {
                        'trigger': e['trigger'], 
                        'arguments': arg_dict[e['start']], 
                        'related_triggers': [], 
                        'related_trigger_offsets': [], 
                        'related_arguments': []
                    }
            Results.append({
                "doc_id": sample['doc_id'], 
                "relate_info": simi_info
            })
    with open(save_file, 'wt', encoding='utf-8') as f:
        for example_result in Results:
            f.write(json.dumps(example_result) + '\n')

def get_event_by_id_for_testfile(event_id, events):
    for e in events:
        if f"e-{e['start']}" == event_id:
            return e
    return None

def create_simi_event_file_for_testfile(data_file, arg_file, simi_file, save_file, cosine_threshold):
    doc_arg_dict = get_pred_arguments(arg_file)
    doc_simi_dict = create_event_simi_dict(simi_file, cosine_threshold)
    Results = []
    with open(data_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            simi_info = {}
            arg_dict, simi_dict = doc_arg_dict[sample['doc_id']], doc_simi_dict[sample['doc_id']]
            events = sample['pred_label']
            for e in events:
                if f"e-{e['start']}" in simi_dict:
                    triggers, trigger_offsets, args = set(), [], []
                    for related_e in simi_dict[f"e-{e['start']}"]:
                        r_e = get_event_by_id_for_testfile(related_e['id'], events)
                        triggers.add(r_e['trigger'])
                        trigger_offsets.append(r_e['start'])
                        args += arg_dict[r_e['start']]
                    simi_info[e['start']] = {
                        'trigger': e['trigger'], 
                        'arguments': arg_dict[e['start']], 
                        'related_triggers': list(triggers), 
                        'related_trigger_offsets': trigger_offsets, 
                        'related_arguments': args
                    }
                else:
                    simi_info[e['start']] = {
                        'trigger': e['trigger'], 
                        'arguments': arg_dict[e['start']], 
                        'related_triggers': [], 
                        'related_trigger_offsets': [], 
                        'related_arguments': []
                    }
            Results.append({
                "doc_id": sample['doc_id'], 
                "relate_info": simi_info
            })
    with open(save_file, 'wt', encoding='utf-8') as f:
        for example_result in Results:
            f.write(json.dumps(example_result) + '\n')

cosine_threshold = 0.9

# create_simi_event_file(
#     '../train_filtered.json', 
#     'omni_train_pred_args.json', 
#     '../train_filtered_with_cos.json', 
#     f'simi_train_related_info_{cosine_threshold}.json', 
#     cosine_threshold
# )
# create_simi_event_file(
#     '../dev_filtered.json', 
#     'omni_dev_pred_args.json', 
#     '../dev_filtered_with_cos.json', 
#     f'simi_dev_related_info_{cosine_threshold}.json', 
#     cosine_threshold
# )
# create_simi_event_file(
#     '../test_filtered.json', 
#     'omni_gold_test_pred_args.json', 
#     '../test_filtered_with_cos.json', 
#     f'simi_gold_test_related_info_{cosine_threshold}.json', 
#     cosine_threshold
# )
# create_simi_event_file_for_testfile(
#     '../epoch_3_test_pred_events.json', 
#     'omni_epoch_3_test_pred_args.json', 
#     '../epoch_3_test_pred_events_with_cos.json', 
#     f'simi_epoch_3_test_related_info_{cosine_threshold}.json', 
#     cosine_threshold
# )

def analysis(simi_file):
    word_filter = set([
        'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'you', 'they', 'them', 'my', 'mine', 'your', 'yours', 'his', 'her', 'hers', 
        'its', 'our', 'ours', 'their', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 
        'other', 'others', 'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'whose', 'which', 'that', 'all', 'each', 'either', 'neither', 
        'one', 'any', 'oneself', 'such', 'same'
    ])
    show = []
    total_event, no_arg_event, find_arg_event = 0, 0, 0
    with open(simi_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            doc_id = sample['doc_id']
            for offset, event in sample['relate_info'].items():
                total_event += 1
                if not event['arguments']:
                    no_arg_event += 1
                    if list(filter(lambda x: x['mention'].lower() not in word_filter, event['related_arguments'])):
                        find_arg_event += 1
                        event['doc_id'] = doc_id
                        event['offset'] = offset
                        show.append(event)
    print(total_event, no_arg_event, find_arg_event, no_arg_event - find_arg_event)
    print(f"{(no_arg_event / total_event * 100):0.1f} => {((no_arg_event - find_arg_event) / total_event * 100):0.1f}")
    # for e in show[:3]:
    #     print(e)

analysis(f'simi_train_related_info_{cosine_threshold}.json')
analysis(f'simi_dev_related_info_{cosine_threshold}.json')
analysis(f'simi_gold_test_related_info_{cosine_threshold}.json')
analysis(f'simi_epoch_3_test_related_info_{cosine_threshold}.json')

