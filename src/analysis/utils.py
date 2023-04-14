import json

WORD_FILTER = set([
    'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'you', 'they', 'them', 'my', 'mine', 'your', 'yours', 'his', 'her', 'hers', 
    'its', 'our', 'ours', 'their', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 
    'other', 'others', 'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'whose', 'which', 'that', 'all', 'each', 'either', 'neither', 
    'one', 'any', 'oneself', 'such', 'same'
])

def get_pred_related_info(simi_file:str) -> dict:
    '''
    # Returns:
        related info dictionary: {doc_id: {event_offset: {
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

def select_args(my_args, other_related_info, match_other_related_args=True):
    if not my_args:
        return []
    other_has_part, other_has_place = False, False
    if match_other_related_args:
        other_args = other_related_info['arguments'] + list(
            filter(lambda x: x['mention'].lower() not in WORD_FILTER, other_related_info['related_arguments'])
        )
    else:
        other_args = other_related_info['arguments']
    for arg in other_args:
        if arg['role'] == 'participant':
            other_has_part = True
        elif arg['role'] == 'place':
            other_has_place = True    
    return [
        arg for arg in my_args 
        if (arg['role'] == 'participant' and other_has_part) or (arg['role'] == 'place' and other_has_place)
    ]

def convert_args_to_str(args:list, use_filter=True):
    if use_filter:
        args = filter(lambda x: x['mention'].lower() not in WORD_FILTER, args)
    participants, places = (
        [arg for arg in args if arg['role'] == 'participant'], 
        [arg for arg in args if arg['role'] == 'place']
    )
    return participants, places

def convert_related_info_to_str(related_args:list, use_filter=True):
    if use_filter:
        related_args = list(filter(lambda x: x['mention'].lower() not in WORD_FILTER, related_args))
    related_participants, related_places = (
        [arg for arg in related_args if arg['role'] == 'participant'], 
        [arg for arg in related_args if arg['role'] == 'place']
    )
    return related_participants, related_places

def get_event_arg_status(prompt_type, e1_related_info, e2_related_info, select_arg_strategy):
    if select_arg_strategy == 'filter_all':
        e1_part, e1_place = convert_args_to_str(select_args(e1_related_info['arguments'], e2_related_info, 'o' in prompt_type), not prompt_type.startswith('m'))
        e2_part, e2_place = convert_args_to_str(select_args(e2_related_info['arguments'], e1_related_info, 'o' in prompt_type), not prompt_type.startswith('m'))
    else:
        e1_part, e1_place = convert_args_to_str(e1_related_info['arguments'], not prompt_type.startswith('m'))
        e2_part, e2_place = convert_args_to_str(e2_related_info['arguments'], not prompt_type.startswith('m'))
    e1_related_triggers, e2_related_triggers = e1_related_info['related_triggers'], e2_related_info['related_triggers']
    if not e1_related_triggers or not e2_related_triggers:
        e1_related_triggers, e2_related_triggers = [], []
    if select_arg_strategy in ['filter_all', 'filter_related_args']:
        e1_related_part, e1_related_places = convert_related_info_to_str(select_args(e1_related_info['related_arguments'], e2_related_info, 'o' in prompt_type))
        e2_related_part, e2_related_places = convert_related_info_to_str(select_args(e2_related_info['related_arguments'], e1_related_info, 'o' in prompt_type))
    else:
        e1_related_part, e1_related_places = convert_related_info_to_str(e1_related_info['related_arguments'])
        e2_related_part, e2_related_places = convert_related_info_to_str(e2_related_info['related_arguments'])
    return {
        'e1_has_part': bool(e1_part), 'e1_has_place': bool(e1_place), 
        'e1_has_related_triggers': bool(e1_related_triggers), 
        'e1_has_related_part': bool(e1_related_part), 
        'e1_has_related_places': bool(e1_related_places), 
        'e2_has_part': bool(e2_part), 'e2_has_place': bool(e2_place), 
        'e2_has_related_triggers': bool(e2_related_triggers), 
        'e2_has_related_part': bool(e2_related_part), 
        'e2_has_related_places': bool(e2_related_places), 
        'e1_part': e1_part, 'e1_place': e1_place, 
        'e1_related_triggers': e1_related_triggers, 
        'e1_related_part': e1_related_part, 
        'e1_related_places': e1_related_places, 
        'e2_part': e2_part, 'e2_place': e2_place, 
        'e2_related_triggers': e2_related_triggers, 
        'e2_related_part': e2_related_part, 
        'e2_related_places': e2_related_places
    }

def pretty_event_mention(sentences, sent_idx, sent_offset, trigger, windows=10):
    before_sentence, after_sentence = '', ''
    for i in range(1,1+windows):
        if sent_idx - i >= 0:
            before_sentence = sentences[sent_idx - i]['text'] + ' ' + before_sentence
        if sent_idx + i < len(sentences):
            after_sentence += ' ' + sentences[sent_idx + i]['text']
    sentence = sentences[sent_idx]['text']
    assert sentence[sent_offset:sent_offset + len(trigger)] == trigger
    mention = "{}{} [START] {} [END] {}{}".format(
        before_sentence, 
        sentence[:sent_offset], 
        trigger, 
        sentence[sent_offset + len(trigger):], 
        after_sentence
    )
    return mention

def get_event_pair_info(
    prompt_type:str, select_arg_strategy:str, 
    gold_test_file:str, gold_test_simi_file:str, 
    pred_test_file:str, pred_test_simi_file:str
    ):
    '''get event pair information
    # Returns:
    event_pair_info_dict: 
        {
            doc_id: {
                {e_i_start}-{e_j_start}: {
                    'e_i_pretty_sent', 'e_j_pretty_sent',
                    'e_i_has_part','e_i_has_place': True/False, 
                    'e_i_has_related_triggers': True/False,  
                    'e_i_has_related_part': True/False, 
                    'e_i_has_related_places': True/False, 
                    ...
                    'e_i_part', 'e_i_place', 
                    'e_i_related_triggers', 
                    'e_i_related_part', 
                    'e_i_related_places',
                    'sent_dist': sentence distance
                }, ...
            }, ...
        }
    '''
    def _find_event_sent(event_start, trigger, sent_list):
        '''find out which sentence the event come from
        '''
        for idx, sent in enumerate(sent_list):
            s_start, s_end = sent['start'], sent['start'] + len(sent['text']) - 1
            if s_start <= event_start <= s_end:
                e_s_start = event_start - s_start
                assert sent['text'][e_s_start:e_s_start+len(trigger)] == trigger
                return idx, e_s_start
        print(event_start, trigger, '\n')
        for sent in sent_list:
            print(sent['start'], sent['start'] + len(sent['text']) - 1)
        return None, None

    event_pair_info_dict = {}
    gold_related_dict = get_pred_related_info(gold_test_simi_file)
    pred_related_dict = get_pred_related_info(pred_test_simi_file)
    with open(gold_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            events = sample['events']
            sentences = sample['sentences']
            event_pairs = {}
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                e_i_pretty_sent = pretty_event_mention(sentences, events[i]['sent_idx'], events[i]['sent_start'], events[i]['trigger'])
                e_i_related_info = gold_related_dict[sample['doc_id']][e_i_start]
                e_i_sent_idx = events[i]['sent_idx']
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    e_j_pretty_sent = pretty_event_mention(sentences, events[j]['sent_idx'], events[j]['sent_start'], events[j]['trigger'])
                    e_j_related_info = gold_related_dict[sample['doc_id']][e_j_start]
                    e_j_sent_idx = events[j]['sent_idx']
                    arg_status = get_event_arg_status(prompt_type, e_i_related_info, e_j_related_info, select_arg_strategy)
                    if e_i_start < e_j_start:
                        event_pairs[f'{e_i_start}-{e_j_start}'] = {
                            'e_i_pretty_sent': e_i_pretty_sent, 'e_j_pretty_sent': e_j_pretty_sent, 
                            'e_i_has_part': arg_status['e1_has_part'], 'e_i_has_place': arg_status['e1_has_place'], 
                            'e_i_has_related_triggers': arg_status['e1_has_related_triggers'], 
                            'e_i_has_related_part': arg_status['e1_has_related_part'], 
                            'e_i_has_related_places': arg_status['e1_has_related_places'], 
                            'e_j_has_part': arg_status['e2_has_part'], 'e_j_has_place': arg_status['e2_has_place'], 
                            'e_j_has_related_triggers': arg_status['e2_has_related_triggers'], 
                            'e_j_has_related_part': arg_status['e2_has_related_part'], 
                            'e_j_has_related_places': arg_status['e2_has_related_places'], 
                            'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 
                            'e_i_part': arg_status['e1_part'], 'e_i_place': arg_status['e1_place'], 
                            'e_i_related_triggers': arg_status['e1_related_triggers'], 
                            'e_i_related_part': arg_status['e1_related_part'], 
                            'e_i_related_places': arg_status['e1_related_places'], 
                            'e_j_part': arg_status['e2_part'], 'e_j_place': arg_status['e2_place'], 
                            'e_j_related_triggers': arg_status['e2_related_triggers'], 
                            'e_j_related_part': arg_status['e2_related_part'], 
                            'e_j_related_places': arg_status['e2_related_places']
                        }
                    else:
                        event_pairs[f'{e_j_start}-{e_i_start}'] = {
                            'e_i_pretty_sent': e_i_pretty_sent, 'e_j_pretty_sent': e_j_pretty_sent, 
                            'e_i_has_part': arg_status['e2_has_part'], 'e_i_has_place': arg_status['e2_has_place'], 
                            'e_i_has_related_triggers': arg_status['e2_has_related_triggers'], 
                            'e_i_has_related_part': arg_status['e2_has_related_part'], 
                            'e_i_has_related_places': arg_status['e2_has_related_places'], 
                            'e_j_has_part': arg_status['e1_has_part'], 'e_j_has_place': arg_status['e1_has_place'], 
                            'e_j_has_related_triggers': arg_status['e1_has_related_triggers'], 
                            'e_j_has_related_part': arg_status['e1_has_related_part'], 
                            'e_j_has_related_places': arg_status['e1_has_related_places'], 
                            'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 
                            'e_i_part': arg_status['e2_part'], 'e_i_place': arg_status['e2_place'], 
                            'e_i_related_triggers': arg_status['e2_related_triggers'], 
                            'e_i_related_part': arg_status['e2_related_part'], 
                            'e_i_related_places': arg_status['e2_related_places'], 
                            'e_j_part': arg_status['e1_part'], 'e_j_place': arg_status['e1_place'], 
                            'e_j_related_triggers': arg_status['e1_related_triggers'], 
                            'e_j_related_part': arg_status['e1_related_part'], 
                            'e_j_related_places': arg_status['e1_related_places']
                        }
            event_pair_info_dict[sample['doc_id']] = event_pairs
    with open(pred_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            events = sample['events']
            sentences = sample['sentences']
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                e_i_sent_idx, e_i_sent_start = _find_event_sent(events[i]['start'], events[i]
                ['trigger'], sentences)
                assert e_i_sent_idx is not None
                e_i_pretty_sent = pretty_event_mention(sentences, e_i_sent_idx, e_i_sent_start, events[i]['trigger'])
                e_i_related_info = pred_related_dict[sample['doc_id']][e_i_start]
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    e_j_sent_idx, e_j_sent_start = _find_event_sent(events[j]['start'], events[j]['trigger'], sentences)
                    assert e_j_sent_idx is not None
                    e_j_pretty_sent = pretty_event_mention(sentences, e_j_sent_idx, e_j_sent_start, events[j]['trigger'])
                    e_j_related_info = pred_related_dict[sample['doc_id']][e_j_start]
                    assert e_i_start < e_j_start
                    arg_status = get_event_arg_status(prompt_type, e_i_related_info, e_j_related_info, select_arg_strategy)
                    event_pair_info_dict[sample['doc_id']][f'{e_i_start}-{e_j_start}'] = { # overwrite same event-pair data
                        'e_i_pretty_sent': e_i_pretty_sent, 'e_j_pretty_sent': e_j_pretty_sent, 
                        'e_i_has_part': arg_status['e1_has_part'], 'e_i_has_place': arg_status['e1_has_place'], 
                        'e_i_has_related_triggers': arg_status['e1_has_related_triggers'], 
                        'e_i_has_related_part': arg_status['e1_has_related_part'], 
                        'e_i_has_related_places': arg_status['e1_has_related_places'], 
                        'e_j_has_part': arg_status['e2_has_part'], 'e_j_has_place': arg_status['e2_has_place'], 
                        'e_j_has_related_triggers': arg_status['e2_has_related_triggers'], 
                        'e_j_has_related_part': arg_status['e2_has_related_part'], 
                        'e_j_has_related_places': arg_status['e2_has_related_places'], 
                        'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 
                        'e_i_part': arg_status['e1_part'], 'e_i_place': arg_status['e1_place'], 
                        'e_i_related_triggers': arg_status['e1_related_triggers'], 
                        'e_i_related_part': arg_status['e1_related_part'], 
                        'e_i_related_places': arg_status['e1_related_places'], 
                        'e_j_part': arg_status['e2_part'], 'e_j_place': arg_status['e2_place'], 
                        'e_j_related_triggers': arg_status['e2_related_triggers'], 
                        'e_j_related_part': arg_status['e2_related_part'], 
                        'e_j_related_places': arg_status['e2_related_places']
                    }
    return event_pair_info_dict

def get_gold_corefs(gold_test_file:str) -> dict:

    def _get_event_cluster_id_and_link_len(event_id, clusters):
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id'], len(cluster['events'])
        return None, None

    gold_dict = {}
    with open(gold_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            clusters = sample['clusters']
            events = sample['events']
            event_pairs = {}
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                e_i_cluster_id, e_i_link_len = _get_event_cluster_id_and_link_len(events[i]['event_id'], clusters)
                assert e_i_cluster_id is not None
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    e_j_cluster_id, e_j_link_len = _get_event_cluster_id_and_link_len(events[j]['event_id'], clusters)
                    assert e_j_cluster_id is not None
                    if e_i_start < e_j_start:
                        event_pairs[f'{e_i_start}-{e_j_start}'] = {
                            'coref': 1 if e_i_cluster_id == e_j_cluster_id else 0, 
                            'e_i_link_len': e_i_link_len, 'e_j_link_len': e_j_link_len
                        }
                    else:
                        event_pairs[f'{e_j_start}-{e_i_start}'] = {
                            'coref': 1 if e_i_cluster_id == e_j_cluster_id else 0, 
                            'e_i_link_len': e_j_link_len, 'e_j_link_len': e_i_link_len
                        }
            gold_dict[sample['doc_id']] = event_pairs
    return gold_dict

def get_pred_coref_results(pred_test_file:str) -> dict:
    pred_dict = {}
    with open(pred_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            events = sample['events']
            pred_labels = sample['pred_label']
            event_pairs = {}
            event_pair_idx = 0
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    assert e_i_start < e_j_start
                    event_pairs[f'{e_i_start}-{e_j_start}'] = {
                        'coref': pred_labels[event_pair_idx], 
                        'e_i_link_len': 0, 'e_j_link_len': 0
                    }
                    event_pair_idx += 1
            assert event_pair_idx == len(pred_labels)
            pred_dict[sample['doc_id']] = event_pairs
    return pred_dict

def get_event_pair_set(
    prompt_type, select_arg_strategy, 
    gold_coref_file, gold_simi_coref_file, 
    pred_coref_file, pred_simi_coref_file
    ):
    event_pair_info = get_event_pair_info(
        prompt_type, select_arg_strategy, 
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file
    )
    gold_coref_results = get_gold_corefs(gold_coref_file)
    pred_coref_results = get_pred_coref_results(pred_coref_file)

    new_gold_coref_results = {}
    for doc_id, event_pairs in gold_coref_results.items():
        pred_event_pairs = pred_coref_results[doc_id]
        unrecognized_event_pairs = {}
        recognized_event_pairs = {}
        for pair_id, results in event_pairs.items():
            if pair_id in pred_event_pairs:
                recognized_event_pairs[pair_id] = results
            else:
                unrecognized_event_pairs[pair_id] = results
        new_gold_coref_results[doc_id] = {
            'unrecognized_event_pairs': unrecognized_event_pairs, 
            'recognized_event_pairs': recognized_event_pairs
        }
    new_pred_coref_results = {}
    for doc_id, event_pairs in pred_coref_results.items():
        gold_event_pairs = gold_coref_results[doc_id]
        recognized_event_pairs = {}
        wrong_event_pairs = {}
        for pair_id, results in event_pairs.items():
            if pair_id in gold_event_pairs:
                recognized_event_pairs[pair_id] = results
            else:
                wrong_event_pairs[pair_id] = results
        new_pred_coref_results[doc_id] = {
            'recognized_event_pairs': recognized_event_pairs, 
            'wrong_event_pairs': wrong_event_pairs
        }

    return event_pair_info, new_gold_coref_results, new_pred_coref_results
