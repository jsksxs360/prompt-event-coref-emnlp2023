import json

WORD_FILTER = set([
    'you', 'your', 'yours', 'yourself', 'yourselves', 
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 
    'he', 'his', 'him', 'himself', 'she', 'her', 'herself', 'hers', 
    'it', 'its', 'itself', 'they', 'their', 'theirs', 'them', 'themselves', 'other', 'others', 
    'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'whose', 'which', 'where', 'why', 
    'that', 'all', 'each', 'either', 'neither', 
    'one', 'any', 'oneself', 'such', 'same', 'everyone', 'anyone', 'there', 
])

def get_pred_related_info(simi_file:str) -> dict:
    '''
    # Returns:
        {doc_id: {event_offset: {\n
            'arguments': [{"global_offset": 798, "mention": "We", "role": "participant"}]\n
            'related_triggers': ['charged'], \n
            'related_arguments': [\n
                {'global_offset': 1408, 'mention': 'Garvina', 'role': 'participant'}, \n
                {'global_offset': 1368, 'mention': 'Prosecutors', 'role': 'participant'}\n
            ]\n
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

def select_args(my_args:list, other_related_info:dict, match_other_related_args:bool) -> list:
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
        if arg['role'] == 'place':
            other_has_place = True    
    return [
        arg for arg in my_args 
        if (arg['role'] == 'participant' and other_has_part) or (arg['role'] == 'place' and other_has_place)
    ]

def convert_info_to_str(info:list, use_filter:bool):
    if use_filter:
        info = list(filter(lambda x: x['mention'].lower() not in WORD_FILTER, info))
    participants, places = (
        [arg for arg in info if arg['role'] == 'participant'], 
        [arg for arg in info if arg['role'] == 'place']
    )
    return participants, places

def get_event_arg_status_easy(e1_related_info, e2_related_info):
    e1_args = e1_related_info['arguments']
    e2_args = e2_related_info['arguments']
    e1_part, e1_place = convert_info_to_str(e1_args, False)
    e2_part, e2_place = convert_info_to_str(e2_args, False)
    e1_related_triggers = e1_related_info['related_triggers']
    e2_related_triggers = e2_related_info['related_triggers']
    e1_related_args = e1_related_info['related_arguments']
    e2_related_args = e2_related_info['related_arguments']
    e1_related_part, e1_related_places = convert_info_to_str(e1_related_args, False)
    e2_related_part, e2_related_places = convert_info_to_str(e2_related_args, False)
    return {
        'e1_has_part': bool(e1_part), 
        'e1_has_place': bool(e1_place), 
        'e1_has_related_triggers': bool(e1_related_triggers), 
        'e1_has_related_part': bool(e1_related_part), 
        'e1_has_related_places': bool(e1_related_places), 
        'e2_has_part': bool(e2_part), 
        'e2_has_place': bool(e2_place), 
        'e2_has_related_triggers': bool(e2_related_triggers), 
        'e2_has_related_part': bool(e2_related_part), 
        'e2_has_related_places': bool(e2_related_places), 
        'e1_part': e1_part, 
        'e1_place': e1_place, 
        'e1_related_triggers': e1_related_triggers, 
        'e1_related_part': e1_related_part, 
        'e1_related_places': e1_related_places, 
        'e2_part': e2_part, 
        'e2_place': e2_place, 
        'e2_related_triggers': e2_related_triggers, 
        'e2_related_part': e2_related_part, 
        'e2_related_places': e2_related_places
    }

def get_event_arg_status_filter(prompt_type, e1_related_info, e2_related_info, select_arg_strategy):
    e1_args = select_args(e1_related_info['arguments'], e2_related_info, 'tao' in prompt_type) \
        if select_arg_strategy == 'filter_all' else e1_related_info['arguments']
    e2_args = select_args(e2_related_info['arguments'], e1_related_info, 'tao' in prompt_type) \
        if select_arg_strategy == 'filter_all' else e2_related_info['arguments']
    e1_part, e1_place = convert_info_to_str(e1_args, not prompt_type.startswith('m'))
    e2_part, e2_place = convert_info_to_str(e2_args, not prompt_type.startswith('m'))
    e1_related_triggers, e2_related_triggers = e1_related_info['related_triggers'], e2_related_info['related_triggers']
    if not e1_related_triggers or not e2_related_triggers:
        e1_related_triggers, e2_related_triggers = [], []
    e1_related_args = select_args(e1_related_info['related_arguments'], e2_related_info, 'tao' in prompt_type) \
        if select_arg_strategy in ['filter_all', 'filter_related_args'] else e1_related_info['related_arguments']
    e2_related_args = select_args(e2_related_info['related_arguments'], e1_related_info, 'tao' in prompt_type) \
        if select_arg_strategy in ['filter_all', 'filter_related_args'] else e2_related_info['related_arguments']
    e1_related_part, e1_related_places = convert_info_to_str(e1_related_args, True)
    e2_related_part, e2_related_places = convert_info_to_str(e2_related_args, True)
    return {
        'e1_has_part': bool(e1_part), 
        'e1_has_place': bool(e1_place), 
        'e1_has_related_triggers': bool(e1_related_triggers), 
        'e1_has_related_part': bool(e1_related_part), 
        'e1_has_related_places': bool(e1_related_places), 
        'e2_has_part': bool(e2_part), 
        'e2_has_place': bool(e2_place), 
        'e2_has_related_triggers': bool(e2_related_triggers), 
        'e2_has_related_part': bool(e2_related_part), 
        'e2_has_related_places': bool(e2_related_places), 
        'e1_part': e1_part, 
        'e1_place': e1_place, 
        'e1_related_triggers': e1_related_triggers, 
        'e1_related_part': e1_related_part, 
        'e1_related_places': e1_related_places, 
        'e2_part': e2_part, 
        'e2_place': e2_place, 
        'e2_related_triggers': e2_related_triggers, 
        'e2_related_part': e2_related_part, 
        'e2_related_places': e2_related_places
    }

def pretty_event_mention(
    sentences, sent_idx, sent_offset, trigger, 
    start_tag='[EVENT]', end_tag='[/EVENT]', context_windows=2
    ):
    sentence = sentences[sent_idx]['text']
    assert sentence[sent_offset:sent_offset + len(trigger)] == trigger
    before_sentence, after_sentence = '', ''
    for i in range(1,1+context_windows):
        if sent_idx - i >= 0:
            before_sentence = sentences[sent_idx - i]['text'] + ' ' + before_sentence
        if sent_idx + i < len(sentences):
            after_sentence += ' ' + sentences[sent_idx + i]['text']
    return "{}{}{} {} {}{}{}".format(
        before_sentence, 
        sentence[:sent_offset], 
        start_tag, 
        trigger, 
        end_tag, 
        sentence[sent_offset + len(trigger):], 
        after_sentence
    )

def get_event_pair_info(
    gold_test_file:str, gold_test_simi_file:str, 
    pred_test_file:str, pred_test_simi_file:str,
    mode:str, prompt_type:str=None, select_arg_strategy:str=None
    ):
    '''get event pair information
    # Args:
        mode:
            easy: output directly according to the extracted arguments
            filter: filter arguments by prompt type and select arg strategy
    # Returns:
    {
        doc_id: {
            {e_i_start}-{e_j_start}: {
                'e_i_pretty_sent', 'e_j_pretty_sent', \n
                'e_i_has_part': True/False, \n
                'e_i_has_place': True/False, \n
                'e_i_has_related_triggers': True/False, \n
                'e_i_has_related_part': True/False, \n
                'e_i_has_related_places': True/False, \n
                ...\n
                'e_i_part', \n
                'e_i_place', \n
                'e_i_related_triggers', \n
                'e_i_related_part', \n
                'e_i_related_places', \n
                'sent_dist': sentence distance\n
            }, ...
        }, ...
    }
    '''
    assert (mode == 'filter') == bool(prompt_type and select_arg_strategy)

    def _find_event_sent(event_start, trigger, sent_list):
        '''find out which sentence the event come from
        # Returns:
        idx:
            sentence index in the sentence list
        e_s_start:
            trigger offset in the host sentence
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
        raise ValueError(f'no matching host sentence for event: {event_start}')

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
                for j in range(i + 1, len(events)):
                    e_i, e_j = events[i], events[j]
                    if e_i['start'] > e_j['start']:
                        e_i, e_j = e_j, e_i
                    e_i_start = e_i['start']
                    e_i_pretty_sent = pretty_event_mention(sentences, e_i['sent_idx'], e_i['sent_start'], e_i['trigger'])
                    e_i_related_info = gold_related_dict[sample['doc_id']][e_i_start]
                    e_i_sent_idx = e_i['sent_idx']
                    e_j_start = e_j['start']
                    e_j_pretty_sent = pretty_event_mention(sentences, e_j['sent_idx'], e_j['sent_start'], e_j['trigger'])
                    e_j_related_info = gold_related_dict[sample['doc_id']][e_j_start]
                    e_j_sent_idx = e_j['sent_idx']
                    arg_status = get_event_arg_status_easy(e_i_related_info, e_j_related_info) if mode == 'easy' else \
                        get_event_arg_status_filter(prompt_type, e_i_related_info, e_j_related_info, select_arg_strategy)
                    event_pairs[f'{e_i_start}-{e_j_start}'] = {
                        'e_i_pretty_sent': e_i_pretty_sent, # event i
                        'e_i_has_part': arg_status['e1_has_part'], 
                        'e_i_has_place': arg_status['e1_has_place'], 
                        'e_i_has_related_triggers': arg_status['e1_has_related_triggers'], 
                        'e_i_has_related_part': arg_status['e1_has_related_part'], 
                        'e_i_has_related_places': arg_status['e1_has_related_places'], 
                        'e_i_part': arg_status['e1_part'], 
                        'e_i_place': arg_status['e1_place'], 
                        'e_i_related_triggers': arg_status['e1_related_triggers'], 
                        'e_i_related_part': arg_status['e1_related_part'], 
                        'e_i_related_places': arg_status['e1_related_places'], 
                        'e_j_pretty_sent': e_j_pretty_sent, # event j
                        'e_j_has_part': arg_status['e2_has_part'], 
                        'e_j_has_place': arg_status['e2_has_place'], 
                        'e_j_has_related_triggers': arg_status['e2_has_related_triggers'], 
                        'e_j_has_related_part': arg_status['e2_has_related_part'], 
                        'e_j_has_related_places': arg_status['e2_has_related_places'], 
                        'e_j_part': arg_status['e2_part'], 
                        'e_j_place': arg_status['e2_place'], 
                        'e_j_related_triggers': arg_status['e2_related_triggers'], 
                        'e_j_related_part': arg_status['e2_related_part'], 
                        'e_j_related_places': arg_status['e2_related_places'], 
                        'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx))
                    }
            event_pair_info_dict[sample['doc_id']] = event_pairs
    with open(pred_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            events = sample['events']
            sentences = sample['sentences']
            for i in range(len(events) - 1):
                for j in range(i + 1, len(events)):
                    e_i, e_j = events[i], events[j]
                    if e_i['start'] > e_j['start']:
                        e_i, e_j = e_j, e_i
                    e_i_start = e_i['start']
                    e_i_sent_idx, e_i_sent_start = _find_event_sent(e_i['start'], e_i['trigger'], sentences)
                    e_i_pretty_sent = pretty_event_mention(sentences, e_i_sent_idx, e_i_sent_start, e_i['trigger'])
                    e_i_related_info = pred_related_dict[sample['doc_id']][e_i_start]
                    e_j_start = e_j['start']
                    e_j_sent_idx, e_j_sent_start = _find_event_sent(e_j['start'], e_j['trigger'], sentences)
                    e_j_pretty_sent = pretty_event_mention(sentences, e_j_sent_idx, e_j_sent_start, e_j['trigger'])
                    e_j_related_info = pred_related_dict[sample['doc_id']][e_j_start]
                    arg_status = get_event_arg_status_easy(e_i_related_info, e_j_related_info) if mode == 'easy' else \
                        get_event_arg_status_filter(prompt_type, e_i_related_info, e_j_related_info, select_arg_strategy)
                    event_pair_info_dict[sample['doc_id']][f'{e_i_start}-{e_j_start}'] = { # overwrite same event-pair data
                        'e_i_pretty_sent': e_i_pretty_sent, # event i
                        'e_i_has_part': arg_status['e1_has_part'], 
                        'e_i_has_place': arg_status['e1_has_place'], 
                        'e_i_has_related_triggers': arg_status['e1_has_related_triggers'], 
                        'e_i_has_related_part': arg_status['e1_has_related_part'], 
                        'e_i_has_related_places': arg_status['e1_has_related_places'], 
                        'e_i_part': arg_status['e1_part'], 
                        'e_i_place': arg_status['e1_place'], 
                        'e_i_related_triggers': arg_status['e1_related_triggers'], 
                        'e_i_related_part': arg_status['e1_related_part'], 
                        'e_i_related_places': arg_status['e1_related_places'], 
                        'e_j_pretty_sent': e_j_pretty_sent, # event j
                        'e_j_has_part': arg_status['e2_has_part'], 
                        'e_j_has_place': arg_status['e2_has_place'], 
                        'e_j_has_related_triggers': arg_status['e2_has_related_triggers'], 
                        'e_j_has_related_part': arg_status['e2_has_related_part'], 
                        'e_j_has_related_places': arg_status['e2_has_related_places'], 
                        'e_j_part': arg_status['e2_part'], 
                        'e_j_place': arg_status['e2_place'], 
                        'e_j_related_triggers': arg_status['e2_related_triggers'], 
                        'e_j_related_part': arg_status['e2_related_part'], 
                        'e_j_related_places': arg_status['e2_related_places'], 
                        'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx))
                    }
    return event_pair_info_dict

def get_gold_corefs(gold_test_file:str) -> dict:

    def _get_event_cluster_id_and_link_len(event_id, clusters):
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id'], len(cluster['events'])
        raise ValueError(f'Unknown event id: {event_id}')

    gold_dict = {}
    with open(gold_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            clusters = sample['clusters']
            events = sample['events']
            event_pairs = {}
            for i in range(len(events) - 1):
                for j in range(i + 1, len(events)):
                    e_i, e_j = events[i], events[j]
                    if e_i['start'] > e_j['start']:
                        e_i, e_j = e_j, e_i
                    e_i_cluster_id, e_i_link_len = _get_event_cluster_id_and_link_len(e_i['event_id'], clusters)
                    e_j_cluster_id, e_j_link_len = _get_event_cluster_id_and_link_len(e_j['event_id'], clusters)
                    event_pairs[f"{e_i['start']}-{e_j['start']}"] = {
                        'coref': 1 if e_i_cluster_id == e_j_cluster_id else 0, 
                        'e_i_link_len': e_i_link_len, 'e_j_link_len': e_j_link_len
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
                for j in range(i + 1, len(events)):
                    e_i, e_j = events[i], events[j]
                    if e_i['start'] > e_j['start']:
                        e_i, e_j = e_j, e_i
                    event_pairs[f"{e_i['start']}-{e_j['start']}"] = {
                        'coref': pred_labels[event_pair_idx], 
                        'e_i_link_len': 0, 'e_j_link_len': 0
                    }
                    event_pair_idx += 1
            assert event_pair_idx == len(pred_labels)
            pred_dict[sample['doc_id']] = event_pairs
    return pred_dict

def get_event_pair_set(
    gold_coref_file:str, gold_simi_coref_file:str, 
    pred_coref_file:str, pred_simi_coref_file:str, 
    mode:str, prompt_type:str=None, select_arg_strategy:str=None, 
    ):
    '''get all event pair info
    # Args:
        mode:
            easy: output directly according to the extracted arguments
            filter: filter arguments by prompt type and select arg strategy
    # Returns:
    event_pair_info:
        event pair argument & related info dict, {
            doc_id: {
                e_i_start-e_j_start: {
                    'e_i_pretty_sent', 'e_j_pretty_sent', ...
                }
            }
        }
    new_gold_coref_results: 
        {
            doc_id: {
                'unrecognized_event_pairs': {
                    'e_i_start-e_j_start': {'coref', 'e_i_link_len', 'e_j_link_len'}
                }, 
                'recognized_event_pairs': {
                    'e_i_start-e_j_start': {'coref', 'e_i_link_len', 'e_j_link_len'}
                }
            }
        }
    new_pred_coref_results:
        {
            doc_id: {
                'recognized_event_pairs': {
                    'e_i_start-e_j_start': {'coref', 'e_i_link_len', 'e_j_link_len'}
                }, 
                'wrong_event_pairs': {
                    'e_i_start-e_j_start': {'coref', 'e_i_link_len', 'e_j_link_len'}
                }
            }
        }
    '''
    assert (mode == 'filter') == bool(prompt_type and select_arg_strategy)

    event_pair_info = get_event_pair_info(
        gold_coref_file, gold_simi_coref_file, 
        pred_coref_file, pred_simi_coref_file, 
        mode, prompt_type, select_arg_strategy, 
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

    assert sum([len(res['recognized_event_pairs']) for res in new_gold_coref_results.values()]) == sum([len(res['recognized_event_pairs']) for res in new_pred_coref_results.values()]) 
    return event_pair_info, new_gold_coref_results, new_pred_coref_results
