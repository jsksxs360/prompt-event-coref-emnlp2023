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

def get_event_arg_status(related_info):
    has_arg, find_arg = False, False
    has_part, has_place = False, False
    related_part, related_place = False, False
    related_arguments = list(filter(lambda x: x['mention'].lower() not in WORD_FILTER, related_info['related_arguments']))
    if related_info['arguments']:
        has_arg = True
        for arg in related_info['arguments']:
            if arg['role'] == 'participant':
                has_part = True
            elif arg['role'] == 'place':
                has_place = True
    else:
        if related_arguments:
            find_arg = True
    if related_arguments:
        for arg in related_arguments:
            if arg['role'] == 'participant':
                related_part = True
            elif arg['role'] == 'place':
                related_place = True
    return {
        'has_arg': has_arg, 'find_arg': find_arg, 
        'has_part': has_part, 'has_place': has_place, 
        'related_part': related_part, 'related_place': related_place
    }

def get_gold_corefs(gold_test_file:str, gold_test_simi_file:str) -> dict:
    '''get gold event pair statistics
    # Returns:
    gold_dict: 
        {
            doc_id: {
                {e_i_start}-{e_j_start}: {
                    'coref': 1/0, 
                    'e_i_has_arg','e_j_has_arg': True/False, 
                    'e_i_find_arg', 'e_j_find_arg': True/False, 
                    'sent_dist': sentence distance, 
                    'e_i_link_len', 'e_j_link_len': event link length
                }, ...
            }, ...
        }
    '''
    def _get_event_cluster_id_and_link_len(event_id, clusters):
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id'], len(cluster['events'])
        return None, None

    gold_dict = {}
    related_dict = get_pred_related_info(gold_test_simi_file)
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
                e_i_arg_status = get_event_arg_status(related_dict[sample['doc_id']][e_i_start])
                e_i_sent_idx = events[i]['sent_idx']
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    e_j_cluster_id, e_j_link_len = _get_event_cluster_id_and_link_len(events[j]['event_id'], clusters)
                    assert e_j_cluster_id is not None
                    e_j_arg_status = get_event_arg_status(related_dict[sample['doc_id']][e_j_start])
                    e_j_sent_idx = events[j]['sent_idx']
                    if e_i_start < e_j_start:
                        event_pairs[f'{e_i_start}-{e_j_start}'] = {
                            'coref': 1 if e_i_cluster_id == e_j_cluster_id else 0, 
                            'e_i_has_arg': e_i_arg_status['has_arg'], 'e_j_has_arg': e_j_arg_status['has_arg'], 
                            'e_i_find_arg': e_i_arg_status['find_arg'], 'e_j_find_arg': e_j_arg_status['find_arg'], 
                            'e_i_has_part': e_i_arg_status['has_part'], 'e_i_has_place': e_i_arg_status['has_place'], 
                            'e_j_has_part': e_j_arg_status['has_part'], 'e_j_has_place': e_j_arg_status['has_place'], 
                            'e_i_related_part': e_i_arg_status['related_part'], 'e_i_related_place': e_i_arg_status['related_place'], 
                            'e_j_related_part': e_j_arg_status['related_part'], 'e_j_related_place': e_j_arg_status['related_place'], 
                            'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 
                            'e_i_link_len': e_i_link_len, 'e_j_link_len': e_j_link_len
                        }
                    else:
                        event_pairs[f'{e_j_start}-{e_i_start}'] = {
                            'coref': 1 if e_i_cluster_id == e_j_cluster_id else 0, 
                            'e_i_has_arg': e_j_arg_status['has_arg'], 'e_j_has_arg': e_i_arg_status['has_arg'], 
                            'e_i_find_arg': e_j_arg_status['find_arg'], 'e_j_find_arg': e_i_arg_status['find_arg'], 
                            'e_i_has_part': e_j_arg_status['has_part'], 'e_i_has_place': e_j_arg_status['has_place'], 
                            'e_j_has_part': e_i_arg_status['has_part'], 'e_j_has_place': e_i_arg_status['has_place'], 
                            'e_i_related_part': e_j_arg_status['related_part'], 'e_i_related_place': e_j_arg_status['related_place'], 
                            'e_j_related_part': e_i_arg_status['related_part'], 'e_j_related_place': e_i_arg_status['related_place'], 
                            'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 
                            'e_i_link_len': e_j_link_len, 'e_j_link_len': e_i_link_len
                        }
            gold_dict[sample['doc_id']] = event_pairs
    return gold_dict

def get_pred_coref_results(pred_test_file:str, pred_test_simi_file:str) -> dict:
    '''get pred event pair statistics
    # Returns:
    pred_dict: 
        {
            doc_id: {
                {e_i_start}-{e_j_start}: {
                    'coref': 1/0, 
                    'e_i_has_arg','e_j_has_arg': True/False, 
                    'e_i_find_arg', 'e_j_find_arg': True/False, 
                    'sent_dist': sentence distance, 
                    'e_i_link_len', 'e_j_link_len': event link length
                }, ...
            }, ...
        }
    '''
    def _get_event_sent_idx(e_start, e_end, sents):
        for sent_idx, sent in enumerate(sents):
            sent_end = sent['start'] + len(sent['text']) - 1
            if e_start >= sent['start'] and e_end <= sent_end:
                return sent_idx
        return None

    pred_dict = {}
    related_dict = get_pred_related_info(pred_test_simi_file)
    with open(pred_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            sents = sample['sentences']
            events = sample['events']
            pred_labels = sample['pred_label']
            event_pairs = {}
            event_pair_idx = 0
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                e_i_sent_idx = _get_event_sent_idx(events[i]['start'], events[i]['end'], sents)
                assert e_i_sent_idx is not None
                e_i_arg_status = get_event_arg_status(related_dict[sample['doc_id']][e_i_start])
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    e_j_sent_idx = _get_event_sent_idx(events[j]['start'], events[j]['end'], sents)
                    assert e_j_sent_idx is not None
                    e_j_arg_status = get_event_arg_status(related_dict[sample['doc_id']][e_j_start])
                    assert e_i_start < e_j_start
                    event_pairs[f'{e_i_start}-{e_j_start}'] = {
                        'coref': pred_labels[event_pair_idx], 
                        'e_i_has_arg': e_i_arg_status['has_arg'], 'e_j_has_arg': e_j_arg_status['has_arg'], 
                        'e_i_find_arg': e_i_arg_status['find_arg'], 'e_j_find_arg': e_j_arg_status['find_arg'], 
                        'e_i_has_part': e_i_arg_status['has_part'], 'e_i_has_place': e_i_arg_status['has_place'], 
                        'e_j_has_part': e_j_arg_status['has_part'], 'e_j_has_place': e_j_arg_status['has_place'], 
                        'e_i_related_part': e_i_arg_status['related_part'], 'e_i_related_place': e_i_arg_status['related_place'], 
                        'e_j_related_part': e_j_arg_status['related_part'], 'e_j_related_place': e_j_arg_status['related_place'], 
                        'sent_dist': abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 
                        'e_i_link_len': 0, 'e_j_link_len': 0
                    }
                    event_pair_idx += 1
            assert event_pair_idx == len(pred_labels)
            pred_dict[sample['doc_id']] = event_pairs
    return pred_dict

def get_event_pair_set(gold_coref_file, gold_simi_coref_file, pred_coref_file, pred_simi_coref_file):

    gold_coref_results = get_gold_corefs(gold_coref_file, gold_simi_coref_file)
    pred_coref_results = get_pred_coref_results(pred_coref_file, pred_simi_coref_file)

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

    return new_gold_coref_results, new_pred_coref_results
