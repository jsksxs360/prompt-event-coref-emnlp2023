def clustering_greedy(events, pred_labels:list):
    '''
    As long as there is a pair of events coreference 
    between any two event chains, merge them.
    '''
    def create_coref_event_set(events, pred_labels):
        if len(events) > 1:
            assert len(pred_labels) == len(events) * (len(events) - 1) / 2
        event_pairs = [
            str(events[i]['start']) + '-' + str(events[j]['start'])
            for i in range(len(events) - 1) for j in range(i + 1, len(events))
        ]
        coref_event_pairs = [event_pair for event_pair, pred in zip(event_pairs, pred_labels) if pred == 1]
        coref_event_pair_set = set()
        for event_pair in coref_event_pairs:
            e1, e2 = event_pair.split('-')
            coref_event_pair_set.add(f'{e1}-{e2}')
            # coref_event_pair_set.add(f'{e2}-{e1}')
        return coref_event_pair_set
    
    def need_merge(set_1, set_2, coref_event_pair_set):
        for e1 in set_1:
            for e2 in set_2:
                if f'{e1}-{e2}' in coref_event_pair_set:
                    return True
        return False

    def find_merge_position(cluster_list, coref_event_pair_set):
        for i in range(len(cluster_list) - 1):
            for j in range(i + 1, len(cluster_list)):
                if need_merge(cluster_list[i], cluster_list[j], coref_event_pair_set):
                    return i, j
        return -1, -1
    
    coref_event_pair_set = create_coref_event_set(events, pred_labels)
    cluster_list = []
    for event in events: # init each link as an event
        cluster_list.append(set([event['start']]))
    while True:
        i, j = find_merge_position(cluster_list, coref_event_pair_set)
        if i == -1: # no cluster can be merged
            break
        cluster_list[i] |= cluster_list[j]
        del cluster_list[j]
    return cluster_list

def create_event_pairs_by_probs(events, pred_labels:list, pred_probs:list):
    '''
    connect each event with the other with the highest predicted coreference probability

    warning: this process will only create clusters of size 2!
    '''
    def create_coref_event_pair_dict(events, pred_labels, pred_probs):
        if len(events) > 1:
            assert len(pred_labels) == len(events) * (len(events) - 1) / 2
        event_pairs = [
            str(events[i]['start']) + '-' + str(events[j]['start'])
            for i in range(len(events) - 1) for j in range(i + 1, len(events))
        ]
        coref_event_pairs = [
            (event_pair, prob) for event_pair, pred, prob in zip(event_pairs, pred_labels, pred_probs) if pred == 1
        ]
        coref_event_pair_dict = {}
        for event_pair, prob in coref_event_pairs:
            e1, e2 = event_pair.split('-')
            coref_event_pair_dict[f'{e1}-{e2}'] = prob
            coref_event_pair_dict[f'{e2}-{e1}'] = prob
        return coref_event_pair_dict
    
    def find_highest_coref_event(event_start, other_event_starts, coref_event_pair_dict):
        max_prob, event_idx = 0., -1
        for idx, other_event in enumerate(other_event_starts):
            event_pair = f'{event_start}-{other_event}'
            if event_pair in coref_event_pair_dict and coref_event_pair_dict[event_pair] > max_prob:
                max_prob = coref_event_pair_dict[event_pair]
                event_idx = idx
        return event_idx

    temp_events = events.copy() # prevent modification of the event list
    # create coref event-pair dict
    coref_event_pair_dict = create_coref_event_pair_dict(temp_events, pred_labels, pred_probs)
    results, check_event_num = [], 0
    for idx, event in enumerate(temp_events):
        event_start = str(event['start'])
        other_event_starts = [str(e['start']) for e in temp_events[idx+1:]]
        event_idx = find_highest_coref_event(event_start, other_event_starts, coref_event_pair_dict)
        if event_idx == -1:
            results.append([event])
            check_event_num += 1
        else:
            results.append([event, temp_events[idx + 1 + event_idx]])
            check_event_num += 2
            del temp_events[idx + 1 + event_idx]
    assert check_event_num == len(events)
    return results
