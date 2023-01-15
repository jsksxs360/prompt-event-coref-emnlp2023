def clustering_greedy(events, pred_labels:list):
    '''
    As long as there is a pair of events coreference 
    between any two event chains, merge them.
    '''
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