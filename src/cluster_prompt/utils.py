from collections import defaultdict, Counter

def create_event_simi_dict(event_pairs_id:list, event_pairs_cos:list, clusters:list) -> dict:
    '''
    create event similarity dict (sort by similarity in descending order)
    format: {e1_id: [{'id': e2_id, 'cos': cos, 'coref': coref}]}
    '''

    def get_event_cluster_id(event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        raise ValueError(f'Unknown event_id: {event_id}')

    simi_dict = defaultdict(list)
    for id_pair, cos in zip(event_pairs_id, event_pairs_cos):
        e1_id, e2_id = id_pair.split('###')
        coref = 1 if get_event_cluster_id(e1_id, clusters) == get_event_cluster_id(e2_id, clusters) else 0
        simi_dict[e1_id].append({'id': e2_id, 'cos': cos, 'coref': coref})
        simi_dict[e2_id].append({'id': e1_id, 'cos': cos, 'coref': coref})
    for simi_list in simi_dict.values():
        simi_list.sort(key=lambda x:x['cos'], reverse=True)
    return simi_dict

def create_fake_cluster(cluster_events, simi_dict, events_dict, fake_cluster_k):
    '''
    create fake cluster by replacing events with similar non-coreference events

    - cluster_events: source cluster
    - simi_dict: event similarity dict (sort by similarity in descending order)
    - events_dict: {event_id: event}
    - fake_cluster_k: number of fake cluster
    '''

    def get_noncoref_ids(simi_list):
        noncoref_ids = []
        for simi in simi_list:
            if simi['coref'] == 0:
                noncoref_ids.append(simi['id'])
        return noncoref_ids

    noncoref_event_ids = {
        e['event_id']:get_noncoref_ids(simi_dict[e['event_id']]) for e in cluster_events
    }
    fake_clusters, saved_fake_clusters = [], []
    for k in range(fake_cluster_k):
        fake_cluster = []
        for e in cluster_events:
            noncoref_events = noncoref_event_ids[e['event_id']]
            if k >= len(noncoref_events):
                break
            for e_id in noncoref_events[k:]:
                if e_id not in fake_cluster:
                    fake_cluster.append(e_id)
                    break
        if len(fake_cluster) == len(cluster_events):
            fake_cluster_set = set(fake_cluster)
            if fake_cluster_set in saved_fake_clusters: # filter same cluster
                continue
            fake_clusters.append([events_dict[e_id] for e_id in fake_cluster])
            saved_fake_clusters.append(fake_cluster_set)
    return fake_clusters

get_all_events_in_cluster = lambda event_list, cluster: [event for event in event_list if event['event_id'] in cluster]

def create_new_sent(
    cluster1_events:list, cluster2_events:list, 
    sentences:list, sentences_lengths:list, 
    special_token_dict:dict, tokenizer, max_length:int
    ):
    '''
    create segment contains event mentions from two clusters

    - cluster1_events, cluster2_events: cluster that contains events, [
        {'sent_idx': sentence idx, 'trigger': trigger word, 'sent_start': trigger offset in the sentence}, ...
    ]
    - sentences: all the sentences (objects)
    - sentences_lengths: sentence lengths

    Return:
    {
        'sent': new segment contains all the event mentions, 
        'event_s_e_offset': event_s_e_offsets, 
        'cluster1_trigger': trigger1, 
        'cluster2_trigger': trigger2
    }
    '''

    def choose_sent_idxs(cluster1_events:list, cluster2_events:list, sent_lengths:list, max_length:int) -> set:
        '''
        choose event sentences to control the total length, prefer sentences that contain events from both clusters
        
        warning: some events may be dropped by this operation

        - cluster1_events, cluster2_events: cluster that contains events
        - sent_lengths: sentence lengths
        - max_length: max total length
        '''

        sent_event_num = {} # {sent_idx: event number}
        for e in cluster1_events + cluster2_events:
            if e['sent_idx'] not in sent_event_num:
                sent_event_num[e['sent_idx']] = 1
            else:
                sent_event_num[e['sent_idx']] += 1
        c1_sent_idxs, c2_sent_idxs = set([e['sent_idx'] for e in cluster1_events]), set([e['sent_idx'] for e in cluster2_events])
        c1_and_c2_sent_idxs = c1_sent_idxs & c2_sent_idxs
        c1_sent_idxs, c2_sent_idxs = sorted(list(c1_sent_idxs - c1_and_c2_sent_idxs)), sorted(list(c2_sent_idxs - c1_and_c2_sent_idxs))
        chosen_sent_idx = set()
        length = 0
        check_c1, check_c2 = False, False # whether events in the cluster are included
        # sentences that contain events from both clusters
        for sent_idx in c1_and_c2_sent_idxs:
            if length + sent_lengths[sent_idx] + sent_event_num[sent_idx] * 4 > max_length:
                break
            chosen_sent_idx.add(sent_idx)
            length += sent_lengths[sent_idx] + sent_event_num[sent_idx] * 4
            check_c1 = True
            check_c2 = True
        # alternately add event sentences in two clusters
        p1, p2 = 0, 0
        while p1 < len(c1_sent_idxs) and p2 < len(c2_sent_idxs):
            if length + sent_lengths[c1_sent_idxs[p1]] + sent_event_num[c1_sent_idxs[p1]] * 4 > max_length:
                break
            chosen_sent_idx.add(c1_sent_idxs[p1])
            length += sent_lengths[c1_sent_idxs[p1]] + sent_event_num[c1_sent_idxs[p1]] * 4
            check_c1 = True
            p1 += 1
            if length + sent_lengths[c2_sent_idxs[p2]] + sent_event_num[c2_sent_idxs[p2]] * 4 > max_length:
                break
            chosen_sent_idx.add(c2_sent_idxs[p2])
            length += sent_lengths[c2_sent_idxs[p2]] + sent_event_num[c2_sent_idxs[p2]] * 4
            check_c2 = True
            p2 += 1
        # add rest event sentences
        for sent_idx in c1_sent_idxs[p1:len(c1_sent_idxs)]:
            if length + sent_lengths[sent_idx] + sent_event_num[sent_idx] * 4 > max_length:
                break
            chosen_sent_idx.add(sent_idx)
            length += sent_lengths[sent_idx] + sent_event_num[sent_idx] * 4
            check_c1 = True
        for sent_idx in c2_sent_idxs[p2:len(c2_sent_idxs)]:
            if length + sent_lengths[sent_idx] + sent_event_num[sent_idx] * 4 > max_length:
                break
            chosen_sent_idx.add(sent_idx)
            length += sent_lengths[sent_idx] + sent_event_num[sent_idx] * 4
            check_c2 = True
        # assert check_c1 and check_c2
        if not (check_c1 and check_c2):
            return None
        chosen_event_num = sum([1 for e in cluster1_events + cluster2_events if e['sent_idx'] in chosen_sent_idx])
        if chosen_event_num < 3: # at least 3 events should be chosen
            return None
        return chosen_sent_idx

    def get_sen_with_events(sentence:str, cluster1_events, cluster2_events):
        '''add special label for triggers in the sentence

        return: 
        - new_sen: new segment that contains all the triggers with special labels
        - new_event_offsets: [e_s_offset, e_e_offset]
        '''
        all_events = []
        all_events += [{'offset': event['sent_start'], 'trigger': event['trigger'], 'cluster': 1} for event in cluster1_events]
        all_events += [{'offset': event['sent_start'], 'trigger': event['trigger'], 'cluster': 2} for event in cluster2_events]
        all_events.sort(key=lambda x:x['offset'])
        new_sen, start_p = '', 0
        new_event_offsets = []
        for event in all_events:
            new_sen += sentence[start_p:event['offset']]
            new_event_offsets.append([
                len(new_sen), len(new_sen) + (len(e1s) if event['cluster'] == 1 else len(e2s)) + len(event['trigger'])
            ])
            new_sen += (e1s + event['trigger'] + e1e) if event['cluster'] == 1 else (e2s + event['trigger'] + e2e)
            start_p = event['offset'] + len(event['trigger'])
        new_sen += sentence[start_p:]
        for event, [s_offset, _] in zip(all_events, new_event_offsets): # check
            e_s, e_e = (e1s, e1e) if event['cluster'] == 1 else (e2s, e2e)
            event_span = e_s + event['trigger'] + e_e
            assert new_sen[s_offset:s_offset+len(event_span)] == event_span
        return new_sen, new_event_offsets
    
    e1s, e1e, e2s, e2e = special_token_dict['e1s_token'], special_token_dict['e1e_token'], special_token_dict['e2s_token'], special_token_dict['e2e_token']
    # choose event sentences to control the total length
    chosen_sent_idxs = choose_sent_idxs(cluster1_events, cluster2_events, sentences_lengths, max_length)
    if not chosen_sent_idxs:
        return None
    sentence_event = {}
    trigger1, trigger2 = [], []
    for e in cluster1_events:
        if e['sent_idx'] not in chosen_sent_idxs:
            continue
        trigger1.append(e['trigger'])
        if e['sent_idx'] not in sentence_event:
            sentence_event[e['sent_idx']] = {
                'text': sentences[e['sent_idx']]['text'], 
                'cluster1_events': [e], 
                'cluster2_events': []
            }
        else:
            sentence_event[e['sent_idx']]['cluster1_events'].append(e)
    for e in cluster2_events:
        if e['sent_idx'] not in chosen_sent_idxs:
            continue
        trigger2.append(e['trigger'])
        if e['sent_idx'] not in sentence_event:
            sentence_event[e['sent_idx']] = {
                'text': sentences[e['sent_idx']]['text'], 
                'cluster1_events': [], 
                'cluster2_events': [e]
            }
        else:
            sentence_event[e['sent_idx']]['cluster2_events'].append(e)
    # select word with the largest number in the cluster as representative trigger
    trigger1, trigger2 = Counter(trigger1).most_common()[0][0], Counter(trigger2).most_common()[0][0]
    sentence_event = sorted(sentence_event.items(), key=lambda x:x[0])
    document, event_s_e_offsets = '', []
    for _, s_e in sentence_event:
        new_sen, new_event_offsets = get_sen_with_events(s_e['text'], s_e['cluster1_events'], s_e['cluster2_events'])
        event_s_e_offsets += [[s+len(document), e+len(document)] for s, e in new_event_offsets]
        document += new_sen + ' '
    
    document_length = len(tokenizer(document).tokens())
    assert document_length <= max_length, f'max length value should >= {document_length}. Now is {max_length}.'
    return {
        'sent': document, 
        'event_s_e_offset': event_s_e_offsets, 
        'cluster1_trigger': trigger1, 
        'cluster2_trigger': trigger2
    }

def get_prompt(
    prompt_type:str, special_token_dict:dict, source_sent:str, 
    e1_trigger:str, e2_trigger:str, event_s_e_offset:list, 
    tokenizer
    ):
    '''
    create different type prompts

    - prompt_type: \n
        'hb_d', 'd_hb'  # hard base template \n
        'hq_d', 'd_hq'  # hard question-style template \n
        'sb_d', 'd_sb'  # soft base template
    '''

    e1s_token, e1e_token = special_token_dict['e1s_token'], special_token_dict['e1e_token']
    e2s_token, e2e_token = special_token_dict['e2s_token'], special_token_dict['e2e_token']
    mask_token = special_token_dict['mask_token']
    l_token1, l_token2, l_token3 = special_token_dict['l_token1'], special_token_dict['l_token2'], special_token_dict['l_token3']
    l_token4, l_token5, l_token6 = special_token_dict['l_token4'], special_token_dict['l_token5'], special_token_dict['l_token6']

    if 'hb' in prompt_type: # hard base template
        prompt = f'In this document, the {e1s_token} {e1_trigger} {e1e_token} event and the {e2s_token} {e2_trigger} {e2e_token} event refer to {mask_token} event. '
    elif 'hq' in prompt_type: # hard question-style template
        prompt = f'In this document, the {e1s_token} {e1_trigger} {e1e_token} event and the {e2s_token} {e2_trigger} {e2e_token} event refer to the same event? {mask_token}. '
    elif 'sb' in prompt_type: # soft base template
        prompt = f'In this document, {l_token1} {e1s_token} {e1_trigger} {e1e_token} {l_token2} {l_token3} {e2s_token} {e2_trigger} {e2e_token} {l_token4} {l_token5} {mask_token} {l_token6}. '
    
    if '_d' in prompt_type: # template + document
        event_s_e_offset = [[e_s + len(prompt), e_e + len(prompt)] for e_s, e_e in event_s_e_offset]
        prompt += source_sent
    elif 'd_' in prompt_type: # document + template
        prompt = source_sent + ' ' + prompt
    
    for e_s, e_e in event_s_e_offset: # check offset
        assert prompt[e_s:e_s + len(e1s_token)] == e1s_token or prompt[e_s:e_s + len(e2s_token)] == e2s_token
        assert prompt[e_e:e_e + len(e1e_token)] == e1e_token or prompt[e_e:e_e + len(e2e_token)] == e2e_token
    # convert char offsets to token idxs
    encoding = tokenizer(prompt)
    mask_idx = encoding.char_to_token(prompt.find(mask_token))
    assert mask_idx is not None
    event_s_e_idxs = []
    for e_s, e_e in event_s_e_offset:
        e_s_idx, e_e_idx = encoding.char_to_token(e_s), encoding.char_to_token(e_e)
        assert e_s_idx is not None and e_e_idx is not None
        event_s_e_idxs.append([e_s_idx, e_e_idx])
    
    return {
        'prompt': prompt, 
        'mask_idx': mask_idx, 
        'event_idx': event_s_e_idxs
    }