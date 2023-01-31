import numpy as np

def findall(p, s):
    '''yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def create_new_sent(
    sent1_idx:int, e1_sent_start:int, e1_trigger:str, 
    sent2_idx:int, e2_sent_start:int, e2_trigger:str, 
    sents:list, sents_lens:list, sents_entities:list, 
    special_token_dict:dict, context_k:int, max_length:int, tokenizer
    ):
    '''
    create segment contains two event mentions

    format: xxx [E1_START] trigger1 [E1_END] xxx [E2_START] trigger2 [E2_END] xxx

    Return:
    {
        'new_sent': new segment contains two event mentions, 
        'e1_trigger': trigger1, 
        'e1_sent_start': trigger1 offset, 
        'e1s_sent_start': E1_START offset, 
        'e1e_sent_start': E1_END offset, 
        'e1_entities': entities in sentence host event1, [{'offset': entity offset, 'text': entity}]
        'e2_trigger': trigger2, 
        'e2_sent_start': trigger2 offset, 
        'e2s_sent_start': E2_START offset, 
        'e2e_sent_start': E2_END offset, 
        'e2_entities': entities in sentence host event2, [{'offset': entity offset, 'text': entity}]
    }
    '''

    context_before_list =  [sent['text'] for sent in sents[sent1_idx - context_k if sent1_idx >= context_k else 0 : sent1_idx]]
    context_before_lengths = [sent_len for sent_len in sents_lens[sent1_idx - context_k if sent1_idx >= context_k else 0 : sent1_idx]]
    context_next_list = [sent['text'] for sent in sents[sent2_idx + 1 : sent2_idx + context_k + 1 if sent2_idx + context_k < len(sents) else len(sents)]]
    context_next_lengths = [sent_len for sent_len in sents_lens[sent2_idx + 1 : sent2_idx + context_k + 1 if sent2_idx + context_k < len(sents) else len(sents)]]

    e1s, e1e, e2s, e2e = special_token_dict['e1s_token'], special_token_dict['e1e_token'], special_token_dict['e2s_token'], special_token_dict['e2e_token']

    if sent1_idx == sent2_idx: # two events in the same sentence
        assert e1_sent_start < e2_sent_start
        sent_text = sents[sent1_idx]['text']
        event_entities = sents_entities[sent1_idx]
        for ent in event_entities:
            assert ent in sent_text
        total_length = sents_lens[sent1_idx] + 8
        before, middle, next = sent_text[:e1_sent_start], sent_text[e1_sent_start + len(e1_trigger):e2_sent_start], sent_text[e2_sent_start + len(e2_trigger):]
        new_sent_before, new_sent_middle, new_sent_next = f'{before}{e1s}', f'{e1e}{middle}{e2s}', f'{e2e}{next}'
        for before_sen, sen_len in zip(context_before_list[-1::-1], context_before_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent_before = before_sen + ' ' + new_sent_before
        for next_sen, sen_len in zip(context_next_list[-1::-1], context_next_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent_next += ' ' + next_sen
        new_sent = f'{new_sent_before}{e1_trigger}{new_sent_middle}{e2_trigger}{new_sent_next}'
        
        e1_new_sent_start = len(new_sent_before)
        e1s_new_sent_start = e1_new_sent_start - len(e1s)
        e1e_new_sent_start = e1_new_sent_start + len(e1_trigger)
        e2_new_sent_start = len(new_sent_before) + len(e1_trigger) + len(new_sent_middle)
        e2s_new_sent_start = e2_new_sent_start - len(e2s)
        e2e_new_sent_start = e2_new_sent_start + len(e2_trigger)
        event_entities = []
        for entity in event_entities:
            for offset in findall(entity, new_sent):
                event_entities.append({'offset': offset, 'text': entity})
        event_entities.sort(key=lambda x:x['offset'])
        
        assert new_sent[e1_new_sent_start:e1_new_sent_start + len(e1_trigger)] == e1_trigger
        assert new_sent[e1s_new_sent_start:e1s_new_sent_start + len(e1s)] == e1s
        assert new_sent[e1e_new_sent_start:e1e_new_sent_start + len(e1e)] == e1e
        assert new_sent[e2_new_sent_start:e2_new_sent_start + len(e2_trigger)] == e2_trigger
        assert new_sent[e2s_new_sent_start:e2s_new_sent_start + len(e2s)] == e2s
        assert new_sent[e2e_new_sent_start:e2e_new_sent_start + len(e2e)] == e2e
        for entity in event_entities:
            assert new_sent[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text']
        return {
            'new_sent': new_sent, 
            'e1_trigger': e1_trigger, 
            'e1_sent_start': e1_new_sent_start, 
            'e1s_sent_start': e1s_new_sent_start, 
            'e1e_sent_start': e1e_new_sent_start, 
            'e1_entities': event_entities if event_entities else [{'offset': e1_new_sent_start, 'text': e1_trigger}], 
            'e2_trigger': e2_trigger, 
            'e2_sent_start': e2_new_sent_start, 
            'e2s_sent_start': e2s_new_sent_start, 
            'e2e_sent_start': e2e_new_sent_start, 
            'e2_entities': event_entities if event_entities else [{'offset': e2_new_sent_start, 'text': e2_trigger}]
        }
    else: # events in different sentence
        sent1_text, sent2_text = sents[sent1_idx]['text'], sents[sent2_idx]['text']
        event1_entities, event2_entities = sents_entities[sent1_idx], sents_entities[sent2_idx]
        for ent in event1_entities:
            assert ent in sent1_text
        for ent in event2_entities:
            assert ent in sent2_text
        before_1, next_1 = sent1_text[:e1_sent_start], sent1_text[e1_sent_start + len(e1_trigger):]
        before_2, next_2 = sent2_text[:e2_sent_start], sent2_text[e2_sent_start + len(e2_trigger):]
        total_length = sents_lens[sent1_idx] + sents_lens[sent2_idx] + 8
        if total_length > max_length:
            span_length = (max_length - 80) // 4
            before_1 = ' '.join([c for c in before_1.split(' ') if c != ''][-span_length:]).strip()
            next_1 = ' '.join([c for c in next_1.split(' ') if c != ''][:span_length]).strip()
            before_2 = ' '.join([c for c in before_2.split(' ') if c != ''][-span_length:]).strip()
            next_2 = ' '.join([c for c in next_2.split(' ') if c != ''][:span_length]).strip()
            total_length = len(tokenizer(f'{before_1} {e1_trigger} {next_1} {before_2} {e2_trigger} {next_2}').tokens()) + 8
        new_sent1_before, new_sent1_next = f'{before_1}{e1s}', f'{e1e}{next_1}'
        new_sent2_before, new_sent2_next = f'{before_2}{e2s}', f'{e2e}{next_2}'
        for before_sen, sen_len in zip(context_before_list[-1::-1], context_before_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent1_before = before_sen + ' ' + new_sent1_before
        for next_sen, sen_len in zip(context_next_list[-1::-1], context_next_lengths[-1::-1]):
            if total_length + sen_len > max_length:
                break
            else:
                total_length += sen_len
                new_sent2_next += ' ' + next_sen
        new_sent1 = f'{new_sent1_before}{e1_trigger}{new_sent1_next}'
        new_sent2 = f'{new_sent2_before}{e2_trigger}{new_sent2_next}'
        
        e1_new_sent_start = len(new_sent1_before)
        e1s_new_sent_start = e1_new_sent_start - len(e1s)
        e1e_new_sent_start = e1_new_sent_start + len(e1_trigger)
        e2_new_sent_start = len(new_sent2_before)
        e2s_new_sent_start = e2_new_sent_start - len(e2s)
        e2e_new_sent_start = e2_new_sent_start + len(e2_trigger)
        # add sentences between new_sent1 and new_sent2
        p, q = sent1_idx + 1, sent2_idx - 1
        while p <= q:
            if p == q:
                if total_length + sents_lens[p] <= max_length:
                    new_sent1 += (' ' + sents[p]['text'])
                final_new_sent = new_sent1 + ' ' + new_sent2
                e2_new_sent_start += len(new_sent1) + 1
                e2s_new_sent_start += len(new_sent1) + 1
                e2e_new_sent_start += len(new_sent1) + 1
                e1_entities, e2_entities = [], []
                for entity in event1_entities:
                    for offset in findall(entity, final_new_sent):
                        e1_entities.append({'offset': offset, 'text': entity})
                e1_entities.sort(key=lambda x:x['offset'])
                for entity in event2_entities:
                    for offset in findall(entity, final_new_sent):
                        e2_entities.append({'offset': offset, 'text': entity})
                e2_entities.sort(key=lambda x:x['offset'])
                
                assert final_new_sent[e1_new_sent_start:e1_new_sent_start + len(e1_trigger)] == e1_trigger
                assert final_new_sent[e1s_new_sent_start:e1s_new_sent_start + len(e1s)] == e1s
                assert final_new_sent[e1e_new_sent_start:e1e_new_sent_start + len(e1e)] == e1e
                assert final_new_sent[e2_new_sent_start:e2_new_sent_start + len(e2_trigger)] == e2_trigger
                assert final_new_sent[e2s_new_sent_start:e2s_new_sent_start + len(e2s)] == e2s
                assert final_new_sent[e2e_new_sent_start:e2e_new_sent_start + len(e2e)] == e2e
                for entity in e1_entities:
                    assert final_new_sent[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text']
                for entity in e2_entities:
                    assert final_new_sent[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text']
                return {
                    'new_sent': final_new_sent, 
                    'e1_trigger': e1_trigger, 
                    'e1_sent_start': e1_new_sent_start, 
                    'e1s_sent_start': e1s_new_sent_start, 
                    'e1e_sent_start': e1e_new_sent_start, 
                    'e1_entities': e1_entities if e1_entities else [{'offset': e1_new_sent_start, 'text': e1_trigger}], 
                    'e2_trigger': e2_trigger, 
                    'e2_sent_start': e2_new_sent_start, 
                    'e2s_sent_start': e2s_new_sent_start, 
                    'e2e_sent_start': e2e_new_sent_start, 
                    'e2_entities': e2_entities if e2_entities else [{'offset': e2_new_sent_start, 'text': e2_trigger}], 
                }
            if total_length + sents_lens[p] > max_length:
                break
            else:
                total_length += sents_lens[p]
                new_sent1 += (' ' + sents[p]['text'])
            if total_length + sents_lens[q] > max_length:
                break
            else:
                total_length += sents_lens[q]
                new_sent2 = sents[q]['text'] + ' ' + new_sent2
                e2_new_sent_start += len(sents[q]['text']) + 1
                e2s_new_sent_start += len(sents[q]['text']) + 1
                e2e_new_sent_start += len(sents[q]['text']) + 1
            p += 1
            q -= 1
        final_new_sent = new_sent1 + ' ' + new_sent2
        e2_new_sent_start += len(new_sent1) + 1
        e2s_new_sent_start += len(new_sent1) + 1
        e2e_new_sent_start += len(new_sent1) + 1
        e1_entities, e2_entities = [], []
        for entity in event1_entities:
            for offset in findall(entity, final_new_sent):
                e1_entities.append({'offset': offset, 'text': entity})
        e1_entities.sort(key=lambda x:x['offset'])
        for entity in event2_entities:
            for offset in findall(entity, final_new_sent):
                e2_entities.append({'offset': offset, 'text': entity})
        e2_entities.sort(key=lambda x:x['offset'])

        assert final_new_sent[e1s_new_sent_start:e1s_new_sent_start + len(e1s)] == e1s
        assert final_new_sent[e1e_new_sent_start:e1e_new_sent_start + len(e1e)] == e1e
        assert final_new_sent[e2s_new_sent_start:e2s_new_sent_start + len(e2s)] == e2s
        assert final_new_sent[e2e_new_sent_start:e2e_new_sent_start + len(e2e)] == e2e
        assert final_new_sent[e1_new_sent_start:e1_new_sent_start + len(e1_trigger)] == e1_trigger
        assert final_new_sent[e2_new_sent_start:e2_new_sent_start + len(e2_trigger)] == e2_trigger
        for entity in e1_entities:
            assert final_new_sent[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text']
        for entity in e2_entities:
            assert final_new_sent[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text']
        return {
            'new_sent': final_new_sent, 
            'e1_trigger': e1_trigger, 
            'e1_sent_start': e1_new_sent_start, 
            'e1s_sent_start': e1s_new_sent_start, 
            'e1e_sent_start': e1e_new_sent_start, 
            'e1_entities': e1_entities if e1_entities else [{'offset': e1_new_sent_start, 'text': e1_trigger}], 
            'e2_trigger': e2_trigger, 
            'e2_sent_start': e2_new_sent_start, 
            'e2s_sent_start': e2s_new_sent_start, 
            'e2e_sent_start': e2e_new_sent_start, 
            'e2_entities': e2_entities if e2_entities else [{'offset': e2_new_sent_start, 'text': e2_trigger}], 
        }

def get_prompt(
    prompt_type:str, special_token_dict:dict, source_sent:str, 
    e1_trigger:str, e1_start:int, e1s_start:int, e1e_start:int, e1_entities:list, 
    e2_trigger:str, e2_start:int, e2s_start:int, e2e_start:int, e2_entities:list, 
    tokenizer
    ):
    '''
    create different type prompts

    - prompt_type: \n
        'hb_d', 'd_hb', 'hbe_d', 'd_hbe', # hard base template\n
        'hq_d', 'd_hq', 'hqe_d', 'd_hqe', # hard question-style template\n
        'sb_d', 'd_sb', 'sbe_d', 'd_sbe'  # soft base template
    '''
    
    e1s_token, e1e_token, e1_arg_token = special_token_dict['e1s_token'], special_token_dict['e1e_token'], special_token_dict['e1_arg_token']
    e2s_token, e2e_token, e2_arg_token = special_token_dict['e2s_token'], special_token_dict['e2e_token'], special_token_dict['e2_arg_token']
    mask_token = special_token_dict['mask_token']
    l_token1, l_token2, l_token3 = special_token_dict['l_token1'], special_token_dict['l_token2'], special_token_dict['l_token3']
    l_token4, l_token5, l_token6 = special_token_dict['l_token4'], special_token_dict['l_token5'], special_token_dict['l_token6']

    tmp_e1s_start, tmp_e1e_start, tmp_e2s_start, tmp_e2e_start = 0, 0, 0, 0
    tmp_p1_start, tmp_p2_start = 0, 0
    if 'hb' in prompt_type: # hard base template
        if 'e' in prompt_type: # with entities (participants)
            s1, s2, s3, s4, s5, s6, s7 = (
                'In this document, the ',  
                f'{e1s_token}{e1_trigger}', 
                f'{e1e_token} event with ', 
                f'{e1_arg_token} as participants and the ', 
                f'{e2s_token}{e2_trigger}', 
                f'{e2e_token} event with ', 
                f'{e2_arg_token} as participants refer to {mask_token} event. '
            )
            tmp_e1s_start = len(s1)
            tmp_e1e_start = tmp_e1s_start + len(s2)
            tmp_p1_start = tmp_e1e_start + len(s3)
            tmp_e2s_start = tmp_p1_start + len(s4)
            tmp_e2e_start = tmp_e2s_start + len(s5)
            tmp_p2_start = tmp_e2e_start + len(s6)
            tmp_e1_start, tmp_e2_start = tmp_e1s_start + len(e1s_token), tmp_e2s_start + len(e2s_token)
            prompt = s1 + s2 + s3 + s4 + s5 + s6 + s7
            assert prompt == (
                f'In this document, the {e1s_token}{e1_trigger}{e1e_token} event with {e1_arg_token} as participants '
                f'and the {e2s_token}{e2_trigger}{e2e_token} event with {e2_arg_token} as participants refer to {mask_token} event. '
            )
        else:
            s1, s2, s3, s4, s5 = (
                'In this document, the ', 
                f'{e1s_token}{e1_trigger}', 
                f'{e1e_token} event and the ', 
                f'{e2s_token}{e2_trigger}', 
                f'{e2e_token} event refer to {mask_token} event. '
            )
            tmp_e1s_start = len(s1)
            tmp_e1e_start = tmp_e1s_start + len(s2)
            tmp_e2s_start = tmp_e1e_start + len(s3)
            tmp_e2e_start = tmp_e2s_start + len(s4)
            tmp_e1_start, tmp_e2_start = tmp_e1s_start + len(e1s_token), tmp_e2s_start + len(e2s_token)
            prompt = s1 + s2 + s3 + s4 + s5
            assert prompt == (
                f'In this document, the {e1s_token}{e1_trigger}{e1e_token} event and '
                f'the {e2s_token}{e2_trigger}{e2e_token} event refer to {mask_token} event. '
            )
    elif 'hq' in prompt_type: # hard question-style template
        if 'e' in prompt_type: # with entities (participants)
            s1, s2, s3, s4, s5, s6, s7 = (
                'In this document, the ',  
                f'{e1s_token}{e1_trigger}', 
                f'{e1e_token} event with ', 
                f'{e1_arg_token} as participants and the ', 
                f'{e2s_token}{e2_trigger}', 
                f'{e2e_token} event with ', 
                f'{e2_arg_token} as participants refer to the same event? {mask_token}. '
            )
            tmp_e1s_start = len(s1)
            tmp_e1e_start = tmp_e1s_start + len(s2)
            tmp_p1_start = tmp_e1e_start + len(s3)
            tmp_e2s_start = tmp_p1_start + len(s4)
            tmp_e2e_start = tmp_e2s_start + len(s5)
            tmp_p2_start = tmp_e2e_start + len(s6)
            tmp_e1_start, tmp_e2_start = tmp_e1s_start + len(e1s_token), tmp_e2s_start + len(e2s_token)
            prompt = s1 + s2 + s3 + s4 + s5 + s6 + s7
            assert prompt == (
                f'In this document, the {e1s_token}{e1_trigger}{e1e_token} event with {e1_arg_token} as participants '
                f'and the {e2s_token}{e2_trigger}{e2e_token} event with {e2_arg_token} as participants refer to the same event? {mask_token}. '
            )
        else:
            s1, s2, s3, s4, s5 = (
                'In this document, the ', 
                f'{e1s_token}{e1_trigger}', 
                f'{e1e_token} event and the ', 
                f'{e2s_token}{e2_trigger}', 
                f'{e2e_token} event refer to the same event? {mask_token}. '
            )
            tmp_e1s_start = len(s1)
            tmp_e1e_start = tmp_e1s_start + len(s2)
            tmp_e2s_start = tmp_e1e_start + len(s3)
            tmp_e2e_start = tmp_e2s_start + len(s4)
            tmp_e1_start, tmp_e2_start = tmp_e1s_start + len(e1s_token), tmp_e2s_start + len(e2s_token)
            prompt = s1 + s2 + s3 + s4 + s5
            assert prompt == (
                f'In this document, the {e1s_token}{e1_trigger}{e1e_token} event and '
                f'the {e2s_token}{e2_trigger}{e2e_token} event refer to the same event? {mask_token}. '
            )
    elif 'sb' in prompt_type: # soft base template
        if 'e' in prompt_type: # with entities (participants)
            s1, s2, s3, s4, s5, s6, s7 = (
                f'In this document, {l_token1} ',  
                f'{e1s_token}{e1_trigger}', 
                f'{e1e_token} {l_token2} ', 
                f'{e1_arg_token} {l_token3} ', 
                f'{e2s_token}{e2_trigger}', 
                f'{e2e_token} {l_token4} ', 
                f'{e2_arg_token} {l_token5} {mask_token} {l_token6}. '
            )
            tmp_e1s_start = len(s1)
            tmp_e1e_start = tmp_e1s_start + len(s2)
            tmp_p1_start = tmp_e1e_start + len(s3)
            tmp_e2s_start = tmp_p1_start + len(s4)
            tmp_e2e_start = tmp_e2s_start + len(s5)
            tmp_p2_start = tmp_e2e_start + len(s6)
            tmp_e1_start, tmp_e2_start = tmp_e1s_start + len(e1s_token), tmp_e2s_start + len(e2s_token)
            prompt = s1 + s2 + s3 + s4 + s5 + s6 + s7
            assert prompt == (
                f'In this document, {l_token1} {e1s_token}{e1_trigger}{e1e_token} {l_token2} {e1_arg_token} '
                f'{l_token3} {e2s_token}{e2_trigger}{e2e_token} {l_token4} {e2_arg_token} {l_token5} {mask_token} {l_token6}. '
            )
        else:
            s1, s2, s3, s4, s5 = (
                f'In this document, {l_token1} ', 
                f'{e1s_token}{e1_trigger}', 
                f'{e1e_token} {l_token2} {l_token3} ', 
                f'{e2s_token}{e2_trigger}', 
                f'{e2e_token} {l_token4} {l_token5} {mask_token} {l_token6}. '
            )
            tmp_e1s_start = len(s1)
            tmp_e1e_start = tmp_e1s_start + len(s2)
            tmp_e2s_start = tmp_e1e_start + len(s3)
            tmp_e2e_start = tmp_e2s_start + len(s4)
            tmp_e1_start, tmp_e2_start = tmp_e1s_start + len(e1s_token), tmp_e2s_start + len(e2s_token)
            prompt = s1 + s2 + s3 + s4 + s5
            assert prompt == (
                f'In this document, {l_token1} {e1s_token}{e1_trigger}{e1e_token} {l_token2} '
                f'{l_token3} {e2s_token}{e2_trigger}{e2e_token} {l_token4} {l_token5} {mask_token} {l_token6}. '
            )
    
    if '_d' in prompt_type: # template + document
        e1_start, e1s_start, e1e_start = np.asarray([e1_start, e1s_start, e1e_start]) + np.full((3,), len(prompt))
        e2_start, e2s_start, e2e_start = np.asarray([e2_start, e2s_start, e2e_start]) + np.full((3,), len(prompt))
        e1_entities = [{'offset': entity['offset'] + len(prompt), 'text': entity['text']} for entity in e1_entities]
        e2_entities = [{'offset': entity['offset'] + len(prompt), 'text': entity['text']} for entity in e2_entities]
        prompt += source_sent
    elif 'd_' in prompt_type: # document + template
        tmp_e1_start, tmp_e1s_start, tmp_e1e_start, tmp_p1_start = \
            np.asarray([tmp_e1_start, tmp_e1s_start, tmp_e1e_start, tmp_p1_start]) + np.full((4,), len(source_sent) + 1)
        tmp_e2_start, tmp_e2s_start, tmp_e2e_start, tmp_p2_start = \
            np.asarray([tmp_e2_start, tmp_e2s_start, tmp_e2e_start, tmp_p2_start]) + np.full((4,), len(source_sent) + 1)
        prompt = source_sent + ' ' + prompt

    assert prompt[tmp_e1_start:tmp_e1_start + len(e1_trigger)] == e1_trigger
    assert prompt[tmp_e1s_start:tmp_e1s_start + len(e1s_token)] == e1s_token
    assert prompt[tmp_e1e_start:tmp_e1e_start + len(e1e_token)] == e1e_token
    assert prompt[tmp_e2_start:tmp_e2_start + len(e2_trigger)] == e2_trigger
    assert prompt[tmp_e2s_start:tmp_e2s_start + len(e2s_token)] == e2s_token
    assert prompt[tmp_e2e_start:tmp_e2e_start + len(e2e_token)] == e2e_token
    if 'e' in prompt_type:
        assert prompt[tmp_p1_start:tmp_p1_start + len(e1_arg_token)] == e1_arg_token
        assert prompt[tmp_p2_start:tmp_p2_start + len(e2_arg_token)] == e2_arg_token
    assert prompt[e1_start:e1_start + len(e1_trigger)] == e1_trigger
    assert prompt[e1s_start:e1s_start + len(e1s_token)] == e1s_token
    assert prompt[e1e_start:e1e_start + len(e1e_token)] == e1e_token
    assert prompt[e2_start:e2_start + len(e2_trigger)] == e2_trigger
    assert prompt[e2s_start:e2s_start + len(e2s_token)] == e2s_token
    assert prompt[e2e_start:e2e_start + len(e2e_token)] == e2e_token
    for entity in e1_entities:
        assert prompt[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text'], f'{prompt}\n{entity}'
    for entity in e2_entities:
        assert prompt[entity['offset']:entity['offset'] + len(entity['text'])] == entity['text'], f'{prompt}\n{entity}'
    # convert char offsets to token idxs
    encoding = tokenizer(prompt)
    mask_idx = encoding.char_to_token(prompt.find(mask_token))
    e1_idx, e1s_idx, e1e_idx = encoding.char_to_token(e1_start), encoding.char_to_token(e1s_start), encoding.char_to_token(e1e_start)
    e2_idx, e2s_idx, e2e_idx = encoding.char_to_token(e2_start), encoding.char_to_token(e2s_start), encoding.char_to_token(e2e_start)
    tmp_e1_idx, tmp_e1s_idx, tmp_e1e_idx, tmp_e1_arg_idx = (
        encoding.char_to_token(tmp_e1_start), 
        encoding.char_to_token(tmp_e1s_start), 
        encoding.char_to_token(tmp_e1e_start), 
        encoding.char_to_token(tmp_p1_start) if 'e' in prompt_type else 0
    )
    tmp_e2_idx, tmp_e2s_idx, tmp_e2e_idx, tmp_e2_arg_idx = (
        encoding.char_to_token(tmp_e2_start), 
        encoding.char_to_token(tmp_e2s_start), 
        encoding.char_to_token(tmp_e2e_start), 
        encoding.char_to_token(tmp_p2_start) if 'e' in prompt_type else 0
    )
    e1_entity_idxs, e2_entity_idxs = [], []
    for entity in e1_entities:
        ent_s_idx, ent_e_idx = encoding.char_to_token(entity['offset']), encoding.char_to_token(entity['offset']  + len(entity['text']) - 1)
        assert ent_s_idx is not None and ent_e_idx is not None
        e1_entity_idxs.append([ent_s_idx, ent_e_idx])
    for entity in e2_entities:
        ent_s_idx, ent_e_idx = encoding.char_to_token(entity['offset']), encoding.char_to_token(entity['offset']  + len(entity['text']) - 1)
        assert ent_s_idx is not None and ent_e_idx is not None
        e2_entity_idxs.append([ent_s_idx, ent_e_idx])
    assert None not in [
        mask_idx, 
        tmp_e1_idx, tmp_e1s_idx, tmp_e1e_idx, tmp_e1_arg_idx, 
        tmp_e2_idx, tmp_e2s_idx, tmp_e2e_idx, tmp_e2_arg_idx, 
        e1_idx, e1s_idx, e1e_idx, e2_idx, e2s_idx, e2e_idx
    ]
    
    return {
        'prompt': prompt, 
        'mask_idx': mask_idx, 
        'tmp_e1_idx': tmp_e1_idx, 
        'tmp_e1s_idx': tmp_e1s_idx, 
        'tmp_e1e_idx': tmp_e1e_idx, 
        'tmp_e1_arg_idx': tmp_e1_arg_idx, 
        'tmp_e2_idx': tmp_e2_idx, 
        'tmp_e2s_idx': tmp_e2s_idx, 
        'tmp_e2e_idx': tmp_e2e_idx, 
        'tmp_e2_arg_idx': tmp_e2_arg_idx, 
        'e1_idx': e1_idx, 
        'e1s_idx': e1s_idx, 
        'e1e_idx': e1e_idx, 
        'e2_idx': e2_idx, 
        'e2s_idx': e2s_idx,
        'e2e_idx': e2e_idx, 
        'e1_entity_idxs': e1_entity_idxs, 
        'e2_entity_idxs': e2_entity_idxs
    }
