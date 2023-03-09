import numpy as np

PROMPT_TYPE = [
    'hn', 'hc', 'hq', # base prompts 
    'sn', 'sc', 'sq', # (hard/soft normal/connect/question)
    't_hn', 'ta_hn', 't_hc', 'ta_hc', 't_hq', 'ta_hq', # knowledge enhanced prompts 
    't_sn', 'ta_sn', 't_sc', 'ta_sc', 't_sq', 'ta_sq', # (subtype/subtype-argument)
    'm_ht_hn', 'm_ht_hc', 'm_ht_hq', 'm_hta_hn', 'm_hta_hc', 'm_hta_hq', # mix prompts
    'm_st_hn', 'm_st_hc', 'm_st_hq', 'm_sta_hn', 'm_sta_hc', 'm_sta_hq'  # (hard/soft subtype/argument/subtype-argument)
]

EVENT_SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
id2subtype = {idx: c for idx, c in enumerate(EVENT_SUBTYPES, start=1)}
id2subtype[0] = 'other'
subtype2id = {v: k for k, v in id2subtype.items()}

def create_event_context(
    e1_sent_idx:int, e1_sent_start:int, e1_trigger:str,  
    e2_sent_idx:int, e2_sent_start:int, e2_trigger:str,  
    sentences:list, sentence_lens:list, 
    s_tokens:dict, tokenizer, max_length:int
    ) -> dict:
    '''create segments contains events
    # Args
    e_sent_idx:
        host sentence index
    e_sent_start: 
        trigger offset in the host sentence
    e_trigger:
        trigger words of the event
    sentences: 
        all the sentences of the document, 
        format {"start": sentence offset in the document, "text": content}
    sentence_lens: 
        token numbers of all the sentences (not include [CLS], [SEP], etc.)
    s_tokens:
        special token dictionary
    tokenizer:
        tokenizer of the chosen PTM
    max_length:
        max total token numbers of the two segments (not include [CLS], [SEP], etc.)
    # Return
    type: context type, 'same_sent' or 'diff_sent', two events in the same/different sentence
    (e1/e2_)core_context: 
        host sentence that contains the event
    (e1/e2_)before_context: 
        sentences before the host sentence
    (e1/e2_)after_context: 
        sentences after the host sentence
    e1s_core_offset, e1e_core_offset, e2s_core_offset, e2e_core_offset: 
        offsets of event triggers in the host sentence
    '''
    if e1_sent_idx == e2_sent_idx: # two events in the same sentence
        assert e1_sent_start < e2_sent_start
        e1_e2_sent = sentences[e1_sent_idx]['text']
        core_context_before = f"{e1_e2_sent[:e1_sent_start]}"
        core_context_after = f"{e1_e2_sent[e2_sent_start + len(e2_trigger):]}"
        e1s_offset = 0
        core_context_middle = f"{s_tokens['e1s']} {e1_trigger} "
        e1e_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e1e']}{e1_e2_sent[e1_sent_start + len(e1_trigger):e2_sent_start]}"
        e2s_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e2e']}"
        # segment contain the two events
        core_context = core_context_before + core_context_middle + core_context_after
        total_length = len(tokenizer.tokenize(core_context))
        before_context, after_context = '', ''
        if total_length > max_length: # cut segment
            before_after_length = (max_length - len(tokenizer.tokenize(core_context_middle))) // 2
            core_context_before = tokenizer.decode(tokenizer.encode(core_context_before)[1:-1][-before_after_length:])
            core_context_after = tokenizer.decode(tokenizer.encode(core_context_after)[1:-1][:before_after_length])
            core_context = core_context_before + core_context_middle + core_context_after
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + np.full((4,), len(core_context_before))
        else: # add other sentences
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + np.full((4,), len(core_context_before))
            e_before, e_after = e1_sent_idx - 1, e1_sent_idx + 1
            while True:
                if e_before >= 0:
                    if total_length + sentence_lens[e_before] <= max_length:
                        before_context = sentences[e_before]['text'] + ' ' + before_context
                        total_length += 1 + sentence_lens[e_before]
                        e_before -= 1
                    else:
                        e_before = -1
                if e_after < len(sentences):
                    if total_length + sentence_lens[e_after] <= max_length:
                        after_context += ' ' + sentences[e_after]['text']
                        total_length += 1 + sentence_lens[e_after]
                        e_after += 1
                    else:
                        e_after = len(sentences)
                if e_before == -1 and e_after == len(sentences):
                    break
        assert core_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
        assert core_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
        assert core_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
        assert core_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        return {
            'type': 'same_sent', 
            'core_context': core_context, 
            'before_context': before_context, 
            'after_context': after_context, 
            'e1s_core_offset': e1s_offset, 
            'e1e_core_offset': e1e_offset, 
            'e2s_core_offset': e2s_offset, 
            'e2e_core_offset': e2e_offset
        }
    else: # two events in different sentences
        e1_sent, e2_sent = sentences[e1_sent_idx]['text'], sentences[e2_sent_idx]['text']
        # e1 source sentence
        e1_core_context_before = f"{e1_sent[:e1_sent_start]}"
        e1_core_context_after = f"{e1_sent[e1_sent_start + len(e1_trigger):]}"
        e1s_offset = 0
        e1_core_context_middle = f"{s_tokens['e1s']} {e1_trigger} "
        e1e_offset = len(e1_core_context_middle)
        e1_core_context_middle += f"{s_tokens['e1e']}"
        # e2 source sentence
        e2_core_context_before = f"{e2_sent[:e2_sent_start]}"
        e2_core_context_after = f"{e2_sent[e2_sent_start + len(e2_trigger):]}"
        e2s_offset = 0
        e2_core_context_middle = f"{s_tokens['e2s']} {e2_trigger} "
        e2e_offset = len(e2_core_context_middle)
        e2_core_context_middle += f"{s_tokens['e2e']}"
        # segment contain the two events
        e1_core_context = e1_core_context_before + e1_core_context_middle + e1_core_context_after
        e2_core_context = e2_core_context_before + e2_core_context_middle + e2_core_context_after
        total_length = len(tokenizer.tokenize(e1_core_context)) + len(tokenizer.tokenize(e2_core_context))
        e1_before_context, e1_after_context, e2_before_context, e2_after_context = '', '', '', ''
        if total_length > max_length:
            e1_e2_middle_length = len(tokenizer.tokenize(e1_core_context_middle)) + len(tokenizer.tokenize(e2_core_context_middle))
            before_after_length = (max_length - e1_e2_middle_length) // 4
            e1_core_context_before = tokenizer.decode(tokenizer.encode(e1_core_context_before)[1:-1][-before_after_length:])
            e1_core_context_after = tokenizer.decode(tokenizer.encode(e1_core_context_after)[1:-1][:before_after_length])
            e1_core_context = e1_core_context_before + e1_core_context_middle + e1_core_context_after
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(e1_core_context_before))
            e2_core_context_before = tokenizer.decode(tokenizer.encode(e2_core_context_before)[1:-1][-before_after_length:])
            e2_core_context_after = tokenizer.decode(tokenizer.encode(e2_core_context_after)[1:-1][:before_after_length])
            e2_core_context = e2_core_context_before + e2_core_context_middle + e2_core_context_after
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(e2_core_context_before))
        else: # add other sentences
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(e1_core_context_before))
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(e2_core_context_before))
            e1_before, e1_after, e2_before, e2_after = e1_sent_idx - 1, e1_sent_idx + 1, e2_sent_idx - 1, e2_sent_idx + 1
            while True:
                e1_after_dead, e2_before_dead = False, False
                if e1_before >= 0:
                    if total_length + sentence_lens[e1_before] <= max_length:
                        e1_before_context = sentences[e1_before]['text'] + ' ' + e1_before_context
                        total_length += 1 + sentence_lens[e1_before]
                        e1_before -= 1
                    else:
                        e1_before = -1
                if e1_after <= e2_before:
                    if total_length + sentence_lens[e1_after] <= max_length:
                        e1_after_context += ' ' + sentences[e1_after]['text']
                        total_length += 1 + sentence_lens[e1_after]
                        e1_after += 1
                    else:
                        e1_after_dead = True
                if e2_before >= e1_after:
                    if total_length + sentence_lens[e2_before] <= max_length:
                        e2_before_context = sentences[e2_before]['text'] + ' ' + e2_before_context
                        total_length += 1 + sentence_lens[e2_before]
                        e2_before -= 1
                    else:
                        e2_before_dead = True
                if e2_after < len(sentences):
                    if total_length + sentence_lens[e2_after] <= max_length:
                        e2_after_context += ' ' + sentences[e2_after]['text']
                        total_length += 1 + sentence_lens[e2_after]
                        e2_after += 1
                    else:
                        e2_after = len(sentences)
                if e1_before == -1 and e2_after == len(sentences) and ((e1_after_dead and e2_before_dead) or e1_after > e2_before):
                    break
        assert e1_core_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
        assert e1_core_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
        assert e2_core_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
        assert e2_core_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        return {
            'type': 'diff_sent', 
            'e1_core_context': e1_core_context, 
            'e1_before_context': e1_before_context, 
            'e1_after_context': e1_after_context, 
            'e1s_core_offset': e1s_offset, 
            'e1e_core_offset': e1e_offset, 
            'e2_core_context': e2_core_context, 
            'e2_before_context': e2_before_context, 
            'e2_after_context': e2_after_context, 
            'e2s_core_offset': e2s_offset, 
            'e2e_core_offset': e2e_offset
        }

def create_base_template(e1_trigger:str, e2_trigger:str, prompt_type:str, s_tokens:dict) -> dict:
    if prompt_type.startswith('h'): # hard template
        if prompt_type == 'hn':
            template = (
                f"In the following text, events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to {s_tokens['mask']} event: "
            )
        elif prompt_type == 'hc':
            template = (
                f"In the following text, the event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"{s_tokens['mask']} the event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}: "
            )
        elif prompt_type == 'hq':
            template = (
                f"In the following text, do events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to the same event? {s_tokens['mask']}. "
            )
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    elif prompt_type.startswith('s'): # soft template
        if prompt_type == 'sn':
            template = (
                f"In the following text, "
                f"{s_tokens['l1']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                f"{s_tokens['l3']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']} "
                f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']}: "
            )
        elif prompt_type == 'sc':
            template = (
                f"In the following text, "
                f"{s_tokens['l1']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']} "
                f"{s_tokens['l3']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}: "
            )
        elif prompt_type == 'sq':
            template = (
                f"In the following text, "
                f"{s_tokens['l1']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                f"{s_tokens['l3']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}? {s_tokens['mask']}. "
            )
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    special_tokens = [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e']
    ] if 'h' in prompt_type else [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], 
        s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
    ]
    if 'c' in prompt_type: # connect template
        special_tokens += [s_tokens['refer'], s_tokens['no_refer']]
    return {
        'template': template, 
        'special_tokens': special_tokens
    }

def create_knowledge_template(e1_trigger:str, e2_trigger:str, e1_arg_str: str, e2_arg_str: str, prompt_type:str, s_tokens:dict) -> dict:
    if prompt_type.startswith('ta'): # subtype argument template
        if prompt_type.startswith('ta_h'): # hard template
            if prompt_type == 'ta_hn':
                template = (
                    f"In the following text, the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}"
                    f"and the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}{' ' + e2_arg_str + ' ' if e2_arg_str else ' '}refer to {s_tokens['mask']} event: "
                )
            elif prompt_type == 'ta_hc':
                template = (
                    f"In the following text, the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}"
                    f"{s_tokens['mask']} the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}{' ' + e2_arg_str if e2_arg_str else ''}: "
                )
            elif prompt_type == 'ta_hq':
                template = (
                    f"In the following text, do the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}"
                    f"and the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}{' ' + e2_arg_str + ' ' if e2_arg_str else ' '}"
                    f"refer to the same event? {s_tokens['mask']}. "
                )
            else:
                raise ValueError(f'Unknown prompt type: {prompt_type}')
        elif prompt_type.startswith('ta_s'): # soft template
            if prompt_type == 'ta_sn':
                template = (
                    f"In the following text, "
                    f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}{s_tokens['l2']} "
                    f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}{' ' + e2_arg_str + ' ' if e2_arg_str else ' '}{s_tokens['l4']} "
                    f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']}: "
                )
            elif prompt_type == 'ta_sc':
                template = (
                    f"In the following text, "
                    f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}{s_tokens['l2']} "
                    f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']} "
                    f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}{' ' + e2_arg_str + ' ' if e2_arg_str else ' '}{s_tokens['l4']}: "
                )
            elif prompt_type == 'ta_sq':
                template = (
                    f"In the following text, "
                    f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}{s_tokens['l2']} "
                    f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}{' ' + e2_arg_str + ' ' if e2_arg_str else ' '}{s_tokens['l4']}? {s_tokens['mask']}. "
                )
            else:
                raise ValueError(f'Unknown prompt type: {prompt_type}')
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    elif prompt_type.startswith('t'): # subtype template
        if prompt_type.startswith('t_h'): # hard template
            if prompt_type == 't_hn':
                template = (
                    f"In the following text, the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                    f"and the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to {s_tokens['mask']} event: "
                )
            elif prompt_type == 't_hc':
                template = (
                    f"In the following text, the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                    f"{s_tokens['mask']} the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}: "
                )
            elif prompt_type == 't_hq':
                template = (
                    f"In the following text, do the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                    f"and the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to the same event? {s_tokens['mask']}. "
                )
            else:
                raise ValueError(f'Unknown prompt type: {prompt_type}')
        elif prompt_type.startswith('t_s'): # soft template
            if prompt_type == 't_sn':
                template = (
                    f"In the following text, "
                    f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                    f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']} "
                    f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']}: "
                )
            elif prompt_type == 't_sc':
                template = (
                    f"In the following text, "
                    f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                    f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']} "
                    f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}: "
                )
            elif prompt_type == 't_sq':
                template = (
                    f"In the following text, "
                    f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                    f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}? {s_tokens['mask']}. "
                )
            else:
                raise ValueError(f'Unknown prompt type: {prompt_type}')
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    special_tokens = [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e']
    ] if 'h' in prompt_type else [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], 
        s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
    ]
    special_tokens += [s_tokens[f'st{i}'] for i in range(len(EVENT_SUBTYPES) + 1)]
    if 'c' in prompt_type: # connect template
        special_tokens += [s_tokens['refer'], s_tokens['no_refer']]
    return {
        'template': template, 
        'special_tokens': special_tokens
    }

def create_mix_template(e1_trigger:str, e2_trigger:str, e1_arg_str: str, e2_arg_str: str, prompt_type:str, s_tokens:dict) -> dict:
    _, anchor_temp_type, inference_temp_type = prompt_type.split('_')
    prefix_template = (
        f"In the following text, the focus is on the events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} and "
        f"{s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}, and it needs to judge whether they refer to the same or different events: "
    )
    if anchor_temp_type.startswith('h'): # hard template
        if anchor_temp_type == 'hta': 
            e1_anchor_temp = "Here "
            e1s_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
            e1e_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1e']} expresses a {s_tokens['mask']} event{' ' + e1_arg_str if e1_arg_str else ''}."
            e2_anchor_temp = "Here "
            e2s_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
            e2e_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2e']} expresses a {s_tokens['mask']} event{' ' + e2_arg_str if e2_arg_str else ''}."
        elif anchor_temp_type == 'ht': 
            e1_anchor_temp = "Here "
            e1s_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
            e1e_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1e']} expresses a {s_tokens['mask']} event."
            e2_anchor_temp = "Here "
            e2s_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
            e2e_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2e']} expresses a {s_tokens['mask']} event."
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    elif anchor_temp_type.startswith('s'): # soft template
        if anchor_temp_type == 'sta': 
            e1_anchor_temp = f"{s_tokens['l1']} {s_tokens['mask']} "
            e1s_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
            e1e_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1e']}{' ' + e1_arg_str + ' ' if e1_arg_str else ' '}{s_tokens['l2']}."
            e2_anchor_temp = f"{s_tokens['l3']} {s_tokens['mask']} "
            e2s_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
            e2e_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2e']}{' ' + e2_arg_str + ' ' if e2_arg_str else ' '}{s_tokens['l4']}."
        elif anchor_temp_type == 'st': 
            e1_anchor_temp = f"{s_tokens['l1']} {s_tokens['mask']} "
            e1s_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
            e1e_anchor_offset = len(e1_anchor_temp)
            e1_anchor_temp += f"{s_tokens['e1e']} {s_tokens['l2']}."
            e2_anchor_temp = f"{s_tokens['l3']} {s_tokens['mask']} "
            e2s_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
            e2e_anchor_offset = len(e2_anchor_temp)
            e2_anchor_temp += f"{s_tokens['e2e']} {s_tokens['l4']}."
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    if inference_temp_type == 'hn': 
        infer_template = (
            f"In conclusion, the events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} "
            f"have {s_tokens['mask']} event type and {s_tokens['mask']} participants, so they refer to {s_tokens['mask']} event."
        )
    elif inference_temp_type == 'hc': 
        infer_template = (
            f"In conclusion, the events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} "
            f"have {s_tokens['mask']} event type and {s_tokens['mask']} participants. Therefore, the event expressed by "
            f"{s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['mask']} the event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}."
        )
    elif inference_temp_type == 'hq': 
        infer_template = (
            f"In conclusion, the events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} "
            f"have {s_tokens['mask']} event type and {s_tokens['mask']} participants. So do they refer to the same event? {s_tokens['mask']}."
        )
    
    special_tokens = [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e']
    ] if 'h' in anchor_temp_type else [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], 
        s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
    ]
    special_tokens += [s_tokens[f'st{i}'] for i in range(len(EVENT_SUBTYPES) + 1)]
    special_tokens += [s_tokens['match'], s_tokens['mismatch']]
    if 'c' in inference_temp_type: # connect template
        special_tokens += [s_tokens['refer'], s_tokens['no_refer']]
    return {
        'prefix_template': prefix_template, 
        'e1_anchor_template': e1_anchor_temp, 
        'e2_anchor_template': e2_anchor_temp, 
        'infer_template': infer_template, 
        'e1s_anchor_offset': e1s_anchor_offset, 
        'e1e_anchor_offset': e1e_anchor_offset, 
        'e2s_anchor_offset': e2s_anchor_offset, 
        'e2e_anchor_offset': e2e_anchor_offset, 
        'special_tokens': special_tokens
    }

def convert_args_to_str(args, use_filter=True, include_participant=True, include_place=True):
    word_filter = set([
        'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'you', 'they', 'them', 'my', 'mine', 'your', 'yours', 'his', 'her', 'hers', 
        'its', 'our', 'ours', 'their', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 
        'other', 'others', 'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'whose', 'which', 'that', 'all', 'each', 'either', 'neither', 
        'one', 'any', 'oneself', 'such', 'same'
    ])
    if use_filter:
        args = filter(lambda x: x['mention'] not in word_filter, args)
    arg_str = ''
    participants, places = [arg for arg in args if arg['role'] == 'participant'], [arg for arg in args if arg['role'] == 'place']
    if include_participant and len(participants) > 0:
        participants.sort(key=lambda x: x['global_offset'])
        arg_str = f"with {', '.join([arg['mention'] for arg in participants])} as participants"
    if include_place and len(places) > 0:
        places.sort(key=lambda x: x['global_offset'])
        arg_str += ' ' + f"at {', '.join([arg['mention'] for arg in places])}"
    return arg_str.strip()

def findall(p, s):
    '''yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def create_prompt(
    e1_sent_idx:int, e1_sent_start:int, e1_trigger:str, e1_args: list, 
    e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, e2_args: list, 
    sentences:list, sentence_lens:list, 
    prompt_type:str, model_type, tokenizer, max_length:int
    ) -> dict:

    special_token_dict = {
        'mask': '[MASK]', 'e1s': '[E1_START]', 'e1e': '[E1_END]', 'e2s': '[E2_START]', 'e2e': '[E2_END]', 
        'l1': '[L1]', 'l2': '[L2]', 'l3': '[L3]', 'l4': '[L4]', 'l5': '[L5]', 'l6': '[L6]', 
        'match': '[MATCH]', 'mismatch': '[MISMATCH]', 'refer': '[REFER_TO]', 'no_refer': '[NOT_REFER_TO]'
    } if model_type == 'bert' else {
        'mask': '<mask>', 'e1s': '<e1_start>', 'e1e': '<e1_end>', 'e2s': '<e2_start>', 'e2e': '<e2_end>', 
        'l1': '<l1>', 'l2': '<l2>', 'l3': '<l3>', 'l4': '<l4>', 'l5': '<l5>', 'l6': '<l6>', 
        'match': '<match>', 'mismatch': '<mismatch>', 'refer': '<refer_to>', 'no_refer': '<not_refer_to>'
    }
    for i in range(len(EVENT_SUBTYPES) + 1):
        special_token_dict[f'st{i}'] = f'[ST_{i}]' if model_type == 'bert' else f'<st_{i}>'

    if prompt_type.startswith('h') or prompt_type.startswith('s'): # base prompt
        template_data = create_base_template(e1_trigger, e2_trigger, prompt_type, special_token_dict)
        assert set(template_data['special_tokens']).issubset(set(tokenizer.additional_special_tokens))
        template_length = len(tokenizer.tokenize(template_data['template'])) + 3
        context_data = create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger,  
            sentences, sentence_lens, 
            special_token_dict, tokenizer, max_length - template_length
        )
        e1s_offset, e1e_offset = context_data['e1s_core_offset'], context_data['e1e_core_offset']
        e2s_offset, e2e_offset = context_data['e2s_core_offset'], context_data['e2e_core_offset']
        if context_data['type'] == 'same_sent': # two events in the same sentence
            prompt = template_data['template'] + context_data['before_context'] + context_data['core_context'] + context_data['after_context']
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
                np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + 
                np.full((4,), len(template_data['template'] + context_data['before_context']))
            )
        else: # two events in different sentences
            prompt = (
                template_data['template'] + 
                context_data['e1_before_context'] + context_data['e1_core_context'] + context_data['e1_after_context'] + ' ' + 
                context_data['e2_before_context'] + context_data['e2_core_context'] + context_data['e2_after_context']
            )
            e1s_offset, e1e_offset = (
                np.asarray([e1s_offset, e1e_offset]) + 
                np.full((2,), len(template_data['template'] + context_data['e1_before_context']))
            )
            e2s_offset, e2e_offset = (
                np.asarray([e2s_offset, e2e_offset]) + 
                np.full((2,), len(template_data['template']) + len(context_data['e1_before_context']) + 
                    len(context_data['e1_core_context']) + len(context_data['e1_after_context']) + 
                    len(context_data['e2_before_context']) + 1
                )
            )
        mask_offset = prompt.find(special_token_dict['mask'])
        assert prompt[mask_offset:mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
        return {
            'prompt': prompt, 
            'mask_offset': mask_offset, 
            'type_match_mask_offset': -1, 
            'arg_match_mask_offset': -1, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e1_type_mask_offset': -1, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset, 
            'e2_type_mask_offset': -1
        }
    elif prompt_type.startswith('t'): # knowledge prompt
        e1_arg_str, e2_arg_str = convert_args_to_str(e1_args), convert_args_to_str(e2_args)
        template_data = create_knowledge_template(e1_trigger, e2_trigger, e1_arg_str, e2_arg_str, prompt_type, special_token_dict)
        assert set(template_data['special_tokens']).issubset(set(tokenizer.additional_special_tokens))
        template_length = len(tokenizer.tokenize(template_data['template'])) + 3
        context_data = create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger,  
            sentences, sentence_lens, 
            special_token_dict, tokenizer, max_length - template_length
        )
        e1s_offset, e1e_offset = context_data['e1s_core_offset'], context_data['e1e_core_offset']
        e2s_offset, e2e_offset = context_data['e2s_core_offset'], context_data['e2e_core_offset']
        if context_data['type'] == 'same_sent': # two events in the same sentence
            prompt = template_data['template'] + context_data['before_context'] + context_data['core_context'] + context_data['after_context']
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
                np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + 
                np.full((4,), len(template_data['template'] + context_data['before_context']))
            )
        else: # two events in different sentences
            prompt = (
                template_data['template'] + 
                context_data['e1_before_context'] + context_data['e1_core_context'] + context_data['e1_after_context'] + ' ' + 
                context_data['e2_before_context'] + context_data['e2_core_context'] + context_data['e2_after_context']
            )
            e1s_offset, e1e_offset = (
                np.asarray([e1s_offset, e1e_offset]) + 
                np.full((2,), len(template_data['template'] + context_data['e1_before_context']))
            )
            e2s_offset, e2e_offset = (
                np.asarray([e2s_offset, e2e_offset]) + 
                np.full((2,), len(template_data['template']) + len(context_data['e1_before_context']) + 
                    len(context_data['e1_core_context']) + len(context_data['e1_after_context']) + 
                    len(context_data['e2_before_context']) + 1
                )
            )
        mask_offsets = list(findall(special_token_dict['mask'], prompt))
        assert len(mask_offsets) == 3
        if 'c' in prompt_type: # connect template
            e1_type_mask_offset, mask_offset, e2_type_mask_offset = mask_offsets
        else:
            e1_type_mask_offset, e2_type_mask_offset, mask_offset = mask_offsets
        assert prompt[e1_type_mask_offset:e1_type_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e2_type_mask_offset:e2_type_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[mask_offset:mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
        return {
            'prompt': prompt, 
            'mask_offset': mask_offset, 
            'type_match_mask_offset': -1, 
            'arg_match_mask_offset': -1, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e1_type_mask_offset': e1_type_mask_offset, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset, 
            'e2_type_mask_offset': e2_type_mask_offset
        }
    elif prompt_type.startswith('m'): # mix prompt
        e1_arg_str, e2_arg_str = convert_args_to_str(e1_args, use_filter=False), convert_args_to_str(e2_args, use_filter=False)
        template_data = create_mix_template(e1_trigger, e2_trigger, e1_arg_str, e2_arg_str, prompt_type, special_token_dict)
        template_length = (
            len(tokenizer.tokenize(template_data['prefix_template'])) + 
            len(tokenizer.tokenize(template_data['e1_anchor_template'])) + 
            len(tokenizer.tokenize(template_data['e2_anchor_template'])) + 
            len(tokenizer.tokenize(template_data['infer_template'])) + 
            6
        )
        assert set(template_data['special_tokens']).issubset(set(tokenizer.additional_special_tokens))
        context_data = create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger,  
            sentences, sentence_lens, 
            special_token_dict, tokenizer, max_length - template_length
        )
        e1s_offset, e1e_offset = template_data['e1s_anchor_offset'], template_data['e1e_anchor_offset']
        e2s_offset, e2e_offset = template_data['e2s_anchor_offset'], template_data['e2e_anchor_offset']
        if context_data['type'] == 'same_sent': # two events in the same sentence
            prompt = template_data['prefix_template'] + context_data['before_context'] + context_data['core_context'] + ' '
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(prompt))
            prompt += template_data['e1_anchor_template'] + ' '
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(prompt))
            prompt += template_data['e2_anchor_template'] + context_data['after_context'] + ' ' + template_data['infer_template']
        else: # two events in different sentences
            prompt = template_data['prefix_template'] + context_data['e1_before_context'] + context_data['e1_core_context'] + ' '
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(prompt))
            prompt += (
                template_data['e1_anchor_template'] + context_data['e1_after_context'] + ' ' + 
                context_data['e2_before_context'] + context_data['e2_core_context'] + ' '
            )
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(prompt))
            prompt += template_data['e2_anchor_template'] + context_data['e2_after_context'] + ' ' + template_data['infer_template']
        mask_offsets = list(findall(special_token_dict['mask'], prompt))
        assert len(mask_offsets) == 5
        e1_type_mask_offset, e2_type_mask_offset, type_match_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
        assert prompt[e1_type_mask_offset:e1_type_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e2_type_mask_offset:e2_type_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[type_match_mask_offset:type_match_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[arg_match_mask_offset:arg_match_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[mask_offset:mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
        return {
            'prompt': prompt, 
            'mask_offset': mask_offset, 
            'type_match_mask_offset': type_match_mask_offset, 
            'arg_match_mask_offset': arg_match_mask_offset, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e1_type_mask_offset': e1_type_mask_offset, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset, 
            'e2_type_mask_offset': e2_type_mask_offset, 
        }

def create_verbalizer(tokenizer, model_type, prompt_type):
    base_verbalizer = {
        'coref': {
            'token': '[REFER_TO]' if model_type == 'bert' else '<refer_to>', 
            'id': tokenizer.convert_tokens_to_ids('[REFER_TO]' if model_type == 'bert' else '<refer_to>'), 
            'description': 'refer to'
        } if 'c' in prompt_type else {
            'token': 'yes', 'id': tokenizer.convert_tokens_to_ids('yes')
        } if 'q' in prompt_type else {
            'token': 'same', 'id': tokenizer.convert_tokens_to_ids('same')
        } , 
        'non-coref': {
            'token': '[NOT_REFER_TO]' if model_type == 'bert' else '<not_refer_to>', 
            'id': tokenizer.convert_tokens_to_ids('[NOT_REFER_TO]' if model_type == 'bert' else '<not_refer_to>'), 
            'description': 'not refer to'
        } if 'c' in prompt_type else {
            'token': 'no', 'id': tokenizer.convert_tokens_to_ids('no')
        } if 'q' in prompt_type else {
            'token': 'different', 'id': tokenizer.convert_tokens_to_ids('different')
        }
    }
    if prompt_type.startswith('h') or prompt_type.startswith('s'): # base prompt
        return base_verbalizer
    else:
        for subtype, s_id in subtype2id.items():
            base_verbalizer[subtype] = {
                'token': f'[ST_{s_id}]' if model_type == 'bert' else f'<st_{s_id}>', 
                'id': tokenizer.convert_tokens_to_ids(f'[ST_{s_id}]' if model_type == 'bert' else f'<st_{s_id}>'), 
                'description': subtype if subtype != 'other' else 'normal'
            }
        if prompt_type.startswith('t'): # knowledge prompt
            return base_verbalizer
        elif prompt_type.startswith('m'): # mix prompt
            base_verbalizer['match'] = {
                'token': '[MATCH]' if model_type == 'bert' else '<match>', 
                'id': tokenizer.convert_tokens_to_ids('[MATCH]' if model_type == 'bert' else '<match>'), 
                'description': 'same related relevant similar matching matched'
            }
            base_verbalizer['mismatch'] = {
                'token': '[MISMATCH]' if model_type == 'bert' else '<mismatch>', 
                'id': tokenizer.convert_tokens_to_ids('[MISMATCH]' if model_type == 'bert' else '<mismatch>'), 
                'description': 'different unrelated irrelevant dissimilar mismatched'
            }
            return base_verbalizer

def get_special_tokens(model_type:str, token_type:str):
    '''
    token_type:
        'base', 'connect', 'match', 'event_subtype'
    '''
    assert token_type in ['base', 'connect', 'match', 'event_subtype']
    if token_type == 'base':
        return [
            '[E1_START]', '[E1_END]', '[E2_START]', '[E2_END]', '[L1]', '[L2]', '[L3]', '[L4]', '[L5]', '[L6]'
        ] if model_type == 'bert' else [
            '<e1_start>', '<e1_end>', '<e2_start>', '<e2_end>', '<l1>', '<l2>', '<l3>', '<l4>', '<l5>', '<l6>'
        ]
    elif token_type == 'connect':
        return [
            '[REFER_TO]', '[NOT_REFER_TO]'
        ] if model_type == 'bert' else [
            '<refer_to>', '<not_refer_to>'
        ]
    elif token_type == 'match':
        return [
            '[MATCH]', '[MISMATCH]'
        ] if model_type == 'bert' else [
            '<match>', '<mismatch>'
        ]
    elif token_type == 'event_subtype':
        return [
            f'[ST_{i}]' if model_type == 'bert' else f'<st_{i}>' 
            for i in range(len(EVENT_SUBTYPES) + 1)
        ]
