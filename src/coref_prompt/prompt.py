import numpy as np

'''
- h, s: hard/soft
- n, c, q: normal/connect/question
- t, ta, tao: type/type-arg/type-arg-other
'''
PROMPT_TYPE = [
    'hn', 'hc', 'hq', # base prompts
    'sn', 'sc', 'sq', 
    'm_ht_hn', 'm_ht_hc', 'm_ht_hq', 'm_hta_hn', 'm_hta_hc', 'm_hta_hq', 'm_htao_hn', 'm_htao_hc', 'm_htao_hq', # mix prompts
    'm_st_hn', 'm_st_hc', 'm_st_hq', 'm_sta_hn', 'm_sta_hc', 'm_sta_hq', 'm_stao_hn', 'm_stao_hc', 'm_stao_hq', 
    'ma_remove-prefix', 'ma_remove-anchor', 'ma_remove-match', 'ma_remove-subtype-match', 'ma_remove-arg-match' # mix prompt m_hta_hn ablations
]
WORD_FILTER = set([
    'you', 'your', 'yours', 'yourself', 'yourselves', 
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 
    'he', 'his', 'him', 'himself', 'she', 'her', 'herself', 'hers', 
    'it', 'its', 'itself', 'they', 'their', 'theirs', 'them', 'themselves', 'other', 'others', 
    'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'whose', 'which', 'where', 'why', 
    'that', 'all', 'each', 'either', 'neither', 
    'one', 'any', 'oneself', 'such', 'same', 'everyone', 'anyone', 'there', 
])
SELECT_ARG_STRATEGY = ['no_filter', 'filter_related_args', 'filter_all']
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
    [e1/e2_]e_sent_idx:
        host sentence index
    [e1/e2_]e_sent_start: 
        trigger offset in the host sentence
    [e1/e2_]e_trigger:
        trigger of the event
    sentences: 
        all the sentences in the document, {"start": sentence offset in the document, "text": content}
    sentence_lens: 
        token numbers of all the sentences (not include [CLS], [SEP], etc.)
    s_tokens:
        special token dictionary
    tokenizer:
        tokenizer of the chosen PTM
    max_length:
        max total token numbers of segments (not include [CLS], [SEP], etc.)
    # Return
    type: 
        context type, 'same_sent' or 'diff_sent', two events in the same/different sentence
    [e1/e2_]core_context: 
        the host sentence contains the event
    [e1/e2_]before_context: 
        context before the host sentence
    [e1/e2_]after_context: 
        context after the host sentence
    e1s_core_offset, e1e_core_offset, e2s_core_offset, e2e_core_offset: 
        offsets of triggers in the host sentence
    '''
    if e1_sent_idx == e2_sent_idx: # two events in the same sentence
        assert e1_sent_start <= e2_sent_start
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
        else: # create contexts before/after the host sentence
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
        tri1s_core_offset, tri1e_core_offset = e1s_offset + len(s_tokens['e1s']) + 1, e1e_offset - 2
        tri2s_core_offset, tri2e_core_offset = e2s_offset + len(s_tokens['e2s']) + 1, e2e_offset - 2
        assert core_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
        assert core_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
        assert core_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
        assert core_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        assert core_context[tri1s_core_offset:tri1e_core_offset+1] == e1_trigger
        assert core_context[tri2s_core_offset:tri2e_core_offset+1] == e2_trigger
        return {
            'type': 'same_sent', 
            'core_context': core_context, 
            'before_context': before_context, 
            'after_context': after_context, 
            'e1s_core_offset': e1s_offset, 
            'e1e_core_offset': e1e_offset, 
            'tri1s_core_offset': tri1s_core_offset, 
            'tri1e_core_offset': tri1e_core_offset, 
            'e2s_core_offset': e2s_offset, 
            'e2e_core_offset': e2e_offset, 
            'tri2s_core_offset': tri2s_core_offset, 
            'tri2e_core_offset': tri2e_core_offset
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
        tri1s_core_offset, tri1e_core_offset = e1s_offset + len(s_tokens['e1s']) + 1, e1e_offset - 2
        tri2s_core_offset, tri2e_core_offset = e2s_offset + len(s_tokens['e2s']) + 1, e2e_offset - 2
        assert e1_core_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
        assert e1_core_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
        assert e2_core_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
        assert e2_core_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        assert e1_core_context[tri1s_core_offset:tri1e_core_offset+1] == e1_trigger
        assert e2_core_context[tri2s_core_offset:tri2e_core_offset+1] == e2_trigger
        return {
            'type': 'diff_sent', 
            'e1_core_context': e1_core_context, 
            'e1_before_context': e1_before_context, 
            'e1_after_context': e1_after_context, 
            'e1s_core_offset': e1s_offset, 
            'e1e_core_offset': e1e_offset, 
            'tri1s_core_offset': tri1s_core_offset, 
            'tri1e_core_offset': tri1e_core_offset, 
            'e2_core_context': e2_core_context, 
            'e2_before_context': e2_before_context, 
            'e2_after_context': e2_after_context, 
            'e2s_core_offset': e2s_offset, 
            'e2e_core_offset': e2e_offset, 
            'tri2s_core_offset': tri2s_core_offset, 
            'tri2e_core_offset': tri2e_core_offset
        }

def create_base_template(e1_trigger:str, e2_trigger:str, prompt_type:str, s_tokens:dict) -> dict:
    trigger_offsets = []
    if prompt_type.startswith('h'): # hard template
        if prompt_type == 'hn':
            template = f"In the following text, events expressed by {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} refer to {s_tokens['mask']} event: "
        elif prompt_type == 'hc':
            template = f"In the following text, the event expressed by {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['mask']} the event expressed by {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']}: "
        elif prompt_type == 'hq':
            template = f"In the following text, do events expressed by {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} refer to the same event? {s_tokens['mask']}. "
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    elif prompt_type.startswith('s'): # soft template
        if prompt_type == 'sn':
            template = f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} {s_tokens['l3']} {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} {s_tokens['l4']} {s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']}: "
        elif prompt_type == 'sc':
            template = f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
            template += f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']} {s_tokens['l3']} {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}: "
        elif prompt_type == 'sq':
            template = f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} "
            trigger_offsets.append([len(template), len(template) + len(e1_trigger) - 1])
            template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} {s_tokens['l3']} {s_tokens['e2s']} "
            trigger_offsets.append([len(template), len(template) + len(e2_trigger) - 1])
            template += f"{e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}? {s_tokens['mask']}. "
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    special_tokens = [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e']
    ] if prompt_type.startswith('h') else [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], 
        s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
    ]
    if 'c' in prompt_type: # connect template
        special_tokens += [s_tokens['refer'], s_tokens['no_refer']]
    return {
        'template': template, 
        'trigger_offsets': trigger_offsets, 
        'special_tokens': special_tokens
    }

def create_mix_template(
    e1_trigger:str, e2_trigger:str, 
    e1_arg_str: str, e2_arg_str: str, e1_related_str:str, e2_related_str:str, 
    prompt_type:str, s_tokens:dict
    ) -> dict:
    remove_prefix_temp, remove_anchor_temp = False, False
    remove_match, remove_subtype_match, remove_arg_match = False, False, False
    if prompt_type.startswith('ma'): # m_hta_hn prompt ablation
        anchor_temp_type, inference_temp_type = 'hta', 'hn'
        ablation = prompt_type.split('_')[1]
        if ablation == 'remove-prefix':
            remove_prefix_temp = True
        elif ablation == 'remove-anchor':
            remove_anchor_temp = True
        elif ablation == 'remove-match':
            remove_match = True
        elif ablation == 'remove-subtype-match':
            remove_subtype_match = True
        elif ablation == 'remove-arg-match':
            remove_arg_match = True
    else:
        _, anchor_temp_type, inference_temp_type = prompt_type.split('_')
    # prefix template
    prefix_trigger_offsets = []
    prefix_template = f"In the following text, the focus is on the events expressed by {s_tokens['e1s']} "
    prefix_trigger_offsets.append([len(prefix_template), len(prefix_template) + len(e1_trigger) - 1])
    prefix_template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
    prefix_trigger_offsets.append([len(prefix_template), len(prefix_template) + len(e2_trigger) - 1])
    prefix_template += f"{e2_trigger} {s_tokens['e2e']}, and it needs to judge whether they refer to the same or different events: "
    # anchor template
    if anchor_temp_type.startswith('h'): # hard template
        e1_anchor_temp = "Here "
        e1s_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
        e1e_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1e']} expresses a {s_tokens['mask']} event"
        e2_anchor_temp = "Here "
        e2s_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2e']} expresses a {s_tokens['mask']} event"
    elif anchor_temp_type.startswith('s'): # soft template
        e1_anchor_temp = f"{s_tokens['l1']} {s_tokens['mask']} {s_tokens['l5']} "
        e1s_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1s']} {e1_trigger} "
        e1e_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{s_tokens['e1e']} {s_tokens['l2']}"
        e2_anchor_temp = f"{s_tokens['l3']} {s_tokens['mask']} {s_tokens['l6']} "
        e2s_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{s_tokens['e2e']} {s_tokens['l4']}"
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    if anchor_temp_type.endswith('tao'): 
        e1_anchor_temp += f"{' ' + e1_arg_str if e1_arg_str else ''}{' ' + e1_related_str if e1_related_str else ''}."
        e2_anchor_temp += f"{' ' + e2_arg_str if e2_arg_str else ''}{' ' + e2_related_str if e2_related_str else ''}."
    elif anchor_temp_type.endswith('ta'): 
        e1_anchor_temp += f"{' ' + e1_arg_str if e1_arg_str else ''}."
        e2_anchor_temp += f"{' ' + e2_arg_str if e2_arg_str else ''}."
    elif anchor_temp_type.endswith('t'): 
        e1_anchor_temp += f"."
        e2_anchor_temp += f"."
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    # inference template
    infer_trigger_offsets = []
    infer_template = f"In conclusion, the events expressed by {s_tokens['e1s']} "
    infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e1_trigger) - 1])
    infer_template += f"{e1_trigger} {s_tokens['e1e']} and {s_tokens['e2s']} "
    infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e2_trigger) - 1])
    infer_template += f"{e2_trigger} {s_tokens['e2e']}"
    if remove_match or remove_subtype_match or remove_arg_match:
        if remove_match:
            infer_template += f" refer to {s_tokens['mask']} event."
        elif remove_subtype_match:
            infer_template += f" have {s_tokens['mask']} participants, so they refer to {s_tokens['mask']} event."
        elif remove_arg_match:
            infer_template += f" have {s_tokens['mask']} event type, so they refer to {s_tokens['mask']} event."
    else:
        infer_template += f" have {s_tokens['mask']} event type and {s_tokens['mask']} participants"
        if inference_temp_type == 'hn': 
            infer_template += f", so they refer to {s_tokens['mask']} event."
        elif inference_temp_type == 'hc': 
            infer_template += f". So the event expressed by {s_tokens['e1s']} "
            infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e1_trigger) - 1])
            infer_template += f"{e1_trigger} {s_tokens['e1e']} {s_tokens['mask']} the event expressed by {s_tokens['e2s']} "
            infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e2_trigger) - 1])
            infer_template += f"{e2_trigger} {s_tokens['e2e']}."
        elif inference_temp_type == 'hq': 
            infer_template += f". So do they refer to the same event? {s_tokens['mask']}."
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    special_tokens = [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e']
    ] if anchor_temp_type.startswith('h') else [
        s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], 
        s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
    ]
    if not remove_anchor_temp:
        special_tokens += [s_tokens[f'st{i}'] for i in range(len(EVENT_SUBTYPES) + 1)]
    if not remove_match:
        special_tokens += [s_tokens['match'], s_tokens['mismatch']]
    if 'c' in inference_temp_type: # connect template
        special_tokens += [s_tokens['refer'], s_tokens['no_refer']]
    return {
        'prefix_template': "" if remove_prefix_temp else prefix_template, 
        'e1_anchor_template': "" if remove_anchor_temp else e1_anchor_temp, 
        'e2_anchor_template': "" if remove_anchor_temp else e2_anchor_temp, 
        'infer_template': infer_template, 
        'prefix_trigger_offsets': [] if remove_prefix_temp else prefix_trigger_offsets, 
        'infer_trigger_offsets': infer_trigger_offsets, 
        'e1s_anchor_offset': -1 if remove_anchor_temp else e1s_anchor_offset, 
        'e1e_anchor_offset': -1 if remove_anchor_temp else e1e_anchor_offset, 
        'e2s_anchor_offset': -1 if remove_anchor_temp else e2s_anchor_offset, 
        'e2e_anchor_offset': -1 if remove_anchor_temp else e2e_anchor_offset, 
        'special_tokens': special_tokens
    }

def findall(p, s):
    '''yields all the positions of p in s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def create_arg_and_related_info_str(prompt_type:str, e1_related_info:dict, e2_related_info:dict, select_arg_strategy:str, s_tokens:dict):

    assert select_arg_strategy in ['no_filter', 'filter_related_args', 'filter_all']

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

    def convert_args_to_str(args:list, use_filter:bool, soft_prompt:bool):
        if use_filter:
            args = filter(lambda x: x['mention'].lower() not in WORD_FILTER, args)
        if soft_prompt:
            return f"{s_tokens['l7']} {', '.join([arg['mention'] for arg in args])} {s_tokens['l8']}".strip() if args else "", [s_tokens['l7'], s_tokens['l8']]
        participants, places, unknows = (
            [arg for arg in args if arg['role'] == 'participant'], 
            [arg for arg in args if arg['role'] == 'place'], 
            [arg for arg in args if arg['role'] == 'unk']
        )
        arg_str = ''
        if participants:
            participants.sort(key=lambda x: x['global_offset'])
            arg_str = f"with {', '.join([arg['mention'] for arg in participants])} as participants"
        if places:
            places.sort(key=lambda x: x['global_offset'])
            arg_str += f" at {', '.join([arg['mention'] for arg in places])}"
        if unknows:
            arg_str += f" (other arguments are {', '.join([arg['mention'] for arg in unknows])})"
        return arg_str.strip(), []

    def convert_related_info_to_str(related_triggers:list, related_args:list, use_filter:bool, soft_prompt:bool):
        if use_filter:
            related_args = list(filter(lambda x: x['mention'].lower() not in WORD_FILTER, related_args))
        if soft_prompt:
            return (
                f"{', '.join(set(related_triggers))} {s_tokens['l9']} " if related_triggers else ""
                f"{', '.join([arg['mention'] for arg in related_args])} {s_tokens['l10']}" if related_args else ""
            ).strip(), [s_tokens['l9'], s_tokens['l10']]
        related_str = ''
        if related_triggers:
            related_str = f"(with related events: {', '.join(set(related_triggers))}"
            related_str += f", and related participants/places: {', '.join([arg['mention'] for arg in related_args])})" if related_args else ')'
        elif related_args:
            related_str = f"(with related participants/places: {', '.join([arg['mention'] for arg in related_args])})"
        return related_str.strip(), []

    special_tokens = []
    e1_args = select_args(e1_related_info['arguments'], e2_related_info, 'tao' in prompt_type) if select_arg_strategy == 'filter_all' else e1_related_info['arguments']
    e2_args = select_args(e2_related_info['arguments'], e1_related_info, 'tao' in prompt_type) if select_arg_strategy == 'filter_all' else e2_related_info['arguments']
    e1_arg_str, arg_special_tokens = convert_args_to_str(e1_args, not prompt_type.startswith('m'), 'st' in prompt_type)
    e2_arg_str, _ = convert_args_to_str(e2_args, not prompt_type.startswith('m'), 'st' in prompt_type)
    special_tokens += arg_special_tokens
    e1_related_triggers, e2_related_triggers = e1_related_info['related_triggers'], e2_related_info['related_triggers']
    if not e1_related_triggers or not e2_related_triggers:
        e1_related_triggers, e2_related_triggers = [], []
    e1_related_args = select_args(e1_related_info['related_arguments'], e2_related_info, 'tao' in prompt_type) \
        if select_arg_strategy in ['filter_all', 'filter_related_args'] else e1_related_info['related_arguments']
    e2_related_args = select_args(e2_related_info['related_arguments'], e1_related_info, 'tao' in prompt_type) \
        if select_arg_strategy in ['filter_all', 'filter_related_args'] else e2_related_info['related_arguments']
    e1_related_str, related_special_tokens = convert_related_info_to_str(e1_related_triggers,e1_related_args, True, 'st' in prompt_type)
    e2_related_str, _ = convert_related_info_to_str(e2_related_triggers, e2_related_args, True, 'st' in prompt_type)
    special_tokens += related_special_tokens
    return e1_arg_str, e2_arg_str, e1_related_str, e2_related_str, list(set(special_tokens))

def create_prompt(
    e1_sent_idx:int, e1_sent_start:int, e1_trigger:str, e1_related_info: dict, 
    e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, e2_related_info: dict, 
    sentences:list, sentence_lens:list, 
    prompt_type:str, select_arg_strategy:str, 
    model_type, tokenizer, max_length:int
    ) -> dict:
    '''create event coreference prompts
    [e1/e2]_sent_idx:
        host sentence index
    [e1/e2]_sent_start: 
        trigger offset in the host sentence
    [e1/e2]_trigger:
        trigger of the event
    [e1/e2]_related_info:
        arguments & related event information dict
    sentences: 
        all the sentences in the document, {"start": sentence offset in the document, "text": content}
    sentence_lens: 
        token numbers of all the sentences (not include [CLS], [SEP], etc.)
    prompt_type:
        prompt type
    select_arg_strategy:
        argument select strategy
    model_type:
        PTM type
    tokenizer:
        tokenizer of the chosen PTM
    max_length:
        max total token numbers of prompt
    # Return
    {
        'prompt': prompt, \n
        'mask_offset': coreference mask offset in the prompt, \n
        'type_match_mask_offset': event type match mask offset in the prompt, \n
        'arg_match_mask_offset': argument match mask offset in the prompt, \n
        'e1s_offset': [e1s] offset in the prompt, \n
        'e1e_offset': [e1e] offset in the prompt, \n
        'e1_type_mask_offset': e1 event type mask offset in the prompt, \n
        'e2s_offset': [e2s] offset in the prompt, \n
        'e2e_offset': [e2e] offset in the prompt, \n
        'e2_type_mask_offset': e2 event type mask offset in the prompt, \n
        'trigger_offsets': all the triggers' offsets in the prompt
    }
    '''

    special_token_dict = {
        'mask': '[MASK]', 'e1s': '[E1_START]', 'e1e': '[E1_END]', 'e2s': '[E2_START]', 'e2e': '[E2_END]', 
        'l1': '[L1]', 'l2': '[L2]', 'l3': '[L3]', 'l4': '[L4]', 'l5': '[L5]', 'l6': '[L6]', 
        'l7': '[L7]', 'l8': '[L8]', 'l9': '[L9]', 'l10': '[L10]', 
        'match': '[MATCH]', 'mismatch': '[MISMATCH]', 'refer': '[REFER_TO]', 'no_refer': '[NOT_REFER_TO]'
    } if model_type == 'bert' else {
        'mask': '<mask>', 'e1s': '<e1_start>', 'e1e': '<e1_end>', 'e2s': '<e2_start>', 'e2e': '<e2_end>', 
        'l1': '<l1>', 'l2': '<l2>', 'l3': '<l3>', 'l4': '<l4>', 'l5': '<l5>', 'l6': '<l6>', 
        'l7': '<l7>', 'l8': '<l8>', 'l9': '<l9>', 'l10': '<l10>', 
        'match': '<match>', 'mismatch': '<mismatch>', 'refer': '<refer_to>', 'no_refer': '<not_refer_to>'
    }
    for i in range(len(EVENT_SUBTYPES) + 1):
        special_token_dict[f'st{i}'] = f'[ST_{i}]' if model_type == 'bert' else f'<st_{i}>'

    if prompt_type.startswith('h') or prompt_type.startswith('s'): # base prompt
        template_data = create_base_template(e1_trigger, e2_trigger, prompt_type, special_token_dict)
        trigger_offsets = template_data['trigger_offsets']
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
                    len(context_data['e1_core_context']) + len(context_data['e1_after_context']) + 1 + 
                    len(context_data['e2_before_context'])
                )
            )
        mask_offset = prompt.find(special_token_dict['mask'])
        trigger_offsets.append([e1s_offset + len(special_token_dict['e1s']) + 1, e1e_offset - 2])
        trigger_offsets.append([e2s_offset + len(special_token_dict['e2s']) + 1, e2e_offset - 2])
        assert prompt[mask_offset:mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
        for s, e in trigger_offsets:
            assert prompt[s:e+1] in [e1_trigger, e2_trigger]
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
            'e2_type_mask_offset': -1, 
            'trigger_offsets': trigger_offsets
        }
    elif prompt_type.startswith('m'): # mix prompt
        remove_anchor_temp, remove_match, remove_subtype_match, remove_arg_match = (
            (prompt_type == 'ma_remove-anchor'), 
            (prompt_type == 'ma_remove-match'), 
            (prompt_type == 'ma_remove-subtype-match'), 
            (prompt_type == 'ma_remove-arg-match')
        )
        e1_arg_str, e2_arg_str, e1_related_str, e2_related_str, special_tokens = create_arg_and_related_info_str(
            prompt_type, e1_related_info, e2_related_info, select_arg_strategy, special_token_dict
        )
        template_data = create_mix_template(e1_trigger, e2_trigger, e1_arg_str, e2_arg_str, e1_related_str, e2_related_str, prompt_type, special_token_dict)
        template_length = (
            len(tokenizer.tokenize(template_data['prefix_template'])) + 
            len(tokenizer.tokenize(template_data['e1_anchor_template'])) + 
            len(tokenizer.tokenize(template_data['e2_anchor_template'])) + 
            len(tokenizer.tokenize(template_data['infer_template'])) + 
            6
        )
        trigger_offsets = template_data['prefix_trigger_offsets']
        assert set(
            ([] if remove_anchor_temp else special_tokens) + template_data['special_tokens'] 
        ).issubset(set(tokenizer.additional_special_tokens))
        context_data = create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger,  
            sentences, sentence_lens, 
            special_token_dict, tokenizer, max_length - template_length
        )
        e1s_offset, e1e_offset = template_data['e1s_anchor_offset'], template_data['e1e_anchor_offset']
        e2s_offset, e2e_offset = template_data['e2s_anchor_offset'], template_data['e2e_anchor_offset']
        e1s_context_offset, e1e_context_offset = context_data['e1s_core_offset'], context_data['e1e_core_offset']
        e2s_context_offset, e2e_context_offset = context_data['e2s_core_offset'], context_data['e2e_core_offset']
        infer_trigger_offsets = template_data['infer_trigger_offsets']
        if context_data['type'] == 'same_sent': # two events in the same sentence
            prompt = template_data['prefix_template'] + context_data['before_context'] + context_data['core_context'] + ' '
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(prompt))
            prompt += template_data['e1_anchor_template'] + ' '
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(prompt))
            prompt += template_data['e2_anchor_template'] + context_data['after_context'] + ' ' + template_data['infer_template']
            e1s_context_offset, e1e_context_offset, e2s_context_offset, e2e_context_offset = (
                np.asarray([e1s_context_offset, e1e_context_offset, e2s_context_offset, e2e_context_offset]) + 
                np.full((4,), len(template_data['prefix_template']) + len(context_data['before_context']))
            )
            if remove_anchor_temp:
                e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
                    e1s_context_offset, e1e_context_offset, e2s_context_offset, e2e_context_offset
                )
            infer_temp_offset = (
                len(template_data['prefix_template']) + len(context_data['before_context']) + len(context_data['core_context']) + 1 + 
                len(template_data['e1_anchor_template']) + 1 + len(template_data['e2_anchor_template']) + len(context_data['after_context']) + 1
            )
            infer_trigger_offsets = [
                [s + infer_temp_offset, e + infer_temp_offset] 
                for s, e in infer_trigger_offsets
            ]
        else: # two events in different sentences
            prompt = template_data['prefix_template'] + context_data['e1_before_context'] + context_data['e1_core_context'] + ' '
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(prompt))
            prompt += (
                template_data['e1_anchor_template'] + context_data['e1_after_context'] + ' ' + 
                context_data['e2_before_context'] + context_data['e2_core_context'] + ' '
            )
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(prompt))
            prompt += template_data['e2_anchor_template'] + context_data['e2_after_context'] + ' ' + template_data['infer_template']
            e1s_context_offset, e1e_context_offset = (
                np.asarray([e1s_context_offset, e1e_context_offset]) + 
                np.full((2,), len(template_data['prefix_template'] + context_data['e1_before_context']))
            )
            e2s_context_offset, e2e_context_offset = (
                np.asarray([e2s_context_offset, e2e_context_offset]) + 
                np.full((2,), len(template_data['prefix_template']) + len(context_data['e1_before_context']) + len(context_data['e1_core_context']) + 1 + 
                    len(template_data['e1_anchor_template']) + len(context_data['e1_after_context']) + 1 + 
                    len(context_data['e2_before_context'])
                )
            )
            if remove_anchor_temp:
                e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
                    e1s_context_offset, e1e_context_offset, e2s_context_offset, e2e_context_offset
                )
            infer_temp_offset = (
                len(template_data['prefix_template']) + len(context_data['e1_before_context']) + len(context_data['e1_core_context']) + 1 + 
                len(template_data['e1_anchor_template']) + len(context_data['e1_after_context']) + 1 + 
                len(context_data['e2_before_context']) + len(context_data['e2_core_context']) + 1 + 
                len(template_data['e2_anchor_template']) + len(context_data['e2_after_context']) + 1
            )
            infer_trigger_offsets = [
                [s + infer_temp_offset, e + infer_temp_offset] 
                for s, e in infer_trigger_offsets
            ]
        mask_offsets = list(findall(special_token_dict['mask'], prompt))
        assert len(mask_offsets) == (3 if remove_anchor_temp or remove_match else 4 if remove_subtype_match or remove_arg_match else 5)
        if remove_anchor_temp:
            type_match_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
        else:
            if remove_match:
                e1_type_mask_offset, e2_type_mask_offset, mask_offset = mask_offsets
            elif remove_subtype_match:
                e1_type_mask_offset, e2_type_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
            elif remove_arg_match:
                e1_type_mask_offset, e2_type_mask_offset, type_match_mask_offset, mask_offset = mask_offsets
            else:
                e1_type_mask_offset, e2_type_mask_offset, type_match_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
            assert prompt[e1_type_mask_offset:e1_type_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
            assert prompt[e2_type_mask_offset:e2_type_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        trigger_offsets.append([e1s_context_offset + len(special_token_dict['e1s']) + 1, e1e_context_offset - 2])
        trigger_offsets.append([e2s_context_offset + len(special_token_dict['e2s']) + 1, e2e_context_offset - 2])
        if not remove_anchor_temp:
            trigger_offsets.append([e1s_offset + len(special_token_dict['e1s']) + 1, e1e_offset - 2])
            trigger_offsets.append([e2s_offset + len(special_token_dict['e2s']) + 1, e2e_offset - 2])
        trigger_offsets += infer_trigger_offsets
        if not remove_match:
            if not remove_subtype_match:
                assert prompt[type_match_mask_offset:type_match_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
            if not remove_arg_match:
                assert prompt[arg_match_mask_offset:arg_match_mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[mask_offset:mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
        for s, e in trigger_offsets:
            assert prompt[s:e+1] == e1_trigger or prompt[s:e+1] == e2_trigger
        return {
            'prompt': prompt, 
            'mask_offset': mask_offset, 
            'type_match_mask_offset': -1 if remove_match or remove_subtype_match else type_match_mask_offset, 
            'arg_match_mask_offset': -1 if remove_match or remove_arg_match else arg_match_mask_offset, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e1_type_mask_offset': -1 if remove_anchor_temp else e1_type_mask_offset, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset, 
            'e2_type_mask_offset': -1 if remove_anchor_temp else e2_type_mask_offset, 
            'trigger_offsets': trigger_offsets
        }

def create_verbalizer(tokenizer, model_type, prompt_type):
    base_verbalizer = {
        'coref': {'token': 'same', 'id': tokenizer.convert_tokens_to_ids('same')}, 
        'non-coref': {'token': 'different', 'id': tokenizer.convert_tokens_to_ids('different')}
    } if prompt_type.startswith('ma') else {
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
    else: # mix prompt
        if prompt_type != 'ma_remove-anchor':
            for subtype, s_id in subtype2id.items():
                base_verbalizer[subtype] = {
                    'token': f'[ST_{s_id}]' if model_type == 'bert' else f'<st_{s_id}>', 
                    'id': tokenizer.convert_tokens_to_ids(f'[ST_{s_id}]' if model_type == 'bert' else f'<st_{s_id}>'), 
                    'description': subtype if subtype != 'other' else 'normal'
                }
        if prompt_type != 'ma_remove-match':
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
            '[E1_START]', '[E1_END]', '[E2_START]', '[E2_END]', 
            '[L1]', '[L2]', '[L3]', '[L4]', '[L5]', '[L6]', '[L7]', '[L8]', '[L9]', '[L10]'
        ] if model_type == 'bert' else [
            '<e1_start>', '<e1_end>', '<e2_start>', '<e2_end>', 
            '<l1>', '<l2>', '<l3>', '<l4>', '<l5>', '<l6>', '<l7>', '<l8>', '<l9>', '<l10>'
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
