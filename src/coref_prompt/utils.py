import numpy as np

def create_event_context(
    e1_sent_idx:int, e1_sent_start:int, e1_trigger:str,  
    e2_sent_idx:int, e2_sent_start:int, e2_trigger:str,  
    sentences:list, sentence_lens:list, 
    s_tokens:dict, tokenizer, max_length:int
    ) -> dict:
    if e1_sent_idx == e2_sent_idx: # two events in the same sentence
        assert e1_sent_start < e2_sent_start
        e1_e2_sent = sentences[e1_sent_idx]['text']
        core_context = f"{e1_e2_sent[:e1_sent_start]}"
        e1s_offset = len(core_context)
        core_context += f"{s_tokens['e1s']} {e1_trigger} "
        e1e_offset = len(core_context)
        core_context += f"{s_tokens['e1e']}{e1_e2_sent[e1_sent_start + len(e1_trigger):e2_sent_start]}"
        e2s_offset = len(core_context)
        core_context += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_offset = len(core_context)
        core_context += f"{s_tokens['e2e']}{e1_e2_sent[e2_sent_start + len(e2_trigger):]}"
        # add other sentences
        total_length = len(tokenizer(core_context).tokens())
        e_before, e_after = e1_sent_idx - 1, e1_sent_idx + 1
        while True:
            if e_before >= 0:
                if total_length + sentence_lens[e_before] <= max_length:
                    core_context = sentences[e_before]['text'] + ' ' + core_context
                    e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + \
                        np.full((4,), len(sentences[e_before]['text']) + 1)
                    total_length += 1 + sentence_lens[e_before]
                    e_before -= 1
                else:
                    e_before = -1
            if e_after < len(sentences):
                if total_length + sentence_lens[e_after] <= max_length:
                    core_context += ' ' + sentences[e_after]['text']
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
            'e1_e2_context': core_context, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset
        }
    else: # two events in different sentences
        e1_sent, e2_sent = sentences[e1_sent_idx]['text'], sentences[e2_sent_idx]['text']
        # e1 source sentence
        e1_context = f"{e1_sent[:e1_sent_start]}"
        e1s_offset = len(e1_context)
        e1_context += f"{s_tokens['e1s']} {e1_trigger} "
        e1e_offset = len(e1_context)
        e1_context += f"{s_tokens['e1e']}{e1_sent[e1_sent_start + len(e1_trigger):]}"
        # e2 source sentence
        e2_context = f"{e2_sent[:e2_sent_start]}"
        e2s_offset = len(e2_context)
        e2_context += f"{s_tokens['e2s']} {e2_trigger} "
        e2e_offset = len(e2_context)
        e2_context += f"{s_tokens['e2e']}{e2_sent[e2_sent_start + len(e2_trigger):]}"
        # add other sentences
        total_length = len(tokenizer(e1_context).tokens()) + len(tokenizer(e2_context).tokens())
        e1_before, e1_after, e2_before, e2_after = e1_sent_idx - 1, e1_sent_idx + 1, e2_sent_idx - 1, e2_sent_idx + 1
        while True:
            e1_after_dead, e2_before_dead = False, False
            if e1_before >= 0:
                if total_length + sentence_lens[e1_before] <= max_length:
                    e1_context = sentences[e1_before]['text'] + ' ' + e1_context
                    e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(sentences[e1_before]['text']) + 1)
                    total_length += 1 + sentence_lens[e1_before]
                    e1_before -= 1
                else:
                    e1_before = -1
            if e1_after <= e2_before:
                if total_length + sentence_lens[e1_after] <= max_length:
                    e1_context += ' ' + sentences[e1_after]['text']
                    total_length += 1 + sentence_lens[e1_after]
                    e1_after += 1
                else:
                    e1_after_dead = True
            if e2_before >= e1_after:
                if total_length + sentence_lens[e2_before] <= max_length:
                    e2_context = sentences[e2_before]['text'] + ' ' + e2_context
                    e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(sentences[e2_before]['text']) + 1)
                    total_length += 1 + sentence_lens[e2_before]
                    e2_before -= 1
                else:
                    e2_before_dead = True
            if e2_after < len(sentences):
                if total_length + sentence_lens[e2_after] <= max_length:
                    e2_context += ' ' + sentences[e2_after]['text']
                    total_length += 1 + sentence_lens[e2_after]
                    e2_after += 1
                else:
                    e2_after = len(sentences)
            if e1_before == -1 and e2_after == len(sentences) and ((e1_after_dead and e2_before_dead) or e1_after > e2_before):
                break
        assert e1_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
        assert e1_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
        assert e2_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
        assert e2_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        return {
            'e1_context': e1_context, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e2_context': e2_context, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset
        }

def create_base_template(e1_trigger:str, e2_trigger:str, prompt_type:str, s_tokens:dict) -> dict:
    if prompt_type.startswith('h'): # hard template
        if prompt_type == 'hn':
            template = (
                f"In the following text, events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to {s_tokens['mask']} event: "
            )
        elif prompt_type == 'hm':
            template = (
                f"In the following text, the event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"{s_tokens['mask']} the event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}: "
            )
        elif prompt_type == 'hq':
            template = (
                f"In the following text, do the events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to the same event? {s_tokens['mask']}. "
            )
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
        normal_s_tokens = [s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e']]
    elif prompt_type.startswith('s'): # soft template
        if prompt_type == 'sn':
            template = (
                f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                f"{s_tokens['l3']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']} {s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']}: "
            )
            normal_s_tokens = [
                s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
            ]
        elif prompt_type == 'sm':
            template = (
                f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                f"{s_tokens['l5']} {s_tokens['mask']} {s_tokens['l6']} {s_tokens['l3']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}: "
            )
            normal_s_tokens = [
                s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4'], s_tokens['l5'], s_tokens['l6']
            ]
        elif prompt_type == 'sq':
            template = (
                f"In the following text, {s_tokens['l1']} {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} {s_tokens['l2']} "
                f"{s_tokens['l3']} {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} {s_tokens['l4']}? {s_tokens['mask']}. "
            )
            normal_s_tokens = [s_tokens['e1s'], s_tokens['e1e'], s_tokens['e2s'], s_tokens['e2e'], s_tokens['l1'], s_tokens['l2'], s_tokens['l3'], s_tokens['l4']]
        else:
            raise ValueError(f'Unknown prompt type: {prompt_type}')
    else:
        raise ValueError(f'Unknown prompt type: {prompt_type}')
    return {
        'template': template, 
        'sprcial_tokens': normal_s_tokens
    }

def create_knowledge_template(e1_trigger:str, e2_trigger:str, e1_arg_str: str, e2_arg_str: str, prompt_type:str, s_tokens:dict) -> dict:
    if prompt_type.startswith('s'): # subtype template
        if prompt_type == 's_hn':
            template = (
                f"In the following text, the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"and the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to {s_tokens['mask']} event: "
            )
        elif prompt_type == 's_hm':
            template = (
                f"In the following text, the {s_tokens['mask']} event expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"{s_tokens['mask']} the {s_tokens['mask']} event expressed by {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']}: "
            )
        elif prompt_type == 's_hq':
            template = (
                f"In the following text, do the events expressed by {s_tokens['e1s']} {e1_trigger} {s_tokens['e1e']} "
                f"and {s_tokens['e2s']} {e2_trigger} {s_tokens['e2e']} refer to the same event? {s_tokens['mask']}. "
            )
        elif 

def create_prompt(
    e1_sent_idx:int, e1_sent_start:int, e1_trigger:str, e1_arg_str: str, 
    e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, e2_arg_str: str, 
    sentences:list, sentence_lens:list, 
    prompt_type:str, model_type, tokenizer, max_length:int
    ) -> dict:

    special_token_dict = {
        'mask': '[MASK]', 'e1s': '[E1_START]', 'e1e': '[E1_END]', 'e2s': '[E2_START]', 'e2e': '[E2_END]', 
        'l1': '[L1]', 'l2': '[L2]', 'l3': '[L3]', 'l4': '[L4]', 'l5': '[L5]', 'l6': '[L6]'
    } if model_type == 'bert' else {
        'mask': '<mask>', 'e1s': '<e1_start>', 'e1e': '<e1_end>', 'e2s': '<e2_start>', 'e2e': '<e2_end>', 
        'l1': '<l1>', 'l2': '<l2>', 'l3': '<l3>', 'l4': '<l4>', 'l5': '<l5>', 'l6': '<l6>'
    }

    if prompt_type.startswith('h') or prompt_type.startswith('s'): # base prompt
        template_data = create_base_template(e1_trigger, e2_trigger, prompt_type, special_token_dict)
        assert set(template_data['sprcial_tokens']).issubset(set(tokenizer.additional_special_tokens))
        context_data = create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger,  
            sentences, sentence_lens, 
            special_token_dict, tokenizer, max_length - len(tokenizer(template_data['template']).tokens()) - 1
        )
        e1s_offset, e1e_offset, e2s_offset, e2e_offset = context_data['e1s_offset'], context_data['e1e_offset'], context_data['e2s_offset'], context_data['e2e_offset']
        if 'e1_e2_context' in context_data: # two events in the same sentence
            prompt = template_data['template'] + context_data['e1_e2_context']
            e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + np.full((4,), len(template_data['template']))
        else: # two events in different sentences
            prompt = template_data['template'] + context_data['e1_context'] + ' ' + context_data['e2_context']
            e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(template_data['template']))
            e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(template_data['template']) + len(context_data['e1_context']) + 1)
        mask_offset = template_data['template'].find(special_token_dict['mask'])
        assert prompt[mask_offset:mask_offset + len(special_token_dict['mask'])] == special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
        return {
            'prompt': prompt, 
            'mask_offset': mask_offset, 
            'e1s_offset': e1s_offset, 
            'e1e_offset': e1e_offset, 
            'e2s_offset': e2s_offset, 
            'e2e_offset': e2e_offset
        }
    elif prompt_type.startswith('s') or prompt_type.startswith('a') or prompt_type.startswith('sa'): # knowledge prompt
        pass

