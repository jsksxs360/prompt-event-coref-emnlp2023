import numpy as np

def create_event_context(
    add_mark:bool, 
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
    type: 
        context type, 'same_sent' or 'diff_sent', two events in the same/different sentence
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
        core_context_middle = f"{s_tokens['e1s']} {e1_trigger} " if add_mark else e1_trigger
        e1e_offset = len(core_context_middle) if add_mark else len(core_context_middle) - 1
        core_context_middle += f"{s_tokens['e1e']}{e1_e2_sent[e1_sent_start + len(e1_trigger):e2_sent_start]}" if add_mark else e1_e2_sent[e1_sent_start + len(e1_trigger):e2_sent_start]
        e2s_offset = len(core_context_middle)
        core_context_middle += f"{s_tokens['e2s']} {e2_trigger} " if add_mark else e2_trigger
        e2e_offset = len(core_context_middle) if add_mark else len(core_context_middle) - 1
        core_context_middle += f"{s_tokens['e2e']}" if add_mark else ""
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
        if add_mark:
            assert core_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
            assert core_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
            assert core_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
            assert core_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        else:
            assert core_context[e1s_offset:e1e_offset+1] == e1_trigger
            assert core_context[e2s_offset:e2e_offset+1] == e2_trigger
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
        e1_core_context_middle = f"{s_tokens['e1s']} {e1_trigger} " if add_mark else e1_trigger
        e1e_offset = len(e1_core_context_middle) if add_mark else len(e1_core_context_middle) - 1
        e1_core_context_middle += f"{s_tokens['e1e']}" if add_mark else ""
        # e2 source sentence
        e2_core_context_before = f"{e2_sent[:e2_sent_start]}"
        e2_core_context_after = f"{e2_sent[e2_sent_start + len(e2_trigger):]}"
        e2s_offset = 0
        e2_core_context_middle = f"{s_tokens['e2s']} {e2_trigger} " if add_mark else e2_trigger
        e2e_offset = len(e2_core_context_middle) if add_mark else len(e2_core_context_middle) - 1
        e2_core_context_middle += f"{s_tokens['e2e']}" if add_mark else ""
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
        if add_mark:
            assert e1_core_context[e1s_offset:e1e_offset] == s_tokens['e1s'] + ' ' + e1_trigger + ' '
            assert e1_core_context[e1e_offset:e1e_offset + len(s_tokens['e1e'])] == s_tokens['e1e']
            assert e2_core_context[e2s_offset:e2e_offset] == s_tokens['e2s'] + ' ' + e2_trigger + ' '
            assert e2_core_context[e2e_offset:e2e_offset + len(s_tokens['e2e'])] == s_tokens['e2e']
        else:
            assert e1_core_context[e1s_offset:e1e_offset+1] == e1_trigger
            assert e2_core_context[e2s_offset:e2e_offset+1] == e2_trigger
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

def create_sample(
    add_mark:bool, 
    e1_sent_idx:int, e1_sent_start:int, e1_trigger:str, 
    e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, 
    sentences:list, sentence_lens:list, 
    model_type, tokenizer, max_length:int
    ) -> dict:

    special_token_dict = {
        'e1s': '[E1_START]', 'e1e': '[E1_END]', 'e2s': '[E2_START]', 'e2e': '[E2_END]'
    } if model_type == 'bert' else {
        'e1s': '<e1_start>', 'e1e': '<e1_end>', 'e2s': '<e2_start>', 'e2e': '<e2_end>'
    }
    if add_mark:
        assert set(special_token_dict.values()).issubset(set(tokenizer.additional_special_tokens))

    context_data = create_event_context(
        add_mark, 
        e1_sent_idx, e1_sent_start, e1_trigger, 
        e2_sent_idx, e2_sent_start, e2_trigger,  
        sentences, sentence_lens, 
        special_token_dict, tokenizer, max_length - 2
    )
    e1s_offset, e1e_offset = context_data['e1s_core_offset'], context_data['e1e_core_offset']
    e2s_offset, e2e_offset = context_data['e2s_core_offset'], context_data['e2e_core_offset']
    if context_data['type'] == 'same_sent': # two events in the same sentence
        sample = context_data['before_context'] + context_data['core_context'] + context_data['after_context']
        e1s_offset, e1e_offset, e2s_offset, e2e_offset = (
            np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + 
            np.full((4,), len(context_data['before_context']))
        )
    else: # two events in different sentences
        sample = (
            context_data['e1_before_context'] + context_data['e1_core_context'] + context_data['e1_after_context'] + ' ' + 
            context_data['e2_before_context'] + context_data['e2_core_context'] + context_data['e2_after_context']
        )
        e1s_offset, e1e_offset = (
            np.asarray([e1s_offset, e1e_offset]) + 
            np.full((2,), len(context_data['e1_before_context']))
        )
        e2s_offset, e2e_offset = (
            np.asarray([e2s_offset, e2e_offset]) + 
            np.full((2,), len(context_data['e1_before_context']) + 
                len(context_data['e1_core_context']) + len(context_data['e1_after_context']) + 
                len(context_data['e2_before_context']) + 1
            )
        )
    if add_mark:
        assert sample[e1s_offset:e1e_offset] == special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert sample[e1e_offset:e1e_offset + len(special_token_dict['e1e'])] == special_token_dict['e1e']
        assert sample[e2s_offset:e2e_offset] == special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert sample[e2e_offset:e2e_offset + len(special_token_dict['e2e'])] == special_token_dict['e2e']
    else:
        assert sample[e1s_offset:e1e_offset+1] == e1_trigger
        assert sample[e2s_offset:e2e_offset+1] == e2_trigger
    return {
        'text': sample, 
        'e1s_offset': e1s_offset, 
        'e1e_offset': e1e_offset, 
        'e2s_offset': e2s_offset, 
        'e2e_offset': e2e_offset
    }
