import os
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.activations import gelu
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

EVENT_SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
id2subtype = {idx: c for idx, c in enumerate(EVENT_SUBTYPES, start=1)}
id2subtype[0] = 'other'
subtype2id = {v: k for k, v in id2subtype.items()}

class RobertaLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def multi_perspective_cosine(cosine_ffnn, cosine_mat_p, cosine_mat_q, batch_event_1_reps, batch_event_2_reps):
    # batch_event_1
    batch_event_1_reps = cosine_ffnn(batch_event_1_reps)
    batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
    batch_event_1_reps = cosine_mat_q * batch_event_1_reps
    batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
    batch_event_1_reps = torch.matmul(batch_event_1_reps, cosine_mat_p)
    batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
    # vector normalization
    norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
    batch_event_1_reps = batch_event_1_reps / norms_1
    # batch_event_2
    batch_event_2_reps = cosine_ffnn(batch_event_2_reps)
    batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
    batch_event_2_reps = cosine_mat_q * batch_event_2_reps
    batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
    batch_event_2_reps = torch.matmul(batch_event_2_reps, cosine_mat_p)
    batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
    # vector normalization
    norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
    batch_event_2_reps = batch_event_2_reps / norms_2
    return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)

class RobertaForMixPrompt(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.cosine_ffnn = nn.Linear(self.hidden_size, 64)
        self.cosine_mat_p = nn.Parameter(torch.rand((4, 128), requires_grad=True))
        self.cosine_mat_q = nn.Parameter(torch.rand((4, 64), requires_grad=True))
        self.coref_mapping = nn.Linear(2 * self.hidden_size + 128, self.hidden_size)
        self.type_match_mapping = nn.Linear(2 * self.hidden_size + 128, self.hidden_size)
        self.arg_match_mapping = nn.Linear(2 * self.hidden_size + 128, self.hidden_size)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        batch_e1_e2_product = batch_event_1_reps * batch_event_2_reps
        batch_multi_cosine = multi_perspective_cosine(
            self.cosine_ffnn, self.cosine_mat_p, self.cosine_mat_q, 
            batch_event_1_reps, batch_event_2_reps
        )
        batch_e1_e2_match = torch.cat([batch_e1_e2_product, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, label_word_id):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        # extract events & matching
        batch_e1_idx, batch_e2_idx = [], []
        for e1s, e1e, e2s, e2e in batch_event_idx:
            batch_e1_idx.append([[e1s, e1e]])
            batch_e2_idx.append([[e2s, e2e]])
        batch_e1_idx, batch_e2_idx = (
            torch.tensor(batch_e1_idx).to(self.use_device), 
            torch.tensor(batch_e2_idx).to(self.use_device)
        )
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        batch_mask_reps = self.coref_mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        coref_logits = self.lm_head(batch_mask_reps)[:, label_word_id]
        return coref_logits

def findall(p, s):
    '''yields all the positions of p in s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)

def cut_sentences(content:str, end_flag:list=['?', '!', '.', '...']):
    sentences, start, tmp = [], 0, ''
    for idx, char in enumerate(content):
        if tmp == '' and char == ' ':
            start += 1
            continue
        tmp += char
        if idx == len(content) - 1:
            sentences.append({
                'text': tmp, 
                'start': start
            })
            break
        if char in end_flag:
            if not content[idx+1] in end_flag:
                sentences.append({
                    'text': tmp, 
                    'start': start
                })
                start, tmp = idx+1, ''
    return sentences

def find_event_sent(event_start, trigger, sent_list):
    '''find out which sentence the event come from
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
    return None

def to_device(device, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k in ['batch_inputs', 'batch_mask_inputs']:
            new_batch_data[k] = {
                k_: v_.to(device) for k_, v_ in v.items()
            }
        elif k in ['batch_event_idx', 'label_word_id', 'match_label_word_id', 'subtype_label_word_id']:
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(device)
    return new_batch_data

class CorefPrompt():
    def __init__(self, model_checkpoint) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._init_model(model_checkpoint)
        self.tokenizer = self._init_tokenizer(model_checkpoint)
        self.verbalizer = self._create_verbalizer()
        self.special_token_dict = self._create_special_token_dict()
        self.document = ''
        self.sentences = ''
        self.sentences_lengths = []

    def _init_model(self, model_checkpoint):
        config = AutoConfig.from_pretrained(model_checkpoint)
        model = RobertaForMixPrompt.from_pretrained(
            model_checkpoint, 
            config=config
        ).to(self.device)
        return model

    def _init_tokenizer(self, model_checkpoint):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return tokenizer

    def _create_verbalizer(self):
        base_verbalizer = {
            'coref': {
                'token': 'same', 'id': self.tokenizer.convert_tokens_to_ids('same')
            } , 
            'non-coref': {
                'token': 'different', 'id': self.tokenizer.convert_tokens_to_ids('different')
            }, 
            'match': {
                'token': '<match>', 'id': self.tokenizer.convert_tokens_to_ids('<match>'), 
                'description': 'same related relevant similar matching matched'
            }, 
            'mismatch': {
                'token': '<mismatch>', 'id': self.tokenizer.convert_tokens_to_ids('<mismatch>'), 
                'description': 'different unrelated irrelevant dissimilar mismatched'
            }
        }
        for subtype, s_id in subtype2id.items():
            base_verbalizer[subtype] = {
                'token': f'<st_{s_id}>', 
                'id': self.tokenizer.convert_tokens_to_ids(f'<st_{s_id}>'), 
                'description': subtype if subtype != 'other' else 'normal'
            }
        return base_verbalizer

    def _create_special_token_dict(self):
        special_token_dict = {
            'mask': '<mask>', 'e1s': '<e1_start>', 'e1e': '<e1_end>', 'e2s': '<e2_start>', 'e2e': '<e2_end>', 
            'l1': '<l1>', 'l2': '<l2>', 'l3': '<l3>', 'l4': '<l4>', 'l5': '<l5>', 'l6': '<l6>', 
            'l7': '<l7>', 'l8': '<l8>', 'l9': '<l9>', 'l10': '<l10>', 
            'match': '<match>', 'mismatch': '<mismatch>', 'refer': '<refer_to>', 'no_refer': '<not_refer_to>'
        }
        for i in range(len(EVENT_SUBTYPES) + 1):
            special_token_dict[f'st{i}'] = f'<st_{i}>'
        return special_token_dict

    def init_document(self, document:str):
        self.document = document
        self.sentences = cut_sentences(document)
        self.sentences_lengths = [
            len(self.tokenizer.tokenize(sent['text'])) for sent in self.sentences
        ]
    
    def _convert_args_to_str(self, args:list):
        participants, places = (
            [arg for arg in args if arg['role'] == 'participant'], 
            [arg for arg in args if arg['role'] == 'place']
        )
        arg_str = ''
        if participants:
            arg_str = f"with {', '.join([arg['mention'] for arg in participants])} as participants"
        if places:
            arg_str += f" at {', '.join([arg['mention'] for arg in places])}"
        return arg_str.strip()

    def _create_mix_template(self, e1_trigger:str, e2_trigger:str, e1_arg_str: str, e2_arg_str: str) -> dict:
        # prefix template
        prefix_trigger_offsets = []
        prefix_template = f"In the following text, the focus is on the events expressed by {self.special_token_dict['e1s']} "
        prefix_trigger_offsets.append([len(prefix_template), len(prefix_template) + len(e1_trigger) - 1])
        prefix_template += f"{e1_trigger} {self.special_token_dict['e1e']} and {self.special_token_dict['e2s']} "
        prefix_trigger_offsets.append([len(prefix_template), len(prefix_template) + len(e2_trigger) - 1])
        prefix_template += f"{e2_trigger} {self.special_token_dict['e2e']}, and it needs to judge whether they refer to the same or different events: "
        # anchor template
        e1_anchor_temp = "Here "
        e1s_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{self.special_token_dict['e1s']} {e1_trigger} "
        e1e_anchor_offset = len(e1_anchor_temp)
        e1_anchor_temp += f"{self.special_token_dict['e1e']} expresses a {self.special_token_dict['mask']} event"
        e2_anchor_temp = "Here "
        e2s_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{self.special_token_dict['e2s']} {e2_trigger} "
        e2e_anchor_offset = len(e2_anchor_temp)
        e2_anchor_temp += f"{self.special_token_dict['e2e']} expresses a {self.special_token_dict['mask']} event" 
        e1_anchor_temp += f"{' ' + e1_arg_str if e1_arg_str else ''}."
        e2_anchor_temp += f"{' ' + e2_arg_str if e2_arg_str else ''}."
        # inference template
        infer_trigger_offsets = []
        infer_template = f"In conclusion, the events expressed by {self.special_token_dict['e1s']} "
        infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e1_trigger) - 1])
        infer_template += f"{e1_trigger} {self.special_token_dict['e1e']} and {self.special_token_dict['e2s']} "
        infer_trigger_offsets.append([len(infer_template), len(infer_template) + len(e2_trigger) - 1])
        infer_template += f"{e2_trigger} {self.special_token_dict['e2e']}"
        infer_template += f" have {self.special_token_dict['mask']} event type and {self.special_token_dict['mask']} participants"
        infer_template += f", so they refer to {self.special_token_dict['mask']} event."
        return {
            'prefix_template': prefix_template, 
            'e1_anchor_template': e1_anchor_temp, 
            'e2_anchor_template': e2_anchor_temp, 
            'infer_template': infer_template, 
            'prefix_trigger_offsets': prefix_trigger_offsets, 
            'infer_trigger_offsets': infer_trigger_offsets, 
            'e1s_anchor_offset': e1s_anchor_offset, 
            'e1e_anchor_offset': e1e_anchor_offset, 
            'e2s_anchor_offset': e2s_anchor_offset, 
            'e2e_anchor_offset': e2e_anchor_offset, 
        }
    
    def _create_event_context(self, 
        e1_sent_idx:int, e1_sent_start:int, e1_trigger:str,  
        e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, 
        max_length:int
        ) -> dict:
        if e1_sent_idx == e2_sent_idx: # two events in the same sentence
            assert e1_sent_start < e2_sent_start
            e1_e2_sent = self.sentences[e1_sent_idx]['text']
            core_context_before = f"{e1_e2_sent[:e1_sent_start]}"
            core_context_after = f"{e1_e2_sent[e2_sent_start + len(e2_trigger):]}"
            e1s_offset = 0
            core_context_middle = f"{self.special_token_dict['e1s']} {e1_trigger} "
            e1e_offset = len(core_context_middle)
            core_context_middle += f"{self.special_token_dict['e1e']}{e1_e2_sent[e1_sent_start + len(e1_trigger):e2_sent_start]}"
            e2s_offset = len(core_context_middle)
            core_context_middle += f"{self.special_token_dict['e2s']} {e2_trigger} "
            e2e_offset = len(core_context_middle)
            core_context_middle += f"{self.special_token_dict['e2e']}"
            # segment contain the two events
            core_context = core_context_before + core_context_middle + core_context_after
            total_length = len(self.tokenizer.tokenize(core_context))
            before_context, after_context = '', ''
            if total_length > max_length: # cut segment
                before_after_length = (max_length - len(self.tokenizer.tokenize(core_context_middle))) // 2
                core_context_before = self.tokenizer.decode(self.tokenizer.encode(core_context_before)[1:-1][-before_after_length:])
                core_context_after = self.tokenizer.decode(self.tokenizer.encode(core_context_after)[1:-1][:before_after_length])
                core_context = core_context_before + core_context_middle + core_context_after
                e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + np.full((4,), len(core_context_before))
            else: # create contexts before/after the host sentence
                e1s_offset, e1e_offset, e2s_offset, e2e_offset = np.asarray([e1s_offset, e1e_offset, e2s_offset, e2e_offset]) + np.full((4,), len(core_context_before))
                e_before, e_after = e1_sent_idx - 1, e1_sent_idx + 1
                while True:
                    if e_before >= 0:
                        if total_length + self.sentences_lengths[e_before] <= max_length:
                            before_context = self.sentences[e_before]['text'] + ' ' + before_context
                            total_length += 1 + self.sentences_lengths[e_before]
                            e_before -= 1
                        else:
                            e_before = -1
                    if e_after < len(self.sentences):
                        if total_length + self.sentences_lengths[e_after] <= max_length:
                            after_context += ' ' + self.sentences[e_after]['text']
                            total_length += 1 + self.sentences_lengths[e_after]
                            e_after += 1
                        else:
                            e_after = len(self.sentences)
                    if e_before == -1 and e_after == len(self.sentences):
                        break
            tri1s_core_offset, tri1e_core_offset = e1s_offset + len(self.special_token_dict['e1s']) + 1, e1e_offset - 2
            tri2s_core_offset, tri2e_core_offset = e2s_offset + len(self.special_token_dict['e2s']) + 1, e2e_offset - 2
            assert core_context[e1s_offset:e1e_offset] == self.special_token_dict['e1s'] + ' ' + e1_trigger + ' '
            assert core_context[e1e_offset:e1e_offset + len(self.special_token_dict['e1e'])] == self.special_token_dict['e1e']
            assert core_context[e2s_offset:e2e_offset] == self.special_token_dict['e2s'] + ' ' + e2_trigger + ' '
            assert core_context[e2e_offset:e2e_offset + len(self.special_token_dict['e2e'])] == self.special_token_dict['e2e']
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
            e1_sent, e2_sent = self.sentences[e1_sent_idx]['text'], self.sentences[e2_sent_idx]['text']
            # e1 source sentence
            e1_core_context_before = f"{e1_sent[:e1_sent_start]}"
            e1_core_context_after = f"{e1_sent[e1_sent_start + len(e1_trigger):]}"
            e1s_offset = 0
            e1_core_context_middle = f"{self.special_token_dict['e1s']} {e1_trigger} "
            e1e_offset = len(e1_core_context_middle)
            e1_core_context_middle += f"{self.special_token_dict['e1e']}"
            # e2 source sentence
            e2_core_context_before = f"{e2_sent[:e2_sent_start]}"
            e2_core_context_after = f"{e2_sent[e2_sent_start + len(e2_trigger):]}"
            e2s_offset = 0
            e2_core_context_middle = f"{self.special_token_dict['e2s']} {e2_trigger} "
            e2e_offset = len(e2_core_context_middle)
            e2_core_context_middle += f"{self.special_token_dict['e2e']}"
            # segment contain the two events
            e1_core_context = e1_core_context_before + e1_core_context_middle + e1_core_context_after
            e2_core_context = e2_core_context_before + e2_core_context_middle + e2_core_context_after
            total_length = len(self.tokenizer.tokenize(e1_core_context)) + len(self.tokenizer.tokenize(e2_core_context))
            e1_before_context, e1_after_context, e2_before_context, e2_after_context = '', '', '', ''
            if total_length > max_length:
                e1_e2_middle_length = len(self.tokenizer.tokenize(e1_core_context_middle)) + len(self.tokenizer.tokenize(e2_core_context_middle))
                before_after_length = (max_length - e1_e2_middle_length) // 4
                e1_core_context_before = self.tokenizer.decode(self.tokenizer.encode(e1_core_context_before)[1:-1][-before_after_length:])
                e1_core_context_after = self.tokenizer.decode(self.tokenizer.encode(e1_core_context_after)[1:-1][:before_after_length])
                e1_core_context = e1_core_context_before + e1_core_context_middle + e1_core_context_after
                e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(e1_core_context_before))
                e2_core_context_before = self.tokenizer.decode(self.tokenizer.encode(e2_core_context_before)[1:-1][-before_after_length:])
                e2_core_context_after = self.tokenizer.decode(self.tokenizer.encode(e2_core_context_after)[1:-1][:before_after_length])
                e2_core_context = e2_core_context_before + e2_core_context_middle + e2_core_context_after
                e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(e2_core_context_before))
            else: # add other sentences
                e1s_offset, e1e_offset = np.asarray([e1s_offset, e1e_offset]) + np.full((2,), len(e1_core_context_before))
                e2s_offset, e2e_offset = np.asarray([e2s_offset, e2e_offset]) + np.full((2,), len(e2_core_context_before))
                e1_before, e1_after, e2_before, e2_after = e1_sent_idx - 1, e1_sent_idx + 1, e2_sent_idx - 1, e2_sent_idx + 1
                while True:
                    e1_after_dead, e2_before_dead = False, False
                    if e1_before >= 0:
                        if total_length + self.sentences_lengths[e1_before] <= max_length:
                            e1_before_context = self.sentences[e1_before]['text'] + ' ' + e1_before_context
                            total_length += 1 + self.sentences_lengths[e1_before]
                            e1_before -= 1
                        else:
                            e1_before = -1
                    if e1_after <= e2_before:
                        if total_length + self.sentences_lengths[e1_after] <= max_length:
                            e1_after_context += ' ' + self.sentences[e1_after]['text']
                            total_length += 1 + self.sentences_lengths[e1_after]
                            e1_after += 1
                        else:
                            e1_after_dead = True
                    if e2_before >= e1_after:
                        if total_length + self.sentences_lengths[e2_before] <= max_length:
                            e2_before_context = self.sentences[e2_before]['text'] + ' ' + e2_before_context
                            total_length += 1 + self.sentences_lengths[e2_before]
                            e2_before -= 1
                        else:
                            e2_before_dead = True
                    if e2_after < len(self.sentences):
                        if total_length + self.sentences_lengths[e2_after] <= max_length:
                            e2_after_context += ' ' + self.sentences[e2_after]['text']
                            total_length += 1 + self.sentences_lengths[e2_after]
                            e2_after += 1
                        else:
                            e2_after = len(self.sentences)
                    if e1_before == -1 and e2_after == len(self.sentences) and ((e1_after_dead and e2_before_dead) or e1_after > e2_before):
                        break
            tri1s_core_offset, tri1e_core_offset = e1s_offset + len(self.special_token_dict['e1s']) + 1, e1e_offset - 2
            tri2s_core_offset, tri2e_core_offset = e2s_offset + len(self.special_token_dict['e2s']) + 1, e2e_offset - 2
            assert e1_core_context[e1s_offset:e1e_offset] == self.special_token_dict['e1s'] + ' ' + e1_trigger + ' '
            assert e1_core_context[e1e_offset:e1e_offset + len(self.special_token_dict['e1e'])] == self.special_token_dict['e1e']
            assert e2_core_context[e2s_offset:e2e_offset] == self.special_token_dict['e2s'] + ' ' + e2_trigger + ' '
            assert e2_core_context[e2e_offset:e2e_offset + len(self.special_token_dict['e2e'])] == self.special_token_dict['e2e']
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

    def _create_prompt(self, 
        e1_sent_idx:int, e1_sent_start:int, e1_trigger:str, e1_args: list, 
        e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, e2_args: list
        ):
        e1_arg_str = self._convert_args_to_str(e1_args)
        e2_arg_str = self._convert_args_to_str(e2_args)
        template_data = self._create_mix_template(e1_trigger, e2_trigger, e1_arg_str, e2_arg_str)
        template_length = (
            len(self.tokenizer.tokenize(template_data['prefix_template'])) + 
            len(self.tokenizer.tokenize(template_data['e1_anchor_template'])) + 
            len(self.tokenizer.tokenize(template_data['e2_anchor_template'])) + 
            len(self.tokenizer.tokenize(template_data['infer_template'])) + 
            6
        )
        trigger_offsets = template_data['prefix_trigger_offsets']
        context_data = self._create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger, 
            512 - template_length
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
        mask_offsets = list(findall(self.special_token_dict['mask'], prompt))
        assert len(mask_offsets) == 5
        e1_type_mask_offset, e2_type_mask_offset, type_match_mask_offset, arg_match_mask_offset, mask_offset = mask_offsets
        assert prompt[e1_type_mask_offset:e1_type_mask_offset + len(self.special_token_dict['mask'])] == self.special_token_dict['mask']
        assert prompt[e2_type_mask_offset:e2_type_mask_offset + len(self.special_token_dict['mask'])] == self.special_token_dict['mask']
        trigger_offsets.append([e1s_context_offset + len(self.special_token_dict['e1s']) + 1, e1e_context_offset - 2])
        trigger_offsets.append([e2s_context_offset + len(self.special_token_dict['e2s']) + 1, e2e_context_offset - 2])
        trigger_offsets.append([e1s_offset + len(self.special_token_dict['e1s']) + 1, e1e_offset - 2])
        trigger_offsets.append([e2s_offset + len(self.special_token_dict['e2s']) + 1, e2e_offset - 2])
        trigger_offsets += infer_trigger_offsets
        assert prompt[type_match_mask_offset:type_match_mask_offset + len(self.special_token_dict['mask'])] == self.special_token_dict['mask']
        assert prompt[arg_match_mask_offset:arg_match_mask_offset + len(self.special_token_dict['mask'])] == self.special_token_dict['mask']
        assert prompt[mask_offset:mask_offset + len(self.special_token_dict['mask'])] == self.special_token_dict['mask']
        assert prompt[e1s_offset:e1e_offset] == self.special_token_dict['e1s'] + ' ' + e1_trigger + ' '
        assert prompt[e1e_offset:e1e_offset + len(self.special_token_dict['e1e'])] == self.special_token_dict['e1e']
        assert prompt[e2s_offset:e2e_offset] == self.special_token_dict['e2s'] + ' ' + e2_trigger + ' '
        assert prompt[e2e_offset:e2e_offset + len(self.special_token_dict['e2e'])] == self.special_token_dict['e2e']
        for s, e in trigger_offsets:
            assert prompt[s:e+1] == e1_trigger or prompt[s:e+1] == e2_trigger
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
            'trigger_offsets': trigger_offsets
        }
    
    def predict_event_pair_coref(self,  
        e1_offset:int, e1_trigger:str, e1_args:list, 
        e2_offset:int, e2_trigger:str, e2_args:list
        ):
        '''
        # Args:
            - e1/e2_offset: event offset in the document
            - e1/e2_trigger: event trigger word
            - e1/e2_args: event arguments, [{'mention', 'role'},...], role shoud be 'participant' or 'place'
        # Returns: \n
            { \n
                'label': 'coref' or 'non-coref', \n
                'probability': probability \n
            }
        '''
        if not self.document or self.document[e1_offset:e1_offset + len(e1_trigger)] != e1_trigger or self.document[e2_offset:e2_offset + len(e2_trigger)] != e2_trigger:
            raise ValueError(f"Can't find event '{e1_trigger}' or '{e2_trigger}' in the document. You should first run `init_document(text)` to load text.")
        e1_sent_idx, e1_sent_start = find_event_sent(e1_offset, e1_trigger, self.sentences)
        e2_sent_idx, e2_sent_start = find_event_sent(e2_offset, e2_trigger, self.sentences)
        prompt_data = self._create_prompt(
            e1_sent_idx, e1_sent_start, e1_trigger, e1_args, 
            e2_sent_idx, e2_sent_start, e2_trigger, e2_args
        )
        prompt_text = prompt_data['prompt']
        # convert char offsets to token idxs
        encoding = self.tokenizer(prompt_text)
        mask_idx = encoding.char_to_token(prompt_data['mask_offset'])
        e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
            encoding.char_to_token(prompt_data['e1s_offset']), 
            encoding.char_to_token(prompt_data['e1e_offset']), 
            encoding.char_to_token(prompt_data['e2s_offset']), 
            encoding.char_to_token(prompt_data['e2e_offset'])
        )
        assert None not in [mask_idx, e1s_idx, e1e_idx, e2s_idx, e2e_idx]
        event_idx = [e1s_idx, e1e_idx, e2s_idx, e2e_idx]
        inputs = self.tokenizer(
            prompt_text, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {
            'batch_inputs': inputs, 
            'batch_mask_idx': [mask_idx], 
            'batch_event_idx': [event_idx], 
            'label_word_id': [self.verbalizer['non-coref']['id'], self.verbalizer['coref']['id']]
        }
        inputs = to_device(self.device, inputs)
        with torch.no_grad():
            logits = self.model(**inputs)
            prob = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred = logits.argmax(dim=-1)[0].item()
        prob = prob[pred].item()
        return {
            'prompt': prompt_text, 
            'label': 'coref' if pred == 1 else 'non-coref', 
            'probability': prob
        }
    
    def predict_coref(self, event1, event2):
        '''
        # Args:
            - event1/2: {\n
                'offset': event offset in the document, \n
                'trigger': event trigger word, \n
                'args': event arguments, [{'mention', 'role'},...], role shoud be 'participant' or 'place'\n
            }
        # Returns: \n
            { \n
                'label': 'coref' or 'non-coref', \n
                'probability': probability \n
            }
        '''
        return self.predict_event_pair_coref(
            event1['offset'], event1['trigger'], event1['args'], 
            event2['offset'], event2['trigger'], event2['args'], 
        )
    
    def predict_coref_in_doc(self, document, event1, event2):
        '''
        # Args:
            - document: document that host the events
            - event1/2: {\n
                'offset': event offset in the document, \n
                'trigger': event trigger word, \n
                'args': event arguments, [{'mention', 'role'},...], role shoud be 'participant' or 'place'\n
            }
        # Returns: \n
            { \n
                'label': 'coref' or 'non-coref', \n
                'probability': probability \n
            }
        '''
        self.init_document(document)
        return self.predict_coref(event1, event2)
