import os
from collections import namedtuple
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoTokenizer
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.activations import gelu
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from coref_prompt.prompt import EVENT_SUBTYPES, id2subtype
from coref_prompt.prompt import create_mix_template, create_event_context, create_verbalizer, get_special_tokens

args = namedtuple('args', [
    'model_type'
    'model_checkpoint'
    'best_weights'
    'matching_style', 
    'device', 
    'prompt_type', 
    'cosine_space_dim', 
    'cosine_slices', 
    'cosine_factor'
])
args.model_type = 'roberta'
args.model_checkpoint=''
args.best_weights=''
args.matching_style = 'product_cosine'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.prompt_type = 'm_hta_hn'
args.cosine_space_dim = 64
args.cosine_slices = 128
args.cosine_factor = 4

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
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        self.remove_match = (args.prompt_type == 'ma_remove-match')
        self.remove_subtype_match = (args.prompt_type == 'ma_remove-subtype-match')
        self.remove_arg_match = (args.prompt_type == 'ma_remove-arg-match')
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'product':
                self.coref_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
                if not self.remove_match:
                    if not self.remove_subtype_match:
                        self.type_match_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
                    if not self.remove_arg_match:
                        self.arg_match_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (
                    args.cosine_space_dim, args.cosine_slices, args.cosine_factor
                )
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                if self.matching_style == 'cosine':
                    self.coref_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                    if not self.remove_match:
                        if not self.remove_subtype_match:
                            self.type_match_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                        if not self.remove_arg_match:
                            self.arg_match_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'product_cosine':
                    self.coref_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                    if not self.remove_match:
                        if not self.remove_subtype_match:
                            self.type_match_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                        if not self.remove_arg_match:
                            self.arg_match_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'product':
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = multi_perspective_cosine(
                self.cosine_ffnn, self.cosine_mat_p, self.cosine_mat_q, 
                batch_event_1_reps, batch_event_2_reps
            )
        elif self.matching_style == 'product_cosine':
            batch_e1_e2_product = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = multi_perspective_cosine(
                self.cosine_ffnn, self.cosine_mat_p, self.cosine_mat_q, 
                batch_event_1_reps, batch_event_2_reps
            )
            batch_e1_e2_match = torch.cat([batch_e1_e2_product, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, 
        batch_inputs, batch_mask_idx, 
        batch_event_idx, batch_t1_mask_idx, batch_t2_mask_idx, 
        label_word_id, subtype_label_word_id, match_label_word_id=None, 
        batch_mask_inputs=None, batch_type_match_mask_idx=None, batch_arg_match_mask_idx=None, 
        labels=None, subtype_match_labels=None, arg_match_labels=None, e1_subtype_labels=None, e2_subtype_labels=None
        ):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if not self.remove_match:
            if not self.remove_subtype_match:
                batch_type_match_mask_reps = batched_index_select(sequence_output, 1, batch_type_match_mask_idx.unsqueeze(-1)).squeeze(1)
            if not self.remove_arg_match:
                batch_arg_match_mask_reps = batched_index_select(sequence_output, 1, batch_arg_match_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t1_mask_reps = batched_index_select(sequence_output, 1, batch_t1_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t2_mask_reps = batched_index_select(sequence_output, 1, batch_t2_mask_idx.unsqueeze(-1)).squeeze(1)
        if batch_mask_inputs is not None:
            mask_outputs = self.roberta(**batch_mask_inputs)
            mask_sequence_output = mask_outputs.last_hidden_state
            batch_mask_mask_reps = batched_index_select(mask_sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
            if not self.remove_match:
                if not self.remove_subtype_match:
                    batch_mask_type_match_mask_reps = batched_index_select(mask_sequence_output, 1, batch_type_match_mask_idx.unsqueeze(-1)).squeeze(1)
                if not self.remove_arg_match:
                    batch_mask_arg_match_mask_reps = batched_index_select(mask_sequence_output, 1, batch_arg_match_mask_idx.unsqueeze(-1)).squeeze(1)
            batch_mask_t1_mask_reps = batched_index_select(mask_sequence_output, 1, batch_t1_mask_idx.unsqueeze(-1)).squeeze(1)
            batch_mask_t2_mask_reps = batched_index_select(mask_sequence_output, 1, batch_t2_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
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
            if not self.remove_match:
                if not self.remove_arg_match:
                    batch_arg_match_mask_reps = self.arg_match_mapping(torch.cat([batch_arg_match_mask_reps, batch_match_reps], dim=-1))
                if not self.remove_subtype_match:
                    batch_subtype_match_reps = self._matching_func(batch_t1_mask_reps, batch_t2_mask_reps)
                    batch_type_match_mask_reps = self.type_match_mapping(torch.cat([batch_type_match_mask_reps, batch_subtype_match_reps], dim=-1))
            if batch_mask_inputs is not None:
                batch_mask_event_1_reps = self.span_extractor(mask_sequence_output, batch_e1_idx).squeeze(dim=1)
                batch_mask_event_2_reps = self.span_extractor(mask_sequence_output, batch_e2_idx).squeeze(dim=1)
                batch_mask_match_reps = self._matching_func(batch_mask_event_1_reps, batch_mask_event_2_reps)
                batch_mask_mask_reps = self.coref_mapping(torch.cat([batch_mask_mask_reps, batch_mask_match_reps], dim=-1))
                if not self.remove_match:
                    if not self.remove_arg_match:
                        batch_mask_arg_match_mask_reps = self.arg_match_mapping(torch.cat([batch_mask_arg_match_mask_reps, batch_mask_match_reps], dim=-1))
                    if not self.remove_subtype_match:
                        batch_mask_subtype_match_reps = self._matching_func(batch_mask_t1_mask_reps, batch_mask_t2_mask_reps)
                        batch_mask_type_match_mask_reps = self.type_match_mapping(torch.cat([batch_mask_type_match_mask_reps, batch_mask_subtype_match_reps], dim=-1))
        coref_logits = self.lm_head(batch_mask_reps)[:, label_word_id]
        if not self.remove_match:
            if not self.remove_subtype_match:
                type_match_logits = self.lm_head(batch_type_match_mask_reps)[:, match_label_word_id]
            if not self.remove_arg_match:
                arg_match_logits = self.lm_head(batch_arg_match_mask_reps)[:, match_label_word_id]
        e1_type_logits = self.lm_head(batch_t1_mask_reps)[:, subtype_label_word_id]
        e2_type_logits = self.lm_head(batch_t2_mask_reps)[:, subtype_label_word_id]
        if batch_mask_inputs is not None:
            mask_coref_logits = self.lm_head(batch_mask_mask_reps)[:, label_word_id]
            if not self.remove_match:
                if not self.remove_subtype_match:
                    mask_type_match_logits = self.lm_head(batch_mask_type_match_mask_reps)[:, match_label_word_id]
                if not self.remove_arg_match:
                    mask_arg_match_logits = self.lm_head(batch_mask_arg_match_mask_reps)[:, match_label_word_id]
            mask_e1_type_logits = self.lm_head(batch_mask_t1_mask_reps)[:, subtype_label_word_id]
            mask_e2_type_logits = self.lm_head(batch_mask_t2_mask_reps)[:, subtype_label_word_id]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            coref_loss = loss_fct(coref_logits, labels)
            subtype_loss = 0.5 * loss_fct(e1_type_logits, e1_subtype_labels) + 0.5 * loss_fct(e2_type_logits, e2_subtype_labels)
            if not self.remove_match:
                if self.remove_subtype_match:
                    match_loss = loss_fct(arg_match_logits, arg_match_labels)
                elif self.remove_arg_match:
                    match_loss = loss_fct(type_match_logits, subtype_match_labels)
                else:
                    match_loss = 0.5 * loss_fct(type_match_logits, subtype_match_labels) + 0.5 * loss_fct(arg_match_logits, arg_match_labels)
                loss = torch.log(1 + coref_loss) + torch.log(1 + match_loss) + torch.log(1 + subtype_loss)
            else:
                loss = torch.log(1 + coref_loss) + torch.log(1 + subtype_loss)
            if batch_mask_inputs is not None:
                mask_coref_loss = loss_fct(mask_coref_logits, labels)
                mask_subtype_loss = 0.5 * loss_fct(mask_e1_type_logits, e1_subtype_labels) + 0.5 * loss_fct(mask_e2_type_logits, e2_subtype_labels)
                if not self.remove_match:
                    if self.remove_subtype_match:
                        mask_match_loss = loss_fct(mask_arg_match_logits, arg_match_labels)
                    elif self.remove_arg_match:
                        mask_match_loss = loss_fct(mask_type_match_logits, subtype_match_labels)
                    else:
                        mask_match_loss = 0.5 * loss_fct(mask_type_match_logits, subtype_match_labels) + 0.5 * loss_fct(mask_arg_match_logits, arg_match_labels)
                    mask_loss = torch.log(1 + mask_coref_loss) + torch.log(1 + mask_match_loss) + torch.log(1 + mask_subtype_loss)
                else:
                    mask_loss = torch.log(1 + mask_coref_loss) + torch.log(1 + mask_subtype_loss)
                loss = 0.5 * loss + 0.5 * mask_loss
        return loss, coref_logits

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
    def __init__(self, model_checkpoint, best_weights, args) -> None:
        self.model = self._init_model(model_checkpoint, args)
        self.device = args.device
        self.tokenizer = self._init_tokenizer(model_checkpoint, args.model_type)
        self.verbalizer = self._init_verbalizer(args.model_type, args.prompt_type)
        # load best weights
        self.model.load_state_dict(
            torch.load(os.path.join(best_weights), map_location=torch.device(self.device))
        )
        self.special_token_dict = self._init_special_token_dict()
    
    def _init_model(self, model_checkpoint, args):
        config = AutoConfig.from_pretrained(model_checkpoint)
        model = RobertaForMixPrompt.from_pretrained(
            model_checkpoint,
            config=config, 
            args=args
        ).to(args.device)
        return model

    def _init_tokenizer(self, model_checkpoint, model_type):
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # add special tokens
        sp_tokens = (
            get_special_tokens(model_type, 'base') + 
            get_special_tokens(model_type, 'match') + 
            get_special_tokens(model_type, 'event_subtype')
        )
        tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
        self.model.resize_token_embeddings(len(tokenizer))
        return tokenizer

    def _init_verbalizer(self, model_type, prompt_type):
        verbalizer = create_verbalizer(self.tokenizer, model_type, prompt_type)
        # initialize embeddings for 'match' and 'mismatch' token
        subtype_sp_token_num = len(EVENT_SUBTYPES) + 1
        match_idx, mismatch_idx = -(subtype_sp_token_num+2), -(subtype_sp_token_num+1)
        with torch.no_grad():
            match_tokenized = self.tokenizer.tokenize(verbalizer['match']['description'])
            match_tokenized_ids = self.tokenizer.convert_tokens_to_ids(match_tokenized)
            mismatch_tokenized = self.tokenizer.tokenize(verbalizer['mismatch']['description'])
            mismatch_tokenized_ids = self.tokenizer.convert_tokens_to_ids(mismatch_tokenized)
            new_embedding = self.model.roberta.embeddings.word_embeddings.weight[match_tokenized_ids].mean(axis=0)
            self.model.roberta.embeddings.word_embeddings.weight[match_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            new_embedding = self.model.roberta.embeddings.word_embeddings.weight[mismatch_tokenized_ids].mean(axis=0)
            self.model.roberta.embeddings.word_embeddings.weight[mismatch_idx, :] = new_embedding.clone().detach().requires_grad_(True)
        # initialize embeddings for event subtype special tokens
        subtype_descriptions = [
            verbalizer[id2subtype[s_id]]['description'] for s_id in range(len(EVENT_SUBTYPES) + 1)
        ]
        with torch.no_grad():
            for i, description in enumerate(reversed(subtype_descriptions), start=1):
                tokenized = self.tokenizer.tokenize(description)
                tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
                new_embedding = self.model.roberta.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
                self.model.roberta.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
        return verbalizer
    
    def _init_special_token_dict(self):
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
    
    def _create_prompt(self, 
        e1_sent_idx:int, e1_sent_start:int, e1_trigger:str, e1_args: list, 
        e2_sent_idx:int, e2_sent_start:int, e2_trigger:str, e2_args: list
        ):
        e1_arg_str = self._convert_args_to_str(e1_args)
        e2_arg_str = self._convert_args_to_str(e2_args)
        template_data = create_mix_template(e1_trigger, e2_trigger, e1_arg_str, e2_arg_str, '', '', 'm_hta_hn', self.special_token_dict)
        template_length = (
            len(self.tokenizer.tokenize(template_data['prefix_template'])) + 
            len(self.tokenizer.tokenize(template_data['e1_anchor_template'])) + 
            len(self.tokenizer.tokenize(template_data['e2_anchor_template'])) + 
            len(self.tokenizer.tokenize(template_data['infer_template'])) + 
            6
        )
        trigger_offsets = template_data['prefix_trigger_offsets']
        context_data = create_event_context(
            e1_sent_idx, e1_sent_start, e1_trigger, 
            e2_sent_idx, e2_sent_start, e2_trigger,  
            self.sentences, self.sentences_lengths, 
            self.special_token_dict, self.tokenizer, 512 - template_length
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
        assert self.document[e1_offset:e1_offset + len(e1_trigger)] == e1_trigger, 'You should first run `init_document(text)` to load text.'
        assert self.document[e2_offset:e2_offset + len(e2_trigger)] == e2_trigger, 'You should first run `init_document(text)` to load text.'
        e1_sent_idx, e1_sent_start = find_event_sent(e1_offset, e1_trigger, self.sentences)
        e2_sent_idx, e2_sent_start = find_event_sent(e2_offset, e2_trigger, self.sentences)
        prompt_data = self._create_prompt(
            e1_sent_idx, e1_sent_start, e1_trigger, e1_args, 
            e2_sent_idx, e2_sent_start, e2_trigger, e2_args
        )
        prompt_text = prompt_data['prompt']
        # convert char offsets to token idxs
        encoding = self.tokenizer(prompt_text)
        mask_idx, type_match_mask_idx, arg_match_mask_idx = (
            encoding.char_to_token(prompt_data['mask_offset']), 
            encoding.char_to_token(prompt_data['type_match_mask_offset']), 
            encoding.char_to_token(prompt_data['arg_match_mask_offset']), 
        )
        e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
            encoding.char_to_token(prompt_data['e1s_offset']), 
            encoding.char_to_token(prompt_data['e1e_offset']), 
            encoding.char_to_token(prompt_data['e2s_offset']), 
            encoding.char_to_token(prompt_data['e2e_offset'])
        )
        e1_type_mask_idx, e2_type_mask_idx = (
            encoding.char_to_token(prompt_data['e1_type_mask_offset']), 
            encoding.char_to_token(prompt_data['e2_type_mask_offset'])
        )
        assert None not in [
            mask_idx, type_match_mask_idx, arg_match_mask_idx, 
            e1s_idx, e1e_idx, e2s_idx, e2e_idx, e1_type_mask_idx, e2_type_mask_idx
        ]
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
            'batch_type_match_mask_idx': [type_match_mask_idx], 
            'batch_arg_match_mask_idx': [arg_match_mask_idx], 
            'batch_event_idx': [event_idx], 
            'batch_t1_mask_idx': [e1_type_mask_idx], 
            'batch_t2_mask_idx': [e2_type_mask_idx], 
            'label_word_id': [self.verbalizer['non-coref']['id'], self.verbalizer['coref']['id']], 
            'match_label_word_id': [self.verbalizer['match']['id'], self.verbalizer['mismatch']['id']], 
            'subtype_label_word_id': [
                self.verbalizer[id2subtype[s_id]]['id'] 
                for s_id in range(len(EVENT_SUBTYPES) + 1)
            ]
        }
        inputs = to_device(self.device, inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[1]
            prob = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred = logits.argmax(dim=-1)[0].item()
        prob = prob[pred].item()
        return {
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
    
