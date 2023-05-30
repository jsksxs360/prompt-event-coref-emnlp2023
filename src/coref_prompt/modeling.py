import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from  ..tools import batched_index_select, BertOnlyMLMHead, RobertaLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

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

class BertForBasePrompt(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'product':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (
                    args.cosine_space_dim, args.cosine_slices, args.cosine_factor
                )
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'product_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
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
    
    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, label_word_id, batch_mask_inputs=None, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if batch_mask_inputs is not None:
            mask_outputs = self.bert(**batch_mask_inputs)
            mask_sequence_output = mask_outputs.last_hidden_state
            batch_mask_mask_reps = batched_index_select(mask_sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
            if batch_mask_inputs is not None:
                batch_mask_event_1_reps = self.span_extractor(mask_sequence_output, batch_e1_idx).squeeze(dim=1)
                batch_mask_event_2_reps = self.span_extractor(mask_sequence_output, batch_e2_idx).squeeze(dim=1)
                batch_mask_match_reps = self._matching_func(batch_mask_event_1_reps, batch_mask_event_2_reps)
                batch_mask_mask_reps = self.mapping(torch.cat([batch_mask_mask_reps, batch_mask_match_reps], dim=-1))
        logits = self.cls(batch_mask_reps)[:, label_word_id]
        if batch_mask_inputs is not None:
            mask_logits = self.cls(batch_mask_mask_reps)[:, label_word_id]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if batch_mask_inputs is not None:
                loss = 0.5 * loss + 0.5 * loss_fct(mask_logits, labels)
        return loss, logits

class RobertaForBasePrompt(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'product':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (
                    args.cosine_space_dim, args.cosine_slices, args.cosine_factor
                )
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'product_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, label_word_id, batch_mask_inputs=None, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if batch_mask_inputs is not None:
            mask_outputs = self.roberta(**batch_mask_inputs)
            mask_sequence_output = mask_outputs.last_hidden_state
            batch_mask_mask_reps = batched_index_select(mask_sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
            if batch_mask_inputs is not None:
                batch_mask_event_1_reps = self.span_extractor(mask_sequence_output, batch_e1_idx).squeeze(dim=1)
                batch_mask_event_2_reps = self.span_extractor(mask_sequence_output, batch_e2_idx).squeeze(dim=1)
                batch_mask_match_reps = self._matching_func(batch_mask_event_1_reps, batch_mask_event_2_reps)
                batch_mask_mask_reps = self.mapping(torch.cat([batch_mask_mask_reps, batch_mask_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)[:, label_word_id]
        if batch_mask_inputs is not None:
            mask_logits = self.lm_head(batch_mask_mask_reps)[:, label_word_id]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if batch_mask_inputs is not None:
                loss = 0.5 * loss + 0.5 * loss_fct(mask_logits, labels)
        return loss, logits

class BertForMixPrompt(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
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
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
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
        outputs = self.bert(**batch_inputs)
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
            mask_outputs = self.bert(**batch_mask_inputs)
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
        coref_logits = self.cls(batch_mask_reps)[:, label_word_id]
        if not self.remove_match:
            if not self.remove_subtype_match:
                type_match_logits = self.cls(batch_type_match_mask_reps)[:, match_label_word_id]
            if not self.remove_arg_match:
                arg_match_logits = self.cls(batch_arg_match_mask_reps)[:, match_label_word_id]
        e1_type_logits = self.cls(batch_t1_mask_reps)[:, subtype_label_word_id]
        e2_type_logits = self.cls(batch_t2_mask_reps)[:, subtype_label_word_id]
        if batch_mask_inputs is not None:
            mask_coref_logits = self.cls(batch_mask_mask_reps)[:, label_word_id]
            if not self.remove_match:
                if not self.remove_subtype_match:
                    mask_type_match_logits = self.cls(batch_mask_type_match_mask_reps)[:, match_label_word_id]
                if not self.remove_arg_match:
                    mask_arg_match_logits = self.cls(batch_mask_arg_match_mask_reps)[:, match_label_word_id]
            mask_e1_type_logits = self.cls(batch_mask_t1_mask_reps)[:, subtype_label_word_id]
            mask_e2_type_logits = self.cls(batch_mask_t2_mask_reps)[:, subtype_label_word_id]

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

class BertForSimpMixPrompt(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'product':
                self.coref_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
                self.type_match_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
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
                    self.type_match_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                    self.arg_match_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'product_cosine':
                    self.coref_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                    self.type_match_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                    self.arg_match_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
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
        batch_inputs, batch_mask_idx, batch_type_match_mask_idx, batch_arg_match_mask_idx, 
        batch_event_idx, 
        label_word_id, match_label_word_id, 
        batch_mask_inputs=None, labels=None, subtype_match_labels=None, arg_match_labels=None
        ):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_type_match_mask_reps = batched_index_select(sequence_output, 1, batch_type_match_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_arg_match_mask_reps = batched_index_select(sequence_output, 1, batch_arg_match_mask_idx.unsqueeze(-1)).squeeze(1)
        if batch_mask_inputs is not None:
            mask_outputs = self.bert(**batch_mask_inputs)
            mask_sequence_output = mask_outputs.last_hidden_state
            batch_mask_mask_reps = batched_index_select(mask_sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
            batch_mask_type_match_mask_reps = batched_index_select(mask_sequence_output, 1, batch_type_match_mask_idx.unsqueeze(-1)).squeeze(1)
            batch_mask_arg_match_mask_reps = batched_index_select(mask_sequence_output, 1, batch_arg_match_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_type_match_mask_reps = self.type_match_mapping(torch.cat([batch_type_match_mask_reps, batch_match_reps], dim=-1))
            batch_arg_match_mask_reps = self.arg_match_mapping(torch.cat([batch_arg_match_mask_reps, batch_match_reps], dim=-1))
            if batch_mask_inputs is not None:
                batch_mask_event_1_reps = self.span_extractor(mask_sequence_output, batch_e1_idx).squeeze(dim=1)
                batch_mask_event_2_reps = self.span_extractor(mask_sequence_output, batch_e2_idx).squeeze(dim=1)
                batch_mask_match_reps = self._matching_func(batch_mask_event_1_reps, batch_mask_event_2_reps)
                batch_mask_mask_reps = self.coref_mapping(torch.cat([batch_mask_mask_reps, batch_mask_match_reps], dim=-1))
                batch_mask_type_match_mask_reps = self.type_match_mapping(torch.cat([batch_mask_type_match_mask_reps, batch_mask_match_reps], dim=-1))
                batch_mask_arg_match_mask_reps = self.arg_match_mapping(torch.cat([batch_mask_arg_match_mask_reps, batch_mask_match_reps], dim=-1))
        coref_logits = self.cls(batch_mask_reps)[:, label_word_id]
        type_match_logits = self.cls(batch_type_match_mask_reps)[:, match_label_word_id]
        arg_match_logits = self.cls(batch_arg_match_mask_reps)[:, match_label_word_id]
        if batch_mask_inputs is not None:
            mask_coref_logits = self.cls(batch_mask_mask_reps)[:, label_word_id]
            mask_type_match_logits = self.cls(batch_mask_type_match_mask_reps)[:, match_label_word_id]
            mask_arg_match_logits = self.cls(batch_mask_arg_match_mask_reps)[:, match_label_word_id]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            coref_loss = loss_fct(coref_logits, labels)
            match_loss = 0.5 * loss_fct(type_match_logits, subtype_match_labels) + 0.5 * loss_fct(arg_match_logits, arg_match_labels)
            loss = torch.log(1 + coref_loss) + torch.log(1 + match_loss)
            if batch_mask_inputs is not None:
                mask_coref_loss = loss_fct(mask_coref_logits, labels)
                mask_match_loss = 0.5 * loss_fct(mask_type_match_logits, subtype_match_labels) + 0.5 * loss_fct(mask_arg_match_logits, arg_match_labels)
                mask_loss = torch.log(1 + mask_coref_loss) + torch.log(1 + mask_match_loss)
                loss = 0.5 * loss + 0.5 * mask_loss
        return loss, coref_logits

class RobertaForSimpMixPrompt(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'product':
                self.coref_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
                self.type_match_mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
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
                    self.type_match_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                    self.arg_match_mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'product_cosine':
                    self.coref_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                    self.type_match_mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
        batch_inputs, batch_mask_idx, batch_type_match_mask_idx, batch_arg_match_mask_idx, 
        batch_event_idx, 
        label_word_id, match_label_word_id, 
        batch_mask_inputs=None, labels=None, subtype_match_labels=None, arg_match_labels=None
        ):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_type_match_mask_reps = batched_index_select(sequence_output, 1, batch_type_match_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_arg_match_mask_reps = batched_index_select(sequence_output, 1, batch_arg_match_mask_idx.unsqueeze(-1)).squeeze(1)
        if batch_mask_inputs is not None:
            mask_outputs = self.roberta(**batch_mask_inputs)
            mask_sequence_output = mask_outputs.last_hidden_state
            batch_mask_mask_reps = batched_index_select(mask_sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
            batch_mask_type_match_mask_reps = batched_index_select(mask_sequence_output, 1, batch_type_match_mask_idx.unsqueeze(-1)).squeeze(1)
            batch_mask_arg_match_mask_reps = batched_index_select(mask_sequence_output, 1, batch_arg_match_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_type_match_mask_reps = self.type_match_mapping(torch.cat([batch_type_match_mask_reps, batch_match_reps], dim=-1))
            batch_arg_match_mask_reps = self.arg_match_mapping(torch.cat([batch_arg_match_mask_reps, batch_match_reps], dim=-1))
            if batch_mask_inputs is not None:
                batch_mask_event_1_reps = self.span_extractor(mask_sequence_output, batch_e1_idx).squeeze(dim=1)
                batch_mask_event_2_reps = self.span_extractor(mask_sequence_output, batch_e2_idx).squeeze(dim=1)
                batch_mask_match_reps = self._matching_func(batch_mask_event_1_reps, batch_mask_event_2_reps)
                batch_mask_mask_reps = self.coref_mapping(torch.cat([batch_mask_mask_reps, batch_mask_match_reps], dim=-1))
                batch_mask_type_match_mask_reps = self.type_match_mapping(torch.cat([batch_mask_type_match_mask_reps, batch_mask_match_reps], dim=-1))
                batch_mask_arg_match_mask_reps = self.arg_match_mapping(torch.cat([batch_mask_arg_match_mask_reps, batch_mask_match_reps], dim=-1))
        coref_logits = self.lm_head(batch_mask_reps)[:, label_word_id]
        type_match_logits = self.lm_head(batch_type_match_mask_reps)[:, match_label_word_id]
        arg_match_logits = self.lm_head(batch_arg_match_mask_reps)[:, match_label_word_id]
        if batch_mask_inputs is not None:
            mask_coref_logits = self.lm_head(batch_mask_mask_reps)[:, label_word_id]
            mask_type_match_logits = self.lm_head(batch_mask_type_match_mask_reps)[:, match_label_word_id]
            mask_arg_match_logits = self.lm_head(batch_mask_arg_match_mask_reps)[:, match_label_word_id]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            coref_loss = loss_fct(coref_logits, labels)
            match_loss = 0.5 * loss_fct(type_match_logits, subtype_match_labels) + 0.5 * loss_fct(arg_match_logits, arg_match_labels)
            loss = torch.log(1 + coref_loss) + torch.log(1 + match_loss)
            if batch_mask_inputs is not None:
                mask_coref_loss = loss_fct(mask_coref_logits, labels)
                mask_match_loss = 0.5 * loss_fct(mask_type_match_logits, subtype_match_labels) + 0.5 * loss_fct(mask_arg_match_logits, arg_match_labels)
                mask_loss = torch.log(1 + mask_coref_loss) + torch.log(1 + mask_match_loss)
                loss = 0.5 * loss + 0.5 * mask_loss
        return loss, coref_logits
