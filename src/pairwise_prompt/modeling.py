import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.gated_sum import GatedSum
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from  ..tools import batched_index_select, BertOnlyMLMHead, RobertaLMHead, LongformerLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

COSINE_SPACE_DIM = 64
COSINE_SLICES = 128
COSINE_FACTOR = 4

class BertForPrompt(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.gate = GatedSum(input_dim=self.hidden_size)
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            self.coref_cls = nn.Linear(self.hidden_size, 2)
            if self.matching_style == 'multi':
                self.match_transf = nn.Linear(3 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.match_transf = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.match_transf = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_coref=None, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
            # extract events & matching
            batch_e1_idx, batch_e2_idx = [], []
            for e1s, e1e, e2s, e2e in batch_event_idx:
                batch_e1_idx.append([[e1s, e1e]])
                batch_e2_idx.append([[e2s, e2e]])
            batch_e1_idx, batch_e2_idx = torch.tensor(batch_e1_idx).to(self.use_device), torch.tensor(batch_e2_idx).to(self.use_device)
            batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
            batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
            batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
            batch_match_reps = self.match_transf(batch_match_reps)
            coref_logits = self.coref_cls(batch_match_reps)
            batch_mask_reps = self.gate(batch_mask_reps, batch_match_reps) # gated sum
        logits = self.cls(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if self.matching_style != 'none':
                match_loss = loss_fct(coref_logits, batch_coref)
                loss = torch.log(1 + loss) + torch.log(1 + match_loss)
        return loss, logits

class RobertaForPrompt(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.gate = GatedSum(input_dim=self.hidden_size)
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            self.coref_cls = nn.Linear(self.hidden_size, 2)
            if self.matching_style == 'multi':
                self.match_transf = nn.Linear(3 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.match_transf = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.match_transf = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_coref=None, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
            # extract events & matching
            batch_e1_idx, batch_e2_idx = [], []
            for e1s, e1e, e2s, e2e in batch_event_idx:
                batch_e1_idx.append([[e1s, e1e]])
                batch_e2_idx.append([[e2s, e2e]])
            batch_e1_idx, batch_e2_idx = torch.tensor(batch_e1_idx).to(self.use_device), torch.tensor(batch_e2_idx).to(self.use_device)
            batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
            batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
            batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
            batch_match_reps = self.match_transf(batch_match_reps)
            coref_logits = self.coref_cls(batch_match_reps)
            batch_mask_reps = self.gate(batch_mask_reps, batch_match_reps) # gated sum
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if self.matching_style != 'none':
                match_loss = loss_fct(coref_logits, batch_coref)
                loss = torch.log(1 + loss) + torch.log(1 + match_loss)
        return loss, logits

class LongformerForPrompt(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.global_att = args.longformer_global_att
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.gate = GatedSum(input_dim=self.hidden_size)
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            self.coref_cls = nn.Linear(self.hidden_size, 2)
            if self.matching_style == 'multi':
                self.match_transf = nn.Linear(3 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.match_transf = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.match_transf = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()

    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_coref=None, labels=None):
        if self.global_att != 'no': # global attention on mask token
            global_attention_mask = torch.zeros_like(batch_inputs['input_ids'])
            if 'mask' in self.global_att:
                global_attention_mask.scatter_(1, batch_mask_idx.unsqueeze(-1), 1)
            if 'event' in self.global_att:
                for b_idx, (e1s, e1e, e2s, e2e) in enumerate(batch_event_idx):
                    global_attention_mask[b_idx][e1s:e1e+1] = 1
                    global_attention_mask[b_idx][e2s:e2e+1] = 1
            batch_inputs['global_attention_mask'] = global_attention_mask
        
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
            # extract events & matching
            batch_e1_idx, batch_e2_idx = [], []
            for e1s, e1e, e2s, e2e in batch_event_idx:
                batch_e1_idx.append([[e1s, e1e]])
                batch_e2_idx.append([[e2s, e2e]])
            batch_e1_idx, batch_e2_idx = torch.tensor(batch_e1_idx).to(self.use_device), torch.tensor(batch_e2_idx).to(self.use_device)
            batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
            batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
            batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
            batch_match_reps = self.match_transf(batch_match_reps)
            coref_logits = self.coref_cls(batch_match_reps)
            batch_mask_reps = self.gate(batch_mask_reps, batch_match_reps) # gated sum
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if self.matching_style != 'none':
                match_loss = loss_fct(coref_logits, batch_coref)
                loss = torch.log(1 + loss) + torch.log(1 + match_loss)
        return loss, logits

class BertForPromptwithMask(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.num_subtypes = args.num_subtypes
        self.matching_style = args.matching_style
        self.use_device = args.device
        self.gate = GatedSum(input_dim=self.hidden_size)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.coref_cls = nn.Linear(self.hidden_size, 2)
        self.subtype_cls = nn.Linear(self.hidden_size, self.num_subtypes)
        if self.matching_style == 'multi':
            self.match_transf = nn.Linear(6 * self.hidden_size, self.hidden_size)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.match_transf = nn.Linear(4 * self.hidden_size + self.cosine_slices, self.hidden_size)
            elif self.matching_style == 'multi_cosine':
                self.match_transf = nn.Linear(6 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_mask_idx, batch_event_idx, batch_coref=None, subtypes=None, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        outputs_with_mask = self.bert(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        # extract events
        batch_e1_idx, batch_e2_idx = [], []
        for e1s, e1e, e2s, e2e in batch_event_idx:
            batch_e1_idx.append([[e1s, e1e]])
            batch_e2_idx.append([[e2s, e2e]])
        batch_e1_idx, batch_e2_idx = torch.tensor(batch_e1_idx).to(self.use_device), torch.tensor(batch_e2_idx).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx)
        batch_event_mask_1_reps = self.span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.span_extractor(sequence_output_with_mask, batch_e2_idx)
        if labels is not None:
            batch_event_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=1)
            batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
            kl_loss = self._cal_kl_loss(batch_event_reps, batch_event_mask_reps)
            subtypes_logits = self.subtype_cls(batch_event_mask_reps)
        batch_new_event_1_reps = torch.cat([batch_event_1_reps.squeeze(dim=1), batch_event_mask_1_reps.squeeze(dim=1)], dim=-1)
        batch_new_event_2_reps = torch.cat([batch_event_2_reps.squeeze(dim=1), batch_event_mask_2_reps.squeeze(dim=1)], dim=-1)
        # matching
        batch_match_reps = self._matching_func(batch_new_event_1_reps, batch_new_event_2_reps)
        batch_match_reps = self.match_transf(batch_match_reps)
        if labels is not None:
            coref_logits = self.coref_cls(batch_match_reps)
        batch_new_mask_reps = self.gate(batch_mask_reps, batch_match_reps) # gated sum
        logits = self.cls(batch_new_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            subtype_loss = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            match_loss = loss_fct(coref_logits, batch_coref)
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss) + torch.log(1 + match_loss) + torch.log(1 + kl_loss)
        return loss, logits

class RobertaForPromptwithMask(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.num_subtypes = args.num_subtypes
        self.matching_style = args.matching_style
        self.use_device = args.device
        self.gate = GatedSum(input_dim=self.hidden_size)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.coref_cls = nn.Linear(self.hidden_size, 2)
        self.subtype_cls = nn.Linear(self.hidden_size, self.num_subtypes)
        if self.matching_style == 'multi':
            self.match_transf = nn.Linear(6 * self.hidden_size, self.hidden_size)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.match_transf = nn.Linear(4 * self.hidden_size + self.cosine_slices, self.hidden_size)
            elif self.matching_style == 'multi_cosine':
                self.match_transf = nn.Linear(6 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_mask_idx, batch_event_idx, batch_coref=None, subtypes=None, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        outputs_with_mask = self.roberta(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        # extract events
        batch_e1_idx, batch_e2_idx = [], []
        for e1s, e1e, e2s, e2e in batch_event_idx:
            batch_e1_idx.append([[e1s, e1e]])
            batch_e2_idx.append([[e2s, e2e]])
        batch_e1_idx, batch_e2_idx = torch.tensor(batch_e1_idx).to(self.use_device), torch.tensor(batch_e2_idx).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx)
        batch_event_mask_1_reps = self.span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.span_extractor(sequence_output_with_mask, batch_e2_idx)
        if labels is not None:
            batch_event_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=1)
            batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
            kl_loss = self._cal_kl_loss(batch_event_reps, batch_event_mask_reps)
            subtypes_logits = self.subtype_cls(batch_event_mask_reps)
        batch_new_event_1_reps = torch.cat([batch_event_1_reps.squeeze(dim=1), batch_event_mask_1_reps.squeeze(dim=1)], dim=-1)
        batch_new_event_2_reps = torch.cat([batch_event_2_reps.squeeze(dim=1), batch_event_mask_2_reps.squeeze(dim=1)], dim=-1)
        # matching
        batch_match_reps = self._matching_func(batch_new_event_1_reps, batch_new_event_2_reps)
        batch_match_reps = self.match_transf(batch_match_reps)
        if labels is not None:
            coref_logits = self.coref_cls(batch_match_reps)
        batch_new_mask_reps = self.gate(batch_mask_reps, batch_match_reps) # gated sum
        logits = self.lm_head(batch_new_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            subtype_loss = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            match_loss = loss_fct(coref_logits, batch_coref)
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss) + torch.log(1 + match_loss) + torch.log(1 + kl_loss)
        return loss, logits

class LongformerForPromptwithMask(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.global_att = args.longformer_global_att
        self.hidden_size = config.hidden_size
        self.num_subtypes = args.num_subtypes
        self.matching_style = args.matching_style
        self.use_device = args.device
        self.gate = GatedSum(input_dim=self.hidden_size)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.coref_cls = nn.Linear(self.hidden_size, 2)
        self.subtype_cls = nn.Linear(self.hidden_size, self.num_subtypes)
        if self.matching_style == 'multi':
            self.match_transf = nn.Linear(6 * self.hidden_size, self.hidden_size)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.match_transf = nn.Linear(4 * self.hidden_size + self.cosine_slices, self.hidden_size)
            elif self.matching_style == 'multi_cosine':
                self.match_transf = nn.Linear(6 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        # batch_event_1
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        # batch_event_2
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2
        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_mask_idx, batch_event_idx, batch_coref=None, subtypes=None, labels=None):
        if self.global_att != 'no': # global attention on mask token
            global_attention_mask = torch.zeros_like(batch_inputs['input_ids'])
            if 'mask' in self.global_att:
                global_attention_mask.scatter_(1, batch_mask_idx.unsqueeze(-1), 1)
            if 'event' in self.global_att:
                for b_idx, (e1s, e1e, e2s, e2e) in enumerate(batch_event_idx):
                    global_attention_mask[b_idx][e1s:e1e+1] = 1
                    global_attention_mask[b_idx][e2s:e2e+1] = 1
            batch_inputs['global_attention_mask'] = global_attention_mask
            batch_inputs_with_mask['global_attention_mask'] = global_attention_mask
        
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        outputs_with_mask = self.longformer(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        # extract events
        batch_e1_idx, batch_e2_idx = [], []
        for e1s, e1e, e2s, e2e in batch_event_idx:
            batch_e1_idx.append([[e1s, e1e]])
            batch_e2_idx.append([[e2s, e2e]])
        batch_e1_idx, batch_e2_idx = torch.tensor(batch_e1_idx).to(self.use_device), torch.tensor(batch_e2_idx).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx)
        batch_event_mask_1_reps = self.span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.span_extractor(sequence_output_with_mask, batch_e2_idx)
        if labels is not None:
            batch_event_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=1)
            batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
            kl_loss = self._cal_kl_loss(batch_event_reps, batch_event_mask_reps)
            subtypes_logits = self.subtype_cls(batch_event_mask_reps)
        batch_new_event_1_reps = torch.cat([batch_event_1_reps.squeeze(dim=1), batch_event_mask_1_reps.squeeze(dim=1)], dim=-1)
        batch_new_event_2_reps = torch.cat([batch_event_2_reps.squeeze(dim=1), batch_event_mask_2_reps.squeeze(dim=1)], dim=-1)
        # matching
        batch_match_reps = self._matching_func(batch_new_event_1_reps, batch_new_event_2_reps)
        batch_match_reps = self.match_transf(batch_match_reps)
        if labels is not None:
            coref_logits = self.coref_cls(batch_match_reps)
        batch_new_mask_reps = self.gate(batch_mask_reps, batch_match_reps) # gated sum
        logits = self.lm_head(batch_new_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            subtype_loss = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            match_loss = loss_fct(coref_logits, batch_coref)
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss) + torch.log(1 + match_loss) + torch.log(1 + kl_loss)
        return loss, logits
