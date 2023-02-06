import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import LongformerPreTrainedModel, LongformerModel
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
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, labels=None):
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.cls(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
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
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, labels=None):
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
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
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, labels=None):
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class BertForPromptwithSubtype(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_t1_mask_idx, batch_t2_mask_idx, batch_mask_idx, batch_event_idx, subtype1=None, subtype2=None, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t1_mask_reps = batched_index_select(sequence_output, 1, batch_t1_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t2_mask_reps = batched_index_select(sequence_output, 1, batch_t2_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.cls(batch_mask_reps)
        
        loss = None
        if labels is not None:
            t1_logits = self.cls(batch_t1_mask_reps)
            t2_logits = self.cls(batch_t2_mask_reps)
            loss_fct = CrossEntropyLoss()
            subtype_loss = 0.5 * loss_fct(t1_logits, subtype1) + 0.5 * loss_fct(t2_logits, subtype2)
            mask_loss = loss_fct(logits, labels)
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss)
        return loss, logits

class RobertaForPromptwithSubtype(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_t1_mask_idx, batch_t2_mask_idx, batch_mask_idx, batch_event_idx, subtype1=None, subtype2=None, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t1_mask_reps = batched_index_select(sequence_output, 1, batch_t1_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t2_mask_reps = batched_index_select(sequence_output, 1, batch_t2_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            t1_logits = self.lm_head(batch_t1_mask_reps)
            t2_logits = self.lm_head(batch_t2_mask_reps)
            loss_fct = CrossEntropyLoss()
            subtype_loss = 0.5 * loss_fct(t1_logits, subtype1) + 0.5 * loss_fct(t2_logits, subtype2)
            mask_loss = loss_fct(logits, labels)
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss)
        return loss, logits

class LongformerForPromptwithSubtype(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.global_att = args.longformer_global_att
        self.hidden_size = config.hidden_size
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def forward(self, batch_inputs, batch_t1_mask_idx, batch_t2_mask_idx, batch_mask_idx, batch_event_idx, subtype1=None, subtype2=None, labels=None):
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
        batch_t1_mask_reps = batched_index_select(sequence_output, 1, batch_t1_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_t2_mask_reps = batched_index_select(sequence_output, 1, batch_t2_mask_idx.unsqueeze(-1)).squeeze(1)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            t1_logits = self.lm_head(batch_t1_mask_reps)
            t2_logits = self.lm_head(batch_t2_mask_reps)
            loss_fct = CrossEntropyLoss()
            subtype_loss = 0.5 * loss_fct(t1_logits, subtype1) + 0.5 * loss_fct(t2_logits, subtype2)
            mask_loss = loss_fct(logits, labels)
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss)
        return loss, logits

class BertForPromptwithEntity(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.hidden_size = config.hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.pooling =  nn.AdaptiveAvgPool1d(1)
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match
    
    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_p_idx, batch_e1_entity_idx, batch_e2_entity_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_parti_reps = batched_index_select(sequence_output, 1, batch_p_idx)
        # obtain entity representations
        e1_entity_max_len = max([len(entities) for entities in batch_e1_entity_idx])
        batch_entity1_mask = [[1] * len(entities) for entities in batch_e1_entity_idx]
        e2_entity_max_len = max([len(entities) for entities in batch_e2_entity_idx])
        batch_entity2_mask = [[1] * len(entities) for entities in batch_e2_entity_idx]
        for b_idx in range(len(batch_mask_idx)):
            pad_length = (e1_entity_max_len - len(batch_entity1_mask[b_idx])) if e1_entity_max_len > 0 else 1
            batch_e1_entity_idx[b_idx] += [[0, 0]] * pad_length
            batch_entity1_mask[b_idx] += [0] * pad_length
            pad_length = (e2_entity_max_len - len(batch_entity2_mask[b_idx])) if e2_entity_max_len > 0 else 1
            batch_e2_entity_idx[b_idx] += [[0, 0]] * pad_length
            batch_entity2_mask[b_idx] += [0] * pad_length
        batch_entity1 = torch.tensor(batch_e1_entity_idx).to(self.use_device)
        batch_mask_1 = torch.tensor(batch_entity1_mask).to(self.use_device)
        batch_entity2 = torch.tensor(batch_e2_entity_idx).to(self.use_device)
        batch_mask_2 = torch.tensor(batch_entity2_mask).to(self.use_device)
        batch_ent1_reps = self.span_extractor(sequence_output, batch_entity1, span_indices_mask=batch_mask_1)
        batch_ent2_reps = self.span_extractor(sequence_output, batch_entity2, span_indices_mask=batch_mask_2)
        batch_entity1_reps, batch_entity2_reps = [], []
        for ent1_reps, ent1_mask, ent2_reps, ent2_mask in zip(
            batch_ent1_reps, batch_entity1_mask, batch_ent2_reps, batch_entity2_mask
            ):
            ent1_reps = ent1_reps[:ent1_mask.count(1)]
            ent1_rep = self.pooling(ent1_reps.permute((1,0))).squeeze(dim=-1)
            batch_entity1_reps.append(ent1_rep.unsqueeze(dim=0))
            ent2_reps = ent2_reps[:ent2_mask.count(1)]
            ent2_rep = self.pooling(ent2_reps.permute((1,0))).squeeze(dim=-1)
            batch_entity2_reps.append(ent2_rep.unsqueeze(dim=0))
        batch_entity1_reps = torch.cat(batch_entity1_reps, dim=0)
        batch_entity2_reps = torch.cat(batch_entity2_reps, dim=0)
        batch_ent_reps = torch.cat([batch_entity1_reps.unsqueeze(dim=1), batch_entity2_reps.unsqueeze(dim=1)], dim=1)
        parti_loss = self._cal_kl_loss(batch_parti_reps, batch_ent_reps)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.cls(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            loss = torch.log(1 + mask_loss) + torch.log(1 + parti_loss)
        return loss, logits

class RobertaForPromptwithEntity(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.pooling =  nn.AdaptiveAvgPool1d(1)
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match
    
    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_p_idx, batch_e1_entity_idx, batch_e2_entity_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        batch_parti_reps = batched_index_select(sequence_output, 1, batch_p_idx)
        # obtain entity representations
        e1_entity_max_len = max([len(entities) for entities in batch_e1_entity_idx])
        batch_entity1_mask = [[1] * len(entities) for entities in batch_e1_entity_idx]
        e2_entity_max_len = max([len(entities) for entities in batch_e2_entity_idx])
        batch_entity2_mask = [[1] * len(entities) for entities in batch_e2_entity_idx]
        for b_idx in range(len(batch_mask_idx)):
            pad_length = (e1_entity_max_len - len(batch_entity1_mask[b_idx])) if e1_entity_max_len > 0 else 1
            batch_e1_entity_idx[b_idx] += [[0, 0]] * pad_length
            batch_entity1_mask[b_idx] += [0] * pad_length
            pad_length = (e2_entity_max_len - len(batch_entity2_mask[b_idx])) if e2_entity_max_len > 0 else 1
            batch_e2_entity_idx[b_idx] += [[0, 0]] * pad_length
            batch_entity2_mask[b_idx] += [0] * pad_length
        batch_entity1 = torch.tensor(batch_e1_entity_idx).to(self.use_device)
        batch_mask_1 = torch.tensor(batch_entity1_mask).to(self.use_device)
        batch_entity2 = torch.tensor(batch_e2_entity_idx).to(self.use_device)
        batch_mask_2 = torch.tensor(batch_entity2_mask).to(self.use_device)
        batch_ent1_reps = self.span_extractor(sequence_output, batch_entity1, span_indices_mask=batch_mask_1)
        batch_ent2_reps = self.span_extractor(sequence_output, batch_entity2, span_indices_mask=batch_mask_2)
        batch_entity1_reps, batch_entity2_reps = [], []
        for ent1_reps, ent1_mask, ent2_reps, ent2_mask in zip(
            batch_ent1_reps, batch_entity1_mask, batch_ent2_reps, batch_entity2_mask
            ):
            ent1_reps = ent1_reps[:ent1_mask.count(1)]
            ent1_rep = self.pooling(ent1_reps.permute((1,0))).squeeze(dim=-1)
            batch_entity1_reps.append(ent1_rep.unsqueeze(dim=0))
            ent2_reps = ent2_reps[:ent2_mask.count(1)]
            ent2_rep = self.pooling(ent2_reps.permute((1,0))).squeeze(dim=-1)
            batch_entity2_reps.append(ent2_rep.unsqueeze(dim=0))
        batch_entity1_reps = torch.cat(batch_entity1_reps, dim=0)
        batch_entity2_reps = torch.cat(batch_entity2_reps, dim=0)
        batch_ent_reps = torch.cat([batch_entity1_reps.unsqueeze(dim=1), batch_entity2_reps.unsqueeze(dim=1)], dim=1)
        parti_loss = self._cal_kl_loss(batch_parti_reps, batch_ent_reps)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            loss = torch.log(1 + mask_loss) + torch.log(1 + parti_loss)
        return loss, logits

class LongformerForPromptwithEntity(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.global_att = args.longformer_global_att
        self.hidden_size = config.hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.pooling =  nn.AdaptiveAvgPool1d(1)
        self.matching_style = args.matching_style
        self.use_device = args.device
        if self.matching_style != 'none':
            if self.matching_style == 'multi':
                self.mapping = nn.Linear(2 * self.hidden_size, self.hidden_size)
            else:
                self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
                self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
                self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
                self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
                if self.matching_style == 'cosine':
                    self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
                elif self.matching_style == 'multi_cosine':
                    self.mapping = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match
    
    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_p_idx, batch_e1_entity_idx, batch_e2_entity_idx, labels=None):
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
        batch_parti_reps = batched_index_select(sequence_output, 1, batch_p_idx)
        # obtain entity representations
        e1_entity_max_len = max([len(entities) for entities in batch_e1_entity_idx])
        batch_entity1_mask = [[1] * len(entities) for entities in batch_e1_entity_idx]
        e2_entity_max_len = max([len(entities) for entities in batch_e2_entity_idx])
        batch_entity2_mask = [[1] * len(entities) for entities in batch_e2_entity_idx]
        for b_idx in range(len(batch_mask_idx)):
            pad_length = (e1_entity_max_len - len(batch_entity1_mask[b_idx])) if e1_entity_max_len > 0 else 1
            batch_e1_entity_idx[b_idx] += [[0, 0]] * pad_length
            batch_entity1_mask[b_idx] += [0] * pad_length
            pad_length = (e2_entity_max_len - len(batch_entity2_mask[b_idx])) if e2_entity_max_len > 0 else 1
            batch_e2_entity_idx[b_idx] += [[0, 0]] * pad_length
            batch_entity2_mask[b_idx] += [0] * pad_length
        batch_entity1 = torch.tensor(batch_e1_entity_idx).to(self.use_device)
        batch_mask_1 = torch.tensor(batch_entity1_mask).to(self.use_device)
        batch_entity2 = torch.tensor(batch_e2_entity_idx).to(self.use_device)
        batch_mask_2 = torch.tensor(batch_entity2_mask).to(self.use_device)
        batch_ent1_reps = self.span_extractor(sequence_output, batch_entity1, span_indices_mask=batch_mask_1)
        batch_ent2_reps = self.span_extractor(sequence_output, batch_entity2, span_indices_mask=batch_mask_2)
        batch_entity1_reps, batch_entity2_reps = [], []
        for ent1_reps, ent1_mask, ent2_reps, ent2_mask in zip(
            batch_ent1_reps, batch_entity1_mask, batch_ent2_reps, batch_entity2_mask
            ):
            ent1_reps = ent1_reps[:ent1_mask.count(1)]
            ent1_rep = self.pooling(ent1_reps.permute((1,0))).squeeze(dim=-1)
            batch_entity1_reps.append(ent1_rep.unsqueeze(dim=0))
            ent2_reps = ent2_reps[:ent2_mask.count(1)]
            ent2_rep = self.pooling(ent2_reps.permute((1,0))).squeeze(dim=-1)
            batch_entity2_reps.append(ent2_rep.unsqueeze(dim=0))
        batch_entity1_reps = torch.cat(batch_entity1_reps, dim=0)
        batch_entity2_reps = torch.cat(batch_entity2_reps, dim=0)
        batch_ent_reps = torch.cat([batch_entity1_reps.unsqueeze(dim=1), batch_entity2_reps.unsqueeze(dim=1)], dim=1)
        parti_loss = self._cal_kl_loss(batch_parti_reps, batch_ent_reps)
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
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            loss = torch.log(1 + mask_loss) + torch.log(1 + parti_loss)
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
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.subtype_cls = nn.Linear(self.hidden_size, self.num_subtypes)
        if self.matching_style == 'multi':
            self.mapping = nn.Linear(3 * self.hidden_size, self.hidden_size)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
            elif self.matching_style == 'multi_cosine':
                self.mapping = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_mask_idx, batch_event_idx, subtypes=None, labels=None):
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
        batch_match_reps = self._matching_func(batch_new_event_1_reps, batch_new_event_2_reps)
        batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.cls(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            subtype_loss = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss) + torch.log(1 + kl_loss)
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
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.subtype_cls = nn.Linear(self.hidden_size, self.num_subtypes)
        if self.matching_style == 'multi':
            self.mapping = nn.Linear(3 * self.hidden_size, self.hidden_size)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
            elif self.matching_style == 'multi_cosine':
                self.mapping = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_mask_idx, batch_event_idx, subtypes=None, labels=None):
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
        batch_match_reps = self._matching_func(batch_new_event_1_reps, batch_new_event_2_reps)
        batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            subtype_loss = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss) + torch.log(1 + kl_loss)
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
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.subtype_cls = nn.Linear(self.hidden_size, self.num_subtypes)
        if self.matching_style == 'multi':
            self.mapping = nn.Linear(3 * self.hidden_size, self.hidden_size)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.mapping = nn.Linear(self.hidden_size + self.cosine_slices, self.hidden_size)
            elif self.matching_style == 'multi_cosine':
                self.mapping = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.hidden_size)
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
            batch_e1_e2_match = batch_event_1_reps * batch_event_2_reps
        elif self.matching_style == 'cosine':
            batch_e1_e2_match = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_e1_e2_match = torch.cat([batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_e1_e2_match

    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_mask_idx, batch_event_idx, subtypes=None, labels=None):
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
        batch_match_reps = self._matching_func(batch_new_event_1_reps, batch_new_event_2_reps)
        batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            mask_loss = loss_fct(logits, labels)
            subtype_loss = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + mask_loss) + torch.log(1 + subtype_loss) + torch.log(1 + kl_loss)
        return loss, logits
