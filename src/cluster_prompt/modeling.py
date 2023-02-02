import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
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
            self.pooling =  nn.AdaptiveAvgPool1d(1)
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
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
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

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_cluster1_idx, batch_cluster2_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
            cluster1_max_len = max([len(cluster) for cluster in batch_cluster1_idx])
            batch_cluster1_mask = [[1] * len(cluster) for cluster in batch_cluster1_idx]
            cluster2_max_len = max([len(cluster) for cluster in batch_cluster2_idx])
            batch_cluster2_mask = [[1] * len(cluster) for cluster in batch_cluster2_idx]
            # padding
            for b_idx in range(len(batch_mask_idx)):
                pad_length = (cluster1_max_len - len(batch_cluster1_mask[b_idx])) if cluster1_max_len > 0 else 1
                batch_cluster1_idx[b_idx] += [[0, 0]] * pad_length
                batch_cluster1_mask[b_idx] += [0] * pad_length
                pad_length = (cluster2_max_len - len(batch_cluster2_mask[b_idx])) if cluster2_max_len > 0 else 1
                batch_cluster2_idx[b_idx] += [[0, 0]] * pad_length
                batch_cluster2_mask[b_idx] += [0] * pad_length
            # extract events
            batch_cluster1 = torch.tensor(batch_cluster1_idx).to(self.use_device)
            batch_mask_1 = torch.tensor(batch_cluster1_mask).to(self.use_device)
            batch_cluster2 = torch.tensor(batch_cluster2_idx).to(self.use_device)
            batch_mask_2 = torch.tensor(batch_cluster2_mask).to(self.use_device)
            batch_cluster1_reps = self.span_extractor(sequence_output, batch_cluster1, span_indices_mask=batch_mask_1)
            batch_cluster2_reps = self.span_extractor(sequence_output, batch_cluster2, span_indices_mask=batch_mask_2)
            # pooling
            batch_event1_reps, batch_event2_reps = [], []
            for cluster1_reps, cluster1_mask, cluster2_reps, cluster2_mask in zip(
                batch_cluster1_reps, batch_cluster1_mask, batch_cluster2_reps, batch_cluster2_mask
                ):
                cluster1_reps = cluster1_reps[:cluster1_mask.count(1)]
                event1_rep = self.pooling(cluster1_reps.permute((1,0))).squeeze(dim=-1)
                batch_event1_reps.append(event1_rep.unsqueeze(dim=0))
                cluster2_reps = cluster2_reps[:cluster2_mask.count(1)]
                event2_rep = self.pooling(cluster2_reps.permute((1,0))).squeeze(dim=-1)
                batch_event2_reps.append(event2_rep.unsqueeze(dim=0))
            batch_event1_reps = torch.cat(batch_event1_reps, dim=0)
            batch_event2_reps = torch.cat(batch_event2_reps, dim=0)
            # matching
            batch_match_reps = self._matching_func(batch_event1_reps, batch_event2_reps)
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
            self.pooling =  nn.AdaptiveAvgPool1d(1)
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
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
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

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_cluster1_idx, batch_cluster2_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
            cluster1_max_len = max([len(cluster) for cluster in batch_cluster1_idx])
            batch_cluster1_mask = [[1] * len(cluster) for cluster in batch_cluster1_idx]
            cluster2_max_len = max([len(cluster) for cluster in batch_cluster2_idx])
            batch_cluster2_mask = [[1] * len(cluster) for cluster in batch_cluster2_idx]
            # padding
            for b_idx in range(len(batch_mask_idx)):
                pad_length = (cluster1_max_len - len(batch_cluster1_mask[b_idx])) if cluster1_max_len > 0 else 1
                batch_cluster1_idx[b_idx] += [[0, 0]] * pad_length
                batch_cluster1_mask[b_idx] += [0] * pad_length
                pad_length = (cluster2_max_len - len(batch_cluster2_mask[b_idx])) if cluster2_max_len > 0 else 1
                batch_cluster2_idx[b_idx] += [[0, 0]] * pad_length
                batch_cluster2_mask[b_idx] += [0] * pad_length
            # extract events
            batch_cluster1 = torch.tensor(batch_cluster1_idx).to(self.use_device)
            batch_mask_1 = torch.tensor(batch_cluster1_mask).to(self.use_device)
            batch_cluster2 = torch.tensor(batch_cluster2_idx).to(self.use_device)
            batch_mask_2 = torch.tensor(batch_cluster2_mask).to(self.use_device)
            batch_cluster1_reps = self.span_extractor(sequence_output, batch_cluster1, span_indices_mask=batch_mask_1)
            batch_cluster2_reps = self.span_extractor(sequence_output, batch_cluster2, span_indices_mask=batch_mask_2)
            # pooling
            batch_event1_reps, batch_event2_reps = [], []
            for cluster1_reps, cluster1_mask, cluster2_reps, cluster2_mask in zip(
                batch_cluster1_reps, batch_cluster1_mask, batch_cluster2_reps, batch_cluster2_mask
                ):
                cluster1_reps = cluster1_reps[:cluster1_mask.count(1)]
                event1_rep = self.pooling(cluster1_reps.permute((1,0))).squeeze(dim=-1)
                batch_event1_reps.append(event1_rep.unsqueeze(dim=0))
                cluster2_reps = cluster2_reps[:cluster2_mask.count(1)]
                event2_rep = self.pooling(cluster2_reps.permute((1,0))).squeeze(dim=-1)
                batch_event2_reps.append(event2_rep.unsqueeze(dim=0))
            batch_event1_reps = torch.cat(batch_event1_reps, dim=0)
            batch_event2_reps = torch.cat(batch_event2_reps, dim=0)
            # matching
            batch_match_reps = self._matching_func(batch_event1_reps, batch_event2_reps)
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
            self.pooling =  nn.AdaptiveAvgPool1d(1)
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
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
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

    def forward(self, batch_inputs, batch_mask_idx, batch_event_idx, batch_cluster1_idx, batch_cluster2_idx, labels=None):
        if self.global_att != 'no': # global attention on mask token
            global_attention_mask = torch.zeros_like(batch_inputs['input_ids'])
            if 'mask' in self.global_att:
                global_attention_mask.scatter_(1, batch_mask_idx.unsqueeze(-1), 1)
            if 'event' in self.global_att:
                for b_idx, event_idxs in enumerate(batch_event_idx):
                    for e_start, e_end in event_idxs:
                        global_attention_mask[b_idx][e_start:e_end+1] = 1
            batch_inputs['global_attention_mask'] = global_attention_mask

        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        if self.matching_style != 'none':
            cluster1_max_len = max([len(cluster) for cluster in batch_cluster1_idx])
            batch_cluster1_mask = [[1] * len(cluster) for cluster in batch_cluster1_idx]
            cluster2_max_len = max([len(cluster) for cluster in batch_cluster2_idx])
            batch_cluster2_mask = [[1] * len(cluster) for cluster in batch_cluster2_idx]
            # padding
            for b_idx in range(len(batch_mask_idx)):
                pad_length = (cluster1_max_len - len(batch_cluster1_mask[b_idx])) if cluster1_max_len > 0 else 1
                batch_cluster1_idx[b_idx] += [[0, 0]] * pad_length
                batch_cluster1_mask[b_idx] += [0] * pad_length
                pad_length = (cluster2_max_len - len(batch_cluster2_mask[b_idx])) if cluster2_max_len > 0 else 1
                batch_cluster2_idx[b_idx] += [[0, 0]] * pad_length
                batch_cluster2_mask[b_idx] += [0] * pad_length
            # extract events
            batch_cluster1 = torch.tensor(batch_cluster1_idx).to(self.use_device)
            batch_mask_1 = torch.tensor(batch_cluster1_mask).to(self.use_device)
            batch_cluster2 = torch.tensor(batch_cluster2_idx).to(self.use_device)
            batch_mask_2 = torch.tensor(batch_cluster2_mask).to(self.use_device)
            batch_cluster1_reps = self.span_extractor(sequence_output, batch_cluster1, span_indices_mask=batch_mask_1)
            batch_cluster2_reps = self.span_extractor(sequence_output, batch_cluster2, span_indices_mask=batch_mask_2)
            # pooling
            batch_event1_reps, batch_event2_reps = [], []
            for cluster1_reps, cluster1_mask, cluster2_reps, cluster2_mask in zip(
                batch_cluster1_reps, batch_cluster1_mask, batch_cluster2_reps, batch_cluster2_mask
                ):
                cluster1_reps = cluster1_reps[:cluster1_mask.count(1)]
                event1_rep = self.pooling(cluster1_reps.permute((1,0))).squeeze(dim=-1)
                batch_event1_reps.append(event1_rep.unsqueeze(dim=0))
                cluster2_reps = cluster2_reps[:cluster2_mask.count(1)]
                event2_rep = self.pooling(cluster2_reps.permute((1,0))).squeeze(dim=-1)
                batch_event2_reps.append(event2_rep.unsqueeze(dim=0))
            batch_event1_reps = torch.cat(batch_event1_reps, dim=0)
            batch_event2_reps = torch.cat(batch_event2_reps, dim=0)
            # matching
            batch_match_reps = self._matching_func(batch_event1_reps, batch_event2_reps)
            batch_mask_reps = self.mapping(torch.cat([batch_mask_reps, batch_match_reps], dim=-1))
        logits = self.lm_head(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits
