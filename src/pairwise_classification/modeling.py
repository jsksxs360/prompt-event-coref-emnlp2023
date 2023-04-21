import logging
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from torch.nn import CrossEntropyLoss
from ..tools import FullyConnectedLayer

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

class BertForPairwiseEC(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.matching_style = args.matching_style
        if self.matching_style == 'product':
            input_dim = 3 * self.hidden_size
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (
                args.cosine_space_dim, args.cosine_slices, args.cosine_factor
            )
            self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            if self.matching_style == 'cosine':
                input_dim = 2 * self.hidden_size + self.cosine_slices
            elif self.matching_style == 'product_cosine':
                input_dim = 3 * self.hidden_size + self.cosine_slices
        self.coref_classifier = nn.Linear(input_dim, 2)
        self.post_init()
    
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
        return torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_match], dim=-1)

    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_match_reps)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class RobertaForPairwiseEC(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.matching_style = args.matching_style
        if self.matching_style == 'product':
            input_dim = 3 * self.hidden_size
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (
                args.cosine_space_dim, args.cosine_slices, args.cosine_factor
            )
            self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            if self.matching_style == 'cosine':
                input_dim = 2 * self.hidden_size + self.cosine_slices
            elif self.matching_style == 'product_cosine':
                input_dim = 3 * self.hidden_size + self.cosine_slices
        self.coref_classifier = nn.Linear(input_dim, 2)
        self.post_init()

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
        return torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_match], dim=-1)
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_match_reps)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class LongformerForPairwiseEC(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.matching_style = args.matching_style
        if self.matching_style == 'product':
            input_dim = 3 * self.hidden_size
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = (
                args.cosine_space_dim, args.cosine_slices, args.cosine_factor
            )
            self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            if self.matching_style == 'cosine':
                input_dim = 2 * self.hidden_size + self.cosine_slices
            elif self.matching_style == 'product_cosine':
                input_dim = 3 * self.hidden_size + self.cosine_slices
        self.coref_classifier = nn.Linear(input_dim, 2)
        self.post_init()
    
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
        return torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_match], dim=-1)

    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_match_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_match_reps)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits
