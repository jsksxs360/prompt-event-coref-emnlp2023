import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from torch.nn import CrossEntropyLoss
from ..tools import FullyConnectedLayer

class BertForPairwiseEC(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.coref_classifier = nn.Linear(3 * self.hidden_size, self.num_labels)
        self.post_init()
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class RobertaForPairwiseEC(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.coref_classifier = nn.Linear(3 * self.hidden_size, self.num_labels)
        self.post_init()
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class BertForPairwiseECwithMark(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.matching_style = args.matching_style
        if self.matching_style == 'cls':
            multiples = 1
        else:
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'event':
                multiples = 3
            elif self.matching_style == 'cls+event':
                multiples = 4
        self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        self.post_init()
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if self.matching_style == 'cls':
            batch_seq_reps = sequence_output[:, 0, :]
        else:
            batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
            batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
            if self.matching_style == 'event':
                batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
                batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
            elif self.matching_style == 'cls+event':
                batch_cls_reps = sequence_output[:, 0, :]
                batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
                batch_seq_reps = torch.cat([batch_cls_reps, batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class RobertaForPairwiseECwithMark(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.matching_style = args.matching_style
        if self.matching_style == 'cls':
            multiples = 1
        else:
            self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
            if self.matching_style == 'event':
                multiples = 3
            elif self.matching_style == 'cls+event':
                multiples = 4
        self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        self.post_init()
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if self.matching_style == 'cls':
            batch_seq_reps = sequence_output[:, 0, :]
        else:
            batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
            batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
            if self.matching_style == 'event':
                batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
                batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
            elif self.matching_style == 'cls+event':
                batch_cls_reps = sequence_output[:, 0, :]
                batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
                batch_seq_reps = torch.cat([batch_cls_reps, batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class BertForPairwiseECWithMask(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.trans = FullyConnectedLayer(config, input_dim=2*self.hidden_size, output_dim=self.hidden_size)
        self.matching_style = args.matching_style
        if self.matching_style == 'base':
            multiples = 2
        elif self.matching_style == 'multi':
            multiples = 3
        self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        self.post_init()
    
    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'base':
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=-1)
        elif self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        return batch_seq_reps

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None, subtypes=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.bert(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx) # [batch, 1, dim]
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx)
        
        batch_event_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=1)
        batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
        if labels is not None:
            kl_loss = self._cal_kl_loss(batch_event_reps, batch_event_mask_reps)
        batch_event_1_reps = torch.cat([batch_event_1_reps.squeeze(dim=1), batch_event_mask_1_reps.squeeze(dim=1)], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps.squeeze(dim=1), batch_event_mask_2_reps.squeeze(dim=1)], dim=-1)
        batch_event_1_reps = self.trans(batch_event_1_reps)
        batch_event_2_reps = self.trans(batch_event_2_reps)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + kl_loss)
        return loss, logits

class RobertaForPairwiseECWithMask(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.trans = FullyConnectedLayer(config, input_dim=2*self.hidden_size, output_dim=self.hidden_size)
        self.matching_style = args.matching_style
        if self.matching_style == 'base':
            multiples = 2
        elif self.matching_style == 'multi':
            multiples = 3
        self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        self.post_init()
    
    def _cal_kl_loss(self, origin_event_reps, mask_event_reps):
        origin_loss = F.kl_div(F.log_softmax(origin_event_reps, dim=-1), F.softmax(mask_event_reps, dim=-1), reduction='none')
        mask_loss = F.kl_div(F.log_softmax(mask_event_reps, dim=-1), F.softmax(origin_event_reps, dim=-1), reduction='none')
        origin_loss = origin_loss.sum()
        mask_loss = mask_loss.sum()
        return (origin_loss + mask_loss) / 2
    
    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'base':
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=-1)
        elif self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        return batch_seq_reps

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None, subtypes=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.roberta(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx) # [batch, 1, dim]
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx)
        
        batch_event_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=1)
        batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
        if labels is not None:
            kl_loss = self._cal_kl_loss(batch_event_reps, batch_event_mask_reps)
        batch_event_1_reps = torch.cat([batch_event_1_reps.squeeze(dim=1), batch_event_mask_1_reps.squeeze(dim=1)], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps.squeeze(dim=1), batch_event_mask_2_reps.squeeze(dim=1)], dim=-1)
        batch_event_1_reps = self.trans(batch_event_1_reps)
        batch_event_2_reps = self.trans(batch_event_2_reps)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + kl_loss)
        return loss, logits
