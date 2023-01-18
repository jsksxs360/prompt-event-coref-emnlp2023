import logging
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import LongformerPreTrainedModel, LongformerModel
from  ..tools import batched_index_select, BertOnlyMLMHead, RobertaLMHead, LongformerLMHead

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

class BertForPrompt(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()

    def forward(self, batch_inputs, batch_mask_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1))
        logits = self.cls(mask_reps.squeeze(1))

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
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.post_init()

    def forward(self, batch_inputs, batch_mask_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1))
        logits = self.lm_head(mask_reps.squeeze(1))

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
        self.post_init()

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
        
        # global attention on mask token
        global_attention_mask = torch.zeros_like(batch_inputs['input_ids'])
        global_attention_mask.scatter_(1, batch_mask_idx.unsqueeze(-1), 1)
        for b_idx, event_idxs in enumerate(batch_event_idx):
            for e_start, e_end in event_idxs:
                global_attention_mask[b_idx][e_start:e_end+1] = 1
        batch_inputs['global_attention_mask'] = global_attention_mask
        
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs.last_hidden_state
        mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1))
        logits = self.lm_head(mask_reps.squeeze(1))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits