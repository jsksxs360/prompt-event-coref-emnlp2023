import torch
from torch import nn
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor

class LongformerSelector(LongformerPreTrainedModel):

    def __init__(self, config, args):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.use_device = args.device
        # encoder & pooler
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.post_init()
    
    def _cal_circle_loss(self, event_1_reps, event_2_reps, coref_labels, l=20.):
        norms_1 = (event_1_reps ** 2).sum(axis=1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps ** 2).sum(axis=1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2
        event_cos = torch.sum(event_1_reps * event_2_reps, dim=1) * l
        # calculate the difference between each pair of Cosine values
        event_cos_diff = event_cos[:, None] - event_cos[None, :]
        # find (noncoref, coref) index
        select_idx = coref_labels[:, None] < coref_labels[None, :]
        select_idx = select_idx.float()

        event_cos_diff = event_cos_diff - (1 - select_idx) * 1e12
        event_cos_diff = event_cos_diff.view(-1)
        event_cos_diff = torch.cat((torch.tensor([0.0], device=self.use_device), event_cos_diff), dim=0)
        return torch.logsumexp(event_cos_diff, dim=0)
    
    def forward(self, batch_inputs, batch_events, batch_event_cluster_ids=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list, batch_event_mask = [], [], []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            for events, event_cluster_ids in zip(batch_events, batch_event_cluster_ids):
                event_1_list, event_2_list, coref_labels = [], [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for events in batch_events:
                event_1_list, event_2_list = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                max_len = max(max_len, len(event_1_list))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        # extract events
        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        # calculate loss 
        loss, batch_labels = None, None
        if batch_event_cluster_ids is not None and max_len > 0:
            # Only keep active parts of the loss
            active_loss = batch_mask.view(-1) == 1
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_labels = batch_labels.view(-1)[active_loss]
            active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
            active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
            loss = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
        return loss, batch_event_1_reps, batch_event_2_reps, batch_mask, batch_labels
