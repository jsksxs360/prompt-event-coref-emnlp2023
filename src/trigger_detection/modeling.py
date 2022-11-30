from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from ..tools import FullyConnectedLayer, CRF

CNN_KERNEL_SIZE = 5

class LongformerSoftmaxForTD(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_addition_layer = args.use_addition_layer
        if self.use_addition_layer == 'ffnn':
            self.ffnn = FullyConnectedLayer(
                config, config.hidden_size, args.addition_layer_dim if args.addition_layer_dim else config.hidden_size
            )
        elif self.use_addition_layer == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size, args.addition_layer_dim if args.addition_layer_dim else config.hidden_size, 
                    kernel_size=CNN_KERNEL_SIZE, stride=1, padding='same'
                ),
                nn.Dropout(config.hidden_dropout_prob)
            )
        elif self.use_addition_layer == 'cnn_pool':
            self.cnn_pool = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size, args.addition_layer_dim if args.addition_layer_dim else config.hidden_size, 
                    kernel_size=CNN_KERNEL_SIZE, stride=1, padding='same'
                ), 
                nn.MaxPool1d(kernel_size=CNN_KERNEL_SIZE, stride=1, padding=((CNN_KERNEL_SIZE-1)//2))
            )
        self.classifier = nn.Linear(args.addition_layer_dim if args.addition_layer_dim else config.hidden_size, self.num_labels)
        self.post_init()
    
    def forward(self, batch_inputs, labels=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if self.use_addition_layer != 'none':
            if self.use_addition_layer == 'ffnn':
                sequence_output = self.ffnn(sequence_output)
            else: # cnn, cnn_pool
                sequence_output = sequence_output.permute((0, 2, 1))
                if self.use_addition_layer == 'cnn':
                    sequence_output = self.cnn(sequence_output)
                elif self.use_addition_layer == 'cnn_pool':
                    sequence_output = self.cnn_pool(sequence_output)
                sequence_output = sequence_output.permute((0, 2, 1))
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            attention_mask = batch_inputs.get('attention_mask')
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class LongformerCRFForTD(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_addition_layer = args.use_addition_layer
        if self.use_addition_layer == 'ffnn':
            self.ffnn = FullyConnectedLayer(
                config, config.hidden_size, args.addition_layer_dim if args.addition_layer_dim else config.hidden_size
            )
        elif self.use_addition_layer == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size, args.addition_layer_dim if args.addition_layer_dim else config.hidden_size, 
                    kernel_size=CNN_KERNEL_SIZE, stride=1, padding='same'
                ),
                nn.Dropout(config.hidden_dropout_prob)
            )
        elif self.use_addition_layer == 'cnn_pool':
            self.cnn_pool = nn.Sequential(
                nn.Conv1d(
                    config.hidden_size, args.addition_layer_dim if args.addition_layer_dim else config.hidden_size, 
                    kernel_size=CNN_KERNEL_SIZE, stride=1, padding='same'
                ), 
                nn.MaxPool1d(kernel_size=CNN_KERNEL_SIZE, stride=1, padding=((CNN_KERNEL_SIZE-1)//2))
            )
        self.classifier = nn.Linear(args.addition_layer_dim if args.addition_layer_dim else config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.post_init()
    
    def forward(self, batch_inputs, labels=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if self.use_addition_layer != 'none':
            if self.use_addition_layer == 'ffnn':
                sequence_output = self.ffnn(sequence_output)
            else: # cnn, cnn_pool
                sequence_output = sequence_output.permute((0, 2, 1))
                if self.use_addition_layer == 'cnn':
                    sequence_output = self.cnn(sequence_output)
                elif self.use_addition_layer == 'cnn_pool':
                    sequence_output = self.cnn_pool(sequence_output)
                sequence_output = sequence_output.permute((0, 2, 1))
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            attention_mask = batch_inputs.get('attention_mask')
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
            
        return loss, logits
