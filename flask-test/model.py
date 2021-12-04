import torch
from torch import nn
from transformers import BertModel


class Bert_Model(nn.Module):
    def __init__(self, device, mode='bert', dropout=False, dimension=768, max_length=128, pretrain_path=None):
        """
        This is the BERT base uncased model, it generates the embedding for a sentence
        :param dropout: Whether to add dropout (If don't add, <=0, if add, it will be used as the probability of dropout)
        :param dimension: The output dimension of BERT model, default 768
        :param mode: 'bert' or 'bert_tweet'
        :param pretrain_path: BERT pretrained path, None will download from huggingface
        """
        super(Bert_Model, self).__init__()
        self.mode = mode
        if pretrain_path:
            self.bert = BertModel.from_pretrained(pretrain_path)
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased").to(
                device) if mode == 'bert' else BertModel.from_pretrained("bert-large-uncased")
        self.embedding_dimension = dimension if mode == 'bert' else 1024
        self.dropout = dropout
        self.max_length = max_length
        assert dropout < 1, "Dropout parameter should be smaller than 1."
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()

    def forward(self, token):
        x = self.dropout_layer(self.bert(input_ids=token['input_ids'], attention_mask=token['attention_mask'],
                                         token_type_ids=token['token_type_ids'])).pooler_output
        return x


class NN(nn.Module):
    def __init__(self, encoder_model, input_dimension=768, output_dimensions=4):
        """
        The neural network classifier
        :param encoder_model: A BERT model, should take in a encoded dict as input and output a feature vector
        :param input_dimension: The dimension of the input feature vector
        :param output_dimensions: The output dimension (number of classes)
        """
        super(NN, self).__init__()
        self.encoder_model = encoder_model
        self.encoder_model_mode = self.encoder_model.mode
        self.nn1 = nn.Sequential(nn.Linear(input_dimension, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3))
        self.nn2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3))
        self.nn3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.nn4 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.linear = nn.Linear(64, output_dimensions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        result = []
        if self.encoder_model_mode == 'bert':
            for token in x:
                out = self.encoder_model(token)
                result.append(out)
            batch_encoding = torch.cat(result, dim=0)
        else:
            for token in x:
                out = self.encoder_model(token)
                result.append(out)
            batch_encoding = torch.cat(result, dim=0)
        out = self.nn1(batch_encoding)
        out = self.nn2(out)
        out = self.nn3(out)
        out = self.nn4(out)
        out = self.linear(out)
        out = self.softmax(out)
        return out

# # Code for counting parameters
# mode = 'bert_large'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test_input = "I am a machine learning student"
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if mode == 'bert' else BertTokenizer.from_pretrained(
#     'bert-large-uncased')
# encoder_model = Bert_Model(device=device, mode=mode)
# model = NN(encoder_model).to(device) if mode == 'bert' else NN(encoder_model, input_dimension=1024).to(device)
# print(sum(p.numel() for p in model.parameters()))
# print(sum(p.numel() for p in encoder_model.parameters()))
# print(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in encoder_model.parameters()))