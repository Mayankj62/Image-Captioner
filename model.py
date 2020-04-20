import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
                            
    def forward(self, features, captions):
        captions = captions[:, :captions.size()[1]]
        # For some reason I get an error that the line below needs to be a long type, when I change it to long, it says it needs a float type, then I change it to a float type and it stops complaining. Weird ...
        captions = captions.type(torch.float)

        embeded = self.word_embedding(captions)
        
        
        inp = torch.cat((embeded, features))
        
        x, _ = self.lstm(inp)
        x = self.fc(x)
            
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         sentence = []
        
#         while len(sentence) < (max_len - 1):
#             encoded = self
        pass