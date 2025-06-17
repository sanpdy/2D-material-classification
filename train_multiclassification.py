'''
Model architecture:
                                                                                BRANCH 1 -> FC -> Classify 1-layer, 2-layer, 3-layer, 4-layer
Image ->                                CNN (Resnet18) -> Vector  1 x D_i ----^
                                                         ----⌄ BRANCH 2
Material (Chosen, not predicted) -> Embedding  -> Vector: 1 x D_m   CONCATENATE with 1x D_i -> 1 x (D_i + D_m) -> FC -> Classify 1-layer, 2-layer, 3-layer, 4-layer

'''

import torch
from transformers import AutoImageProcessor
import torch.nn as nn

resnet = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

class Classification(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(Classification, self).__init__()
        self.cnn = resnet
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=embedding_dim) # 
        self.fc1 = nn.Linear(self.cnn.config.hidden_size + embedding_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, images, materials):
        cnn_output = self.cnn(images).logits  # Get logits from CNN
        embedding_output = self.embedding(materials)  # Get embeddings for materials
        combined = torch.cat((cnn_output, embedding_output), dim=1)  # Concatenate outputs
        x = torch.relu(self.fc1(combined))  # First fully connected layer with ReLU activation
        x = self.fc2(x)  # Second fully connected layer for classification
        return x