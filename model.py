
import torch 
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_shape, hidden_layers, output_shape):
        super(MultiLayerPerceptron, self).__init__()

        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.output_shape = output_shape

        layers = []

        layers = [nn.Linear(input_shape, hidden_layers[0])]
        layers.append(nn.ReLU())

        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], output_shape))
        layers.append(nn.Softmax(dim=1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
