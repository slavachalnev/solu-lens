import torch
from torch import nn


class SoLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x, temperature=1.0, alpha=1.0):
        solu = x * torch.softmax(x * (1/temperature), dim=-1)
        gelu = self.gelu(x)

        return (1 - alpha) * gelu + alpha * solu


class SoluMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, norm=True, temp=1.0, alpha=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        self.norm = norm
        self.activation = SoLU()
        self.temperature = temp
        self.alpha = alpha
    
    def forward(self, x, return_activations=False):
        """
        args:
            return_activations: if True, only return the activations of the first layer.
        """
        x = self.fc1(x)
        x = self.activation(x, temperature=self.temperature, alpha=self.alpha)

        if return_activations:
            return x

        if self.norm:
            x = self.layer_norm(x)
        x = self.fc2(x)
        return x

class GeluMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()
    
    def forward(self, x, return_activations=False):
        """
        args:
            return_activations: if True, only return the activations of the first layer.
        """
        x = self.fc1(x)
        x = self.activation(x)

        if return_activations:
            return x

        x = self.fc2(x)
        return x
    