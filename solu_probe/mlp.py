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


class GSoLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim=-1, temp=1.0):
        x = x / temp
        x_max = x.max(dim, keepdim=True).values
        e_x = torch.exp(x - x_max)
        return (x * e_x) / torch.sum(e_x, dim, keepdim=True)


class SoluMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, norm=True, skip=False, temp=1.0, alpha=1.0, pre_gelu=False):
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
        self.skip = skip

        if pre_gelu:
            self.pre_gelu = nn.GELU()
        else:
            self.pre_gelu = None
    
    def forward(self, x, return_activations=False):
        """
        args:
            return_activations: if True, only return the activations of the first layer.
        """
        h = self.fc1(x)

        if self.pre_gelu is not None:
            h = self.pre_gelu(h)

        h = self.activation(h, temperature=self.temperature, alpha=self.alpha)

        if return_activations:
            return h

        if self.norm:
            h = self.layer_norm(h)
        h = self.fc2(h)

        if self.skip:
            h = h + x

        return h
    

class GSoLUMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, temp=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.activation = GSoLU()
        self.temperature = temp
    
    def forward(self, x, return_activations=False):
        """
        args:
            return_activations: if True, only return the activations of the first layer.
        """
        h = self.fc1(x)
        h = self.activation(h, temp=self.temperature)
        if return_activations:
            return h
        h = self.fc2(h)
        return h


class GeluMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, skip=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()
        self.skip = skip
    
    def forward(self, x, return_activations=False):
        """
        args:
            return_activations: if True, only return the activations of the first layer.
        """
        h = self.fc1(x)
        h = self.activation(h)

        if return_activations:
            return h

        h = self.fc2(h)

        if self.skip:
            h = h + x

        return h
    