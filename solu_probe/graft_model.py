# this is the main script for training the graft mlps from a pre-trained transformer model.

import torch
import torch.nn as nn

import transformer_lens
from transformer_lens import HookedTransformer

from dataset import ModelDataset
from mlp import SoluMLP


def train(layers, dataset, steps, lr=1e-4, device="cpu"):

    optimizers = []
    for layer in layers:
        layer.to(device)
        layer.train()
        optimizers.append(torch.optim.Adam(layer.parameters(), lr=lr))

    criterion = nn.MSELoss()

    for batch_idx, (in_acts, out_acts) in enumerate(dataset.generate_activations()):
        for layer_idx, layer in enumerate(layers):
            optimizer = optimizers[layer_idx]
            optimizer.zero_grad()
            pred = layer(in_acts[layer_idx])
            loss = criterion(pred, out_acts[layer_idx])
            loss.backward()
            optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"batch {batch_idx}, loss {loss.item()}")
        
        if batch_idx >= steps:
            break


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_model = 512
    n_layers = 4
    model_type = "gelu-4l"

    # pre-trained model
    gpt_model = HookedTransformer.from_pretrained(model_type, device=device)

    dataset = ModelDataset(model=gpt_model, batch_size=8, device=device)

    # graft models
    graft_layers = []
    for _ in range(n_layers):
        graft_layers.append(SoluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model))
    
    # train
    train(layers=graft_layers, dataset=dataset, steps=1000, lr=1e-4, device=device)


if __name__ == '__main__':
    main()
