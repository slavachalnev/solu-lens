# this is the main script for training the graft mlps from a pre-trained transformer model.
import os
import time
import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import transformer_lens
from transformer_lens import HookedTransformer

from dataset import ModelDataset
from mlp import SoluMLP
from utils import big_data_loader, mlp_dists


def train(layers, dataset, steps, writer, checkpoints_dir, lr=1e-4, device="cpu"):
    val_model = HookedTransformer.from_pretrained("solu-4l", device=device)
    val_ds = ModelDataset(val_model, batch_size=1, n_random=0, device=device, big=False, mid=True)

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
                print(f"batch {batch_idx}, layer {layer_idx}, loss {loss.item()}")
                writer.add_scalar('Layer {}/Loss'.format(layer_idx), loss.item(), batch_idx)
        
        if batch_idx % 1000 == 0:
            dists = mlp_dists(layers=layers, dataset=val_ds, device=device)
            print(f'batch {batch_idx}, dists is ', dists)
            for i, dist in enumerate(dists):
                writer.add_scalar(f"Layer {i}/act_dist", dist, batch_idx)
        
        if batch_idx % 10000 == 0:
            for layer_idx, layer in enumerate(layers):
                torch.save(layer.state_dict(), os.path.join(checkpoints_dir, f"layer_{layer_idx}_step_{batch_idx}.pt"))
        
        if batch_idx >= steps:
            print("Done training at step", batch_idx)
            break


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = SummaryWriter()
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    d_model = 512
    n_layers = 4
    model_type = "gelu-4l"

    # pre-trained model
    gpt_model = HookedTransformer.from_pretrained(model_type, device=device)

    dataset = ModelDataset(model=gpt_model, batch_size=32, n_random=50000, device=device)

    # graft models
    graft_layers = []
    for _ in range(n_layers):
        graft_layers.append(SoluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model))
    
    train(layers=graft_layers,
          dataset=dataset,
          writer=writer,
          checkpoints_dir=checkpoints_dir,
          steps=200001,
          lr=3e-3,
          device=device,
          )


if __name__ == '__main__':
    main()
