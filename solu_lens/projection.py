import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from toy_data import ReProjectorDataset
from mlp import GeluMLP, SoluMLP


class OneHotDataset(Dataset):
    def __init__(self, proj):
        self.proj = proj

    def __len__(self):
        return self.proj.shape[0]

    def __getitem__(self, idx):
        sample = torch.zeros(self.proj.shape[0], device=self.proj.device)
        sample[idx] = 1

        return sample @ self.proj, idx


@torch.no_grad()
def measure_monosemanticity(model, projection_matrix, norm, device="cpu"):
    """
    model: a d -> h -> d mlp.
    projection_matrix: np array that projects G -> d.
    norm: nn.LayerNorm(d)
    """
    dataset = OneHotDataset(projection_matrix)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    model.to(device)
    model.eval()

    # for each neuron in the activation, compute:
    # - max activation over all samples
    # - average activation over all samples
    num_neurons = model.hidden_size
    max_activations = torch.zeros(num_neurons).to(device)
    sum_activations = torch.zeros(num_neurons).to(device)

    for batch_idx, (sample, n_idx) in enumerate(data_loader):
        sample = sample.to(device)
        normed = norm(sample)

        activations = model(normed, return_activations=True)

        batch_max = torch.amax(activations, dim=0)
        max_activations = torch.max(max_activations, batch_max)

        clipped_activations = torch.clamp(activations, min=0)
        sum_activations += torch.sum(clipped_activations, dim=0)
    
    monosemanticity = max_activations / (sum_activations + 1e-10)

    return monosemanticity


def train(model,
          dataset,
          writer,
          name,
          layernorm=None,
          target_model=None,
          num_steps=100000,
          batch_size=1024,
          learning_rate=1e-3,
          device="cpu",
          ):

    model.to(device)
    model.train()

    if target_model is not None:
        # if we have a target model, we need to normalize the input.
        # if we don't have a target model, assume layernorm is part of the model.
        assert layernorm is not None
        target_model.to(device)
        target_model.eval()
        layernorm.to(device)
        layernorm.eval()
    else:
        assert layernorm is None
        assert isinstance(model, nn.Sequential)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for batch_idx in range(num_steps):
        sample, target = dataset.get_batch(batch_size)
        sample = sample.to(device)

        if target_model is not None:
            with torch.no_grad():
                sample = layernorm(sample)
                target = target_model(sample)
        else:
            target = target.to(device)

        optimizer.zero_grad()
        output = model(sample)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f"batch_idx: {batch_idx}, loss: {loss.item()}")
            writer.add_scalar(f"Loss/{name}", loss.item(), batch_idx)
        
        if batch_idx % 10000 == 0:
            if target_model is None:
                monosemanticity = measure_monosemanticity(model[1], dataset.proj, model[0], device=device)
            else:
                monosemanticity = measure_monosemanticity(model, dataset.proj, layernorm, device=device)

            writer.add_scalar(f"Monosemanticity/{name}", monosemanticity.mean().item(), batch_idx)
            writer.add_scalar(f"Monosemanticity/{name}_max", monosemanticity.max().item(), batch_idx)
            np_mono = monosemanticity.cpu().numpy()
            np_mono = np.asarray(sorted(np_mono))
            writer.add_scalar(f"Monosemanticity/{name}_mean_top", np_mono[-100:].mean(), batch_idx)
        
        if batch_idx >= num_steps:
            break


def main():
    d = 64
    G = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ReProjectorDataset(d=d, G=G, device=device)

    # save projection matrices
    proj = dataset.proj.cpu().numpy()
    target_proj = dataset.target_proj.cpu().numpy()
    np.save("projection_out/proj.npy", proj)
    np.save("projection_out/target_proj.npy", target_proj)

    layernorm = nn.LayerNorm(d)
    gelu_mlp = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)
    sequential_mlp = nn.Sequential(layernorm, gelu_mlp)

    writer = SummaryWriter()

    train(model=sequential_mlp, dataset=dataset, writer=writer, name="original_gelu", device=device)

    # save the models
    torch.save(layernorm.state_dict(), "projection_out/layernorm.pt")
    torch.save(gelu_mlp.state_dict(), "projection_out/gelu_mlp.pt")

    solu_mlp = SoluMLP(input_size=d, hidden_size=d*8, output_size=d)

    train(model=solu_mlp, dataset=dataset, writer=writer, name="solu", layernorm=layernorm, target_model=gelu_mlp, device=device)

    # save the model
    torch.save(solu_mlp.state_dict(), "projection_out/solu_mlp.pt")

    writer.close()



if __name__ == "__main__":
    main()
