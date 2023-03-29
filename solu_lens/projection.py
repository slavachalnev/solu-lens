import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset

from toy_data import ReProjectorDataset
from mlp import GeluMLP, SoluMLP


class OneHotDataset(Dataset):
    def __init__(self, proj):
        self.proj = proj

    def __len__(self):
        return self.proj.shape[0]

    def __getitem__(self, idx):
        sample = torch.zeros(self.proj.shape[0])
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
    max_activations = torch.zeros(num_neurons)
    sum_activations = torch.zeros(num_neurons)

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
    

def train_mlps(d, G, num_steps_train=1000, num_steps_graft=1000, device="cpu"):
    """
    # step 1. train a gelu mlp on sample -> target.
    # step 2. train a solu mlp on sample -> gelu_mlp(sample)
    """
    dataset = ReProjectorDataset(d=d, G=G)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    # save projection matrices
    proj = dataset.proj.cpu().numpy()
    target_proj = dataset.target_proj.cpu().numpy()
    np.save("projection_out/proj.npy", proj)
    np.save("projection_out/target_proj.npy", target_proj)

    layernorm = nn.LayerNorm(d)
    gelu_mlp = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)

    mlp = nn.Sequential(layernorm, gelu_mlp)
    mlp.to(device)
    mlp.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    for batch_idx, (sample, target) in enumerate(data_loader):
        sample = sample.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = mlp(sample)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"batch {batch_idx}, loss {loss.item()}")

        if batch_idx >= num_steps_train:
            break

    # save the models
    torch.save(layernorm.state_dict(), "projection_out/layernorm.pt")
    torch.save(gelu_mlp.state_dict(), "projection_out/gelu_mlp.pt")

    solu_mlp = SoluMLP(input_size=d, hidden_size=d*4, output_size=d)
    solu_mlp.to(device)
    solu_mlp.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(solu_mlp.parameters(), lr=1e-3)

    for batch_idx, (sample, target) in enumerate(data_loader):
        sample = sample.to(device)

        optimizer.zero_grad()

        normed_sample = layernorm(sample)
        gelu_output = gelu_mlp(normed_sample)
        solu_output = solu_mlp(normed_sample)

        loss = criterion(solu_output, gelu_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"batch {batch_idx}, loss {loss.item()}")

        if batch_idx >= num_steps_graft:
            break

    # save the model
    torch.save(solu_mlp.state_dict(), "projection_out/solu_mlp.pt")


if __name__ == "__main__":
    d = 64
    G = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_mlps(d, G, num_steps_train=50000, num_steps_graft=10000, device=device)

    # load the models
    layernorm = nn.LayerNorm(d)
    layernorm.load_state_dict(torch.load("projection_out/layernorm.pt"))
    layernorm.to(device)
    layernorm.eval()

    gelu_mlp = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)
    gelu_mlp.load_state_dict(torch.load("projection_out/gelu_mlp.pt"))
    gelu_mlp.to(device)
    gelu_mlp.eval()

    solu_mlp = SoluMLP(input_size=d, hidden_size=d*4, output_size=d)
    solu_mlp.load_state_dict(torch.load("projection_out/solu_mlp.pt"))
    solu_mlp.to(device)
    solu_mlp.eval()


    # load the projection matrices
    proj = np.load("projection_out/proj.npy")
    target_proj = np.load("projection_out/target_proj.npy")

    # measure monosemanticity
    monosemanticity = measure_monosemanticity(solu_mlp, proj, layernorm, device=device)
    print(f"monosemanticity: {monosemanticity}")
    print(f"mean monosemanticity: {torch.mean(monosemanticity)}")
    print(f"max monosemanticity: {torch.max(monosemanticity)}")


