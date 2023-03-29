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
    

def train_mlps(d, G, num_steps_train=1000, num_steps_graft=1000, batch_size=1024, learning_rate=1e-3, device="cpu"):
    """
    # step 1. train a gelu mlp on sample -> target.
    # step 2. train a solu mlp on sample -> gelu_mlp(sample)
    """
    dataset = ReProjectorDataset(d=d, G=G, device=device)

    writer = SummaryWriter()

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
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=learning_rate)

    for batch_idx in range(num_steps_train):
        # faster than using a dataloader
        sample, target = dataset.get_batch(batch_size=batch_size)

        optimizer.zero_grad()
        output = mlp(sample)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f"batch {batch_idx}, loss {loss.item()}")
            writer.add_scalar("Loss/gelu_mlp_train", loss.item(), batch_idx)
        
        if batch_idx % 10000 == 0:
            # measure monosemanticity
            monosemanticity = measure_monosemanticity(gelu_mlp, proj, layernorm, device=device)
            writer.add_scalar("Monosemanticity/gelu_mlp_train", monosemanticity.mean().item(), batch_idx)
            writer.add_scalar("Monosemanticity/gelu_mlp_train_max", monosemanticity.max().item(), batch_idx)
            np_mono = monosemanticity.cpu().numpy()
            np_mono = np.asarray(sorted(np_mono))
            writer.add_scalar("Monosemanticity/gelu_mlp_train_mean_top", np_mono[-100:].mean(), batch_idx)

        if batch_idx >= num_steps_train:
            break

    # save the models
    torch.save(layernorm.state_dict(), "projection_out/layernorm.pt")
    torch.save(gelu_mlp.state_dict(), "projection_out/gelu_mlp.pt")

    solu_mlp = SoluMLP(input_size=d, hidden_size=d*8, output_size=d)
    solu_mlp.to(device)
    solu_mlp.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(solu_mlp.parameters(), lr=learning_rate)

    for batch_idx in range(num_steps_graft):
        sample, target = dataset.get_batch(batch_size=batch_size)

        optimizer.zero_grad()

        normed_sample = layernorm(sample)
        gelu_output = gelu_mlp(normed_sample)
        solu_output = solu_mlp(normed_sample)

        loss = criterion(solu_output, gelu_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f"batch {batch_idx}, loss {loss.item()}")
            writer.add_scalar("Loss/solu_mlp_train", loss.item(), batch_idx)

        if batch_idx % 10000 == 0:
            # measure monosemanticity
            monosemanticity = measure_monosemanticity(solu_mlp, proj, layernorm, device=device)
            writer.add_scalar("Monosemanticity/solu_mlp_train", monosemanticity.mean().item(), batch_idx)
            writer.add_scalar("Monosemanticity/solu_mlp_train_max", monosemanticity.max().item(), batch_idx)
            np_mono = monosemanticity.cpu().numpy()
            np_mono = np.asarray(sorted(np_mono))
            writer.add_scalar("Monosemanticity/solu_mlp_train_mean_top", np_mono[-100:].mean(), batch_idx)

        if batch_idx >= num_steps_graft:
            break

    # save the model
    torch.save(solu_mlp.state_dict(), "projection_out/solu_mlp.pt")

    writer.close()


if __name__ == "__main__":
    d = 64
    G = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_mlps(d, G, num_steps_train=500000, num_steps_graft=1000000, device=device)

    # load the models
    layernorm = nn.LayerNorm(d)
    layernorm.load_state_dict(torch.load("projection_out/layernorm.pt"))
    layernorm.to(device)
    layernorm.eval()

    gelu_mlp = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)
    gelu_mlp.load_state_dict(torch.load("projection_out/gelu_mlp.pt"))
    gelu_mlp.to(device)
    gelu_mlp.eval()

    solu_mlp = SoluMLP(input_size=d, hidden_size=d*8, output_size=d)
    solu_mlp.load_state_dict(torch.load("projection_out/solu_mlp.pt"))
    solu_mlp.to(device)
    solu_mlp.eval()


    # load the projection matrices
    proj = np.load("projection_out/proj.npy")
    target_proj = np.load("projection_out/target_proj.npy")

    # measure monosemanticity
    monosemanticity = measure_monosemanticity(solu_mlp, proj, layernorm, device=device)
    print('monosemanticity of solu mlp')
    print(f"monosemanticity: {monosemanticity}")
    print(f"mean monosemanticity: {torch.mean(monosemanticity)}")
    print(f"max monosemanticity: {torch.max(monosemanticity)}")

    monosemanticity = measure_monosemanticity(gelu_mlp, proj, layernorm, device=device)
    print('monosemanticity of gelu mlp')
    print(f"monosemanticity: {monosemanticity}")
    print(f"mean monosemanticity: {torch.mean(monosemanticity)}")
    print(f"max monosemanticity: {torch.max(monosemanticity)}")

