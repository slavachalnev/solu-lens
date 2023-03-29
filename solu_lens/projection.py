import torch
from torch import nn

from toy_data import ReProjectorDataset
from mlp import GeluMLP, SoluMLP

# Plan:
# compute metrics along the way.


def train_mlps(d, G, num_steps_train=1000, num_steps_graft=1000, device="cpu"):
    """
    # step 1. train a gelu mlp on sample -> target.
    # step 2. train a solu mlp on sample -> gelu_mlp(sample)
    """
    dataset = ReProjectorDataset(d=d, G=G)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

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

    train_mlps(d, G, num_steps_train=1000, num_steps_graft=1000, device=device)

