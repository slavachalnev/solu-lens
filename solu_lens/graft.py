import os
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
# from datasets import load_dataset
import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils

from mlp import SoluMLP, GeluMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
layer_to_hook = 9

checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def save_modified_gpt_model(model, new_layer, layer_to_replace, epoch, mlp_name):
    model_copy = copy.deepcopy(model)
    print(model_copy)
    model_copy.blocks[layer_to_replace].mlp = new_layer  # pretty sure this is wrong.
    print(model_copy)
    model_name = f"gpt2_modified_{mlp_name}_layer_{layer_to_replace}_epoch_{epoch}.pt"
    torch.save(model_copy.state_dict(), os.path.join(checkpoint_dir, model_name))

d_model = 768
solu_layer = SoluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)
big_solu_layer = SoluMLP(input_size=d_model, hidden_size=d_model*8, output_size=d_model)
gelu_layer = GeluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)

solu_layer.to(device)
big_solu_layer.to(device)
gelu_layer.to(device)

solu_layer.train()
big_solu_layer.train()
gelu_layer.train()

criterion = nn.MSELoss()
s_optimizer = torch.optim.AdamW(solu_layer.parameters(), lr=2e-4)
bs_optimizer = torch.optim.AdamW(big_solu_layer.parameters(), lr=2e-4)
g_optimizer = torch.optim.AdamW(gelu_layer.parameters(), lr=2e-4)

writer = SummaryWriter()

class PrePostActivationDataset(Dataset):
    def __init__(self, pre_activations_path, post_activations_path):
        self.pre_activations = np.load(pre_activations_path)
        self.post_activations = np.load(post_activations_path)

    def __len__(self):
        return len(self.pre_activations)

    def __getitem__(self, idx):
        pre_act = torch.tensor(self.pre_activations[idx], dtype=torch.float32)
        post_act = torch.tensor(self.post_activations[idx], dtype=torch.float32)
        return pre_act, post_act

pre_activations_path = os.path.join(checkpoint_dir, "pre_activations.npy")
post_activations_path = os.path.join(checkpoint_dir, "post_activations.npy")
val_pre_activations_path = os.path.join(checkpoint_dir, "val_pre_activations.npy")
val_post_activations_path = os.path.join(checkpoint_dir, "val_post_activations.npy")

dataset = PrePostActivationDataset(pre_activations_path, post_activations_path)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
val_dataset = PrePostActivationDataset(val_pre_activations_path, val_post_activations_path)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
for epoch in range(1000):
    s_loss_total = 0
    bs_loss_total = 0
    g_loss_total = 0

    for idx, (pre_batch, post_batch) in enumerate(data_loader):
        s_optimizer.zero_grad()
        bs_optimizer.zero_grad()
        g_optimizer.zero_grad()

        pre_batch = pre_batch.to(device)
        post_batch = post_batch.to(device)

        s_loss = criterion(solu_layer(pre_batch), post_batch)
        bs_loss = criterion(big_solu_layer(pre_batch), post_batch)
        g_loss = criterion(gelu_layer(pre_batch), post_batch)

        s_loss.backward()
        bs_loss.backward()
        g_loss.backward()

        s_optimizer.step()
        bs_optimizer.step()
        g_optimizer.step()

        s_loss_total += s_loss.item()
        bs_loss_total += bs_loss.item()
        g_loss_total += g_loss.item()

        if idx % 10 == 0:
            # log to tensorboard
            writer.add_scalar("Loss/S", s_loss.item(), epoch * len(data_loader) + idx)
            writer.add_scalar("Loss/G", g_loss.item(), epoch * len(data_loader) + idx)
            writer.add_scalar("Loss/BS", bs_loss.item(), epoch * len(data_loader) + idx)

        print(f"Epoch: {epoch} step {idx}, S Loss: {s_loss}, BS Loss: {bs_loss}, G Loss: {g_loss}")

    # val
    # eval mode
    solu_layer.eval()
    big_solu_layer.eval()
    gelu_layer.eval()

    s_loss_total = 0
    bs_loss_total = 0
    g_loss_total = 0

    for idx, (pre_batch, post_batch) in enumerate(val_dataloader):
        pre_batch = pre_batch.to(device)
        post_batch = post_batch.to(device)

        with torch.no_grad():
            s_loss = criterion(solu_layer(pre_batch), post_batch)
            bs_loss = criterion(big_solu_layer(pre_batch), post_batch)
            g_loss = criterion(gelu_layer(pre_batch), post_batch)

        s_loss_total += s_loss.item()
        bs_loss_total += bs_loss.item()
        g_loss_total += g_loss.item()
    
    writer.add_scalar("Val Loss/S", s_loss_total / len(val_dataloader), epoch)
    writer.add_scalar("Val Loss/G", g_loss_total / len(val_dataloader), epoch)
    writer.add_scalar("Val Loss/BS", bs_loss_total / len(val_dataloader), epoch)

    
    # train mode
    solu_layer.train()
    big_solu_layer.train()
    gelu_layer.train()

    print(f"end Epoch: {epoch}, S Loss: {s_loss_total / len(data_loader)}, BS Loss: {bs_loss_total / len(data_loader)}, G Loss: {g_loss_total / len(data_loader)}")


layer_str = f"layer_{layer_to_hook}"
torch.save(solu_layer.state_dict(), os.path.join(checkpoint_dir, f"solu_{layer_str}_{epoch}.pt"))
torch.save(big_solu_layer.state_dict(), os.path.join(checkpoint_dir, f"big_solu_{layer_str}_{epoch}.pt"))
torch.save(gelu_layer.state_dict(), os.path.join(checkpoint_dir, f"gelu_{layer_str}_{epoch}.pt"))

save_modified_gpt_model(gpt2, solu_layer, layer_to_hook, epoch, mlp_name="solu")
save_modified_gpt_model(gpt2, big_solu_layer, layer_to_hook, epoch, mlp_name="big_solu")
save_modified_gpt_model(gpt2, gelu_layer, layer_to_hook, epoch, mlp_name="gelu")

writer.close()


