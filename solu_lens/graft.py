import os
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

#gpt2
from transformers import AutoTokenizer, AutoModel

# import transformer_lens
# from transformer_lens import HookedTransformer
# from transformer_lens import utils

from mlp import SoluMLP, GeluMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
gpt2 = AutoModel.from_pretrained("gpt2")

for name, module in gpt2.named_modules():
    print(f"{name}: {module.__class__.__name__}")

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

orig = copy.deepcopy(gpt2.h[layer_to_hook].mlp) ###

models = [solu_layer, big_solu_layer, gelu_layer, orig]
names = ["solu", "big_solu", "gelu", "orig"]

for model in models:
    model.to(device)
    model.train()

criterion = nn.MSELoss()
optimisers = [torch.optim.AdamW(m.parameters(), lr=2e-4) for m in models]

writer = SummaryWriter()

class PrePostActivationDataset(Dataset):
    def __init__(self, pre_activations_path, post_activations_path, add_random=False):
        self.pre_activations = np.load(pre_activations_path)
        self.post_activations = np.load(post_activations_path)
        self.add_random = add_random

    def __len__(self):
        if self.add_random:
            return len(self.pre_activations) * 2
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
    loss_totals = [0 for _ in models]

    for idx, (pre_batch, post_batch) in enumerate(data_loader):
        # flatten batch
        pre_batch = pre_batch.view(-1, pre_batch.shape[-1])
        post_batch = post_batch.view(-1, post_batch.shape[-1])
        pre_batch = pre_batch.to(device)
        post_batch = post_batch.to(device)

        for idx, (model, optimizer) in enumerate(zip(models, optimisers)):
            optimizer.zero_grad()
            res = model(pre_batch)
            loss = criterion(res, post_batch)
            loss.backward()
            optimizer.step()
            loss_totals[idx] += loss.item()

    for idx, name in enumerate(names):
        writer.add_scalar(f"Loss/{name}", loss_totals[idx] / len(data_loader), epoch)
    
    # evaluate
    for model in models:
        model.eval()

    val_loss_totals = [0 for _ in models]

    for idx, (pre_batch, post_batch) in enumerate(val_dataloader):
        pre_batch = pre_batch.view(-1, pre_batch.shape[-1])
        post_batch = post_batch.view(-1, post_batch.shape[-1])
        pre_batch = pre_batch.to(device)
        post_batch = post_batch.to(device)

        with torch.no_grad():
            for idx, model in enumerate(models):
                loss = criterion(model(pre_batch), post_batch)
                val_loss_totals[idx] += loss.item()

    for idx, name in enumerate(names):
        writer.add_scalar(f"val loss/{name}", val_loss_totals[idx] / len(val_dataloader), epoch)

    # train mode
    for model in models:
        model.train()


layer_str = f"layer_{layer_to_hook}"
torch.save(solu_layer.state_dict(), os.path.join(checkpoint_dir, f"solu_{layer_str}_{epoch}.pt"))
torch.save(big_solu_layer.state_dict(), os.path.join(checkpoint_dir, f"big_solu_{layer_str}_{epoch}.pt"))
torch.save(gelu_layer.state_dict(), os.path.join(checkpoint_dir, f"gelu_{layer_str}_{epoch}.pt"))

writer.close()


