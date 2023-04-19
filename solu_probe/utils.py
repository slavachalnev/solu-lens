import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler

from datasets import load_dataset
from transformer_lens import utils as tutils


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
def measure_monosemanticity(model, projection_matrix, norm, plot=False, plot_dir=None, device="cpu"):
    """
    model: a d -> h -> d mlp.
    projection_matrix: np array that projects G -> d.
    norm: nn.LayerNorm(d)
    """
    t0 = time.time()
    projection_matrix = projection_matrix.to(torch.float32)
    dataset = OneHotDataset(projection_matrix)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    model.to(device)
    model.eval()

    num_neurons = model.hidden_size
    num_features = projection_matrix.shape[0]
    # activations_all = torch.zeros(num_neurons, num_features).to(device)
    activations_all = torch.zeros(num_features, num_neurons).to(device)

    for batch_idx, (sample, n_idx) in enumerate(data_loader):
        sample = sample.to(device)
        normed = norm(sample)

        activations = model(normed, return_activations=True)

        clipped_activations = torch.clamp(activations, min=0)
        activations_all[n_idx, :] = clipped_activations.squeeze()

    monosemanticity = torch.zeros(num_neurons).to(device)
    max_activations = torch.max(activations_all, dim=0).values
    sum_activations = torch.sum(activations_all, dim=0)
    monosemanticity = max_activations / (sum_activations + 1e-10)

    if plot:
        # os.makedirs(plot_dir, exist_ok=True)

        # Sort neurons by monosemanticity
        sorted_neurons = torch.argsort(monosemanticity, descending=True)

        # Sort features as per https://arxiv.org/pdf/2211.09169.pdf.
        activations_sorted_neurons = activations_all[:, sorted_neurons]
        most_activated_neurons = torch.argmax(activations_sorted_neurons, dim=1)
        sorted_features = torch.argsort(most_activated_neurons)

        activations_sorted = activations_all[sorted_features, :][:, sorted_neurons].cpu().numpy()

        # Rescale neuron activations
        max_activations_sorted = np.max(activations_sorted, axis=0)
        rescaled_activations_sorted = activations_sorted / (max_activations_sorted + 1e-10)
        rescaled_activations_sorted = rescaled_activations_sorted.T

        max_n = max(num_features, num_neurons)
        plt.figure(figsize=(7*num_features / max_n, 7*num_neurons / max_n))
        plt.imshow(rescaled_activations_sorted, aspect='auto', cmap='viridis')
        plt.xlabel('Features')
        plt.ylabel('Neurons')
        plt.title('Neuron Activations by Features')
        plt.colorbar(label='Activation')
        plt.show()

    return monosemanticity


def compute_difference_matrix(activations_mlp1, activations_mlp2):
    return torch.abs(activations_mlp1[:, None] - activations_mlp2)


@torch.jit.script # gives a 3x speedup
def process_acts(acts, out_acts, total_dists):
    for a1, a2 in zip(acts, out_acts):
        d = compute_difference_matrix(a1, a2)
        total_dists += d
    return total_dists


@torch.no_grad()
def mlp_dists(layers, dataset, device, num_batches=50):
    for layer in layers:
        layer.to(device)
        layer.eval()

    d_mlp_model = dataset.model.cfg.d_model * 4
    d_mlp_layers = layers[0].hidden_size  # assuming all layers have the same number of neurons
    d_mlp = min(d_mlp_model, d_mlp_layers)
    n_layers = len(layers)

    total_dists = [torch.zeros((d_mlp_layers, d_mlp_model), device="cpu") for _ in range(n_layers)]
    total_activation_strengths = [torch.zeros(d_mlp_layers, device="cpu") for _ in range(n_layers)]

    for b_idx, (in_acts, out_acts) in enumerate(dataset.generate_activations()):
        print('b_idx is ', b_idx)
        if b_idx >= num_batches:
            break

        for layer_idx, layer in enumerate(layers):
            acts = layer(in_acts[layer_idx], return_activations=True)

            acts = acts.cpu()
            out_acts[layer_idx] = out_acts[layer_idx].cpu()

            total_dists[layer_idx] = process_acts(acts, out_acts[layer_idx], total_dists[layer_idx])
            total_activation_strengths[layer_idx] += torch.sum(torch.abs(acts), dim=0)

    # Select top d_mlp neurons based on the sum of activation strengths
    top_indices = [torch.topk(strengths, d_mlp).indices for strengths in total_activation_strengths]
    selected_dists = [dists[top_idx] for dists, top_idx in zip(total_dists, top_indices)]

    # Average across all batches
    total_num_samples = num_batches * dataset.batch_size * d_mlp
    selected_dists = [dist.cpu().numpy() / total_num_samples for dist in selected_dists]

    # Compute optimal permutation of neurons for each layer
    perms = [linear_sum_assignment(dist) for dist in selected_dists]

    # Compute average distance between neurons for each layer after optimal permutation
    avg_dists = [np.mean(selected_dists[layer][perm]) for layer, perm in enumerate(perms)]

    return avg_dists


def big_data_loader(tokenizer, batch_size=8, big=True):

    if big:
        data = load_dataset("openwebtext", split="train[:10%]")
    else:
        data = load_dataset("NeelNanda/pile-10k", split="train")

    dataset = tutils.tokenize_and_concatenate(data, tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader
