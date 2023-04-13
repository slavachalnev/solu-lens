import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler




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
    return np.abs(activations_mlp1[:, np.newaxis] - activations_mlp2)


@torch.no_grad()
def mlp_dists(model_1, model_2, dataloader, device):
    """
    Args:
        model_1: HookedTransformer GPT model
        model_2: HookedTransformer GPT model
        dataloader: dataloader for the dataset
    """
    
    model_1.to(device)
    model_2.to(device)
    model_1.eval()
    model_2.eval()

    d_mlp = model_1.cfg.d_model * 4
    n_layers = len(model_1.blocks)

    acts_1 = [None] * n_layers
    acts_2 = [None] * n_layers

    def create_callback(layer, is_model_1):
        """Save mlp activations for a given layer."""

        def callback(value, hook):
            """Callback for a given layer."""
            h = value.detach().clone().cpu()
            h = h.reshape(-1, d_mlp).numpy()

            if is_model_1:
                acts_1[layer] = h
            else:
                acts_2[layer] = h
            return value

        return callback
    
    m1_hooks = [(f"blocks.{layer}.mlp.hook_post", create_callback(layer, is_model_1=True)) for layer in range(n_layers)]
    m2_hooks = [(f"blocks.{layer}.mlp.hook_post", create_callback(layer, is_model_1=False)) for layer in range(n_layers)]

    total_dists = [np.zeros((d_mlp, d_mlp)) for _ in range(n_layers)]

    with model_1.hooks(fwd_hooks=m1_hooks), model_2.hooks(fwd_hooks=m2_hooks):
        for batch in dataloader:
            batch = batch.to(device)
            model_1(batch)
            model_2(batch)

            # Compute (d_mlp x d_mlp) l1 distance matrix for each layer
            dists = [compute_difference_matrix(acts_1[layer], acts_2[layer]) for layer in range(n_layers)]

            # Sum across all batches
            total_dists = [total_dists[layer] + dists[layer] for layer in range(n_layers)]

    # Average across all batches
    total_dists = [total_dists[layer] / len(dataloader) for layer in range(n_layers)]

    # compute optimal permutation of neurons for each layer
    perms = [linear_sum_assignment(dist) for dist in total_dists]

    # compute average distance between neurons for each layer after optimal permutation
    avg_dists = [np.mean(total_dists[layer][perm]) for layer, perm in enumerate(perms)]

    return avg_dists


