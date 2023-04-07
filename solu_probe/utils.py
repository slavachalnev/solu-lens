import time

import numpy as np
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


def train(model,
          dataset,
          writer,
          name,
          layernorm=None,
          target_model=None,
          num_steps=500000,
          warmup_steps=0,
          batch_size=65536,
          learning_rate=5e-3,
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
    scaler = amp.GradScaler()

    t0 = time.time()

    total_steps = num_steps + warmup_steps
    for batch_idx in range(total_steps):
        # t0 = time.time()
        if batch_idx < warmup_steps:
            # random sample
            d = dataset.d
            sample, target = torch.rand(batch_size, d, device=device), torch.rand(batch_size, d, device=device)
        else:
            sample, target = dataset.get_batch(batch_size)
        # t1 = time.time()
        # print('getting batch took \t', t1 - t0, 'seconds.')

        sample = sample.to(device)

        if target_model is not None:
            with torch.no_grad():
                with amp.autocast():
                    sample = layernorm(sample)
                    target = target_model(sample)
        else:
            target = target.to(device)

        optimizer.zero_grad()
        with amp.autocast():
            output = model(sample)
            loss = criterion(output, target)
        # t2 = time.time()
        # print('forward pass took \t', t2 - t1, 'seconds.')
        scaler.scale(loss).backward()
        # t3 = time.time()
        # print('backward took \t\t', t3 - t2, 'seconds.')
        scaler.step(optimizer)
        scaler.update()
        # t4 = time.time()
        # print('update took \t\t', t4 - t3, 'seconds.')
        
        if batch_idx % 1000 == 0:
            print(f"batch_idx: {batch_idx}, loss: {loss.item()}")
            writer.add_scalar(f"Loss/{name}", loss.item(), batch_idx)
            # Log learning rate to TensorBoard
            print('training took \t\t', time.time() - t0, 'seconds.')
            t0 = time.time()
        
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
            writer.add_scalar(f"Monosemanticity/{name}_num_mono", np.count_nonzero(np_mono > 0.9), batch_idx)
        
