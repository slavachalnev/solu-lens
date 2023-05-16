import torch
import torch.nn as nn

import numpy as np
import seaborn as sns

from solu_probe.mlp import SoluMLP, GeluMLP
from solu_probe.utils import interference

import matplotlib.pyplot as plt


# plan:
# 1. load the model and optionally the layernorm.
# 2. load the dataset and get the projection matrices.
# 3. compute the interference losses.


def main(data_dir, device='cpu'):
    d_model = 64
    d_hidden = d_model * 4
    # G = 512

    # load the model
    model = GeluMLP(d_model, d_hidden, d_model)
    layernorm = nn.LayerNorm(d_model, elementwise_affine=True) # elementwise for now. Need to re-train with elementwise_affine=False
    model.load_state_dict(torch.load(f"{data_dir}/gelu_mlp.pt", map_location=device))
    layernorm.load_state_dict(torch.load(f"{data_dir}/layernorm.pt", map_location=device))

    # load the dataset numpy arrays and convert from Half to np float.
    proj = np.load(f"{data_dir}/proj.npy").astype(np.float32)
    target_proj = np.load(f"{data_dir}/target_proj.npy").astype(np.float32)
    # convert to torch tensors
    proj = torch.tensor(proj)
    target_proj = torch.tensor(target_proj)
    print('proj shape ', proj.shape)
    print('target_proj shape ', target_proj.shape)

    # compute the interference losses
    # reproj_losses, decompression_losses = interference(model=model, p1=proj, p2=target_proj, norm=layernorm)
    # print(f"Reprojection losses: {reproj_losses}")
    # print(f"Decompression losses: {decompression_losses}")

    # fig, ax = plt.subplots(1, 3)

    # pic = interference(model=model, p1=proj, p2=target_proj, norm=layernorm)
    # plt.matshow(pic)
    # # plt.show()

    # pic = interference(model=model, p1=proj, p2=target_proj, norm=layernorm, additional_feat_idx=0)
    # plt.matshow(pic)
    # # plt.show()
    
    # pic = interference(model=model, p1=proj, p2=target_proj, norm=layernorm, additional_feat_idx=300)
    # plt.matshow(pic)
    # plt.show()


    # # Example nxn arrays
    array1 = interference(model=model, p1=proj, p2=target_proj, norm=layernorm)
    array2 = interference(model=model, p1=proj, p2=target_proj, norm=layernorm, additional_feat_idx=0)
    array3 = interference(model=model, p1=proj, p2=target_proj, norm=layernorm, additional_feat_idx=200)
    array4 = interference(model=model, p1=proj, p2=target_proj, norm=layernorm, additional_feat_idx=400)
    print('finished computing. Now plotting.')

    # Find the minimum and maximum values across all four arrays
    vmin = np.min([array1.min(), array2.min(), array3.min(), array4.min()])
    vmax = np.max([array1.max(), array2.max(), array3.max(), array4.max()])

    # Plotting the heatmaps
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    sns.heatmap(array1, annot=False, fmt='.2f', cmap='viridis', ax=axs[0, 0], vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Array 1")

    sns.heatmap(array2, annot=False, fmt='.2f', cmap='viridis', ax=axs[0, 1], vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("Array 2")

    sns.heatmap(array3, annot=False, fmt='.2f', cmap='viridis', ax=axs[1, 0], vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Array 3")

    sns.heatmap(array4, annot=False, fmt='.2f', cmap='viridis', ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("Array 4")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(data_dir="../projection_out/964786/0", device=device)
