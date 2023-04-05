import numpy as np
import time
import os
import random
import json
import argparse

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim.lr_scheduler as sched
from torch.utils.data import Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp

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

    # Add the cosine decay scheduler
    final_learning_rate = 0.1 * learning_rate
    scheduler = sched.CosineAnnealingLR(optimizer, num_steps, final_learning_rate)


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
            scheduler.step()
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
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar(f"Learning_rate/{name}", current_lr, batch_idx)
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
        

def do_analysis(checkpoint_dir):
    # build dataset
    proj = np.load(f"{checkpoint_dir}/proj.npy")
    target_proj = np.load(f"{checkpoint_dir}/target_proj.npy")
    G, d = proj.shape

    dataset = ReProjectorDataset(d=d, G=G, device="cpu", proj=proj, target_proj=target_proj)

    # load layernorm
    layernorm = nn.LayerNorm(d)
    layernorm.load_state_dict(torch.load(f"{checkpoint_dir}/layernorm.pt", map_location="cpu"))

    # load and analyse models
    model_names = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    model_names = [f for f in model_names if "layernorm" not in f]
    model_names = [os.path.splitext(f)[0] for f in model_names] # remove .pt

    for model_name in model_names:
        model_dict = torch.load(f"{checkpoint_dir}/{model_name}.pt", map_location="cpu")

        if model_name in ['gelu_mlp', 'graft_gelu']:
            model = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)
        elif model_name in ['graft_big_gelu']:
            model = GeluMLP(input_size=d, hidden_size=d*8, output_size=d)
        elif model_name in ['graft_solu']:
            model = SoluMLP(input_size=d, hidden_size=d*4, output_size=d)
        elif model_name in ['graft_big_solu']:
            model = SoluMLP(input_size=d, hidden_size=d*8, output_size=d)
        else:
            raise ValueError(f"unknown model name {model_name}")
        
        model.load_state_dict(model_dict)
        analyse_model(model, model_name, dataset, layernorm)


def analyse_model(model, name, dataset, layernorm):
    print()
    print('analyzing', name)
    monosemanticity = measure_monosemanticity(model, dataset.proj, layernorm, device="cpu", plot=True)
    print('monosemanticity', monosemanticity.mean().item())
    print('monosemanticity max', monosemanticity.max().item())
    np_mono = monosemanticity.cpu().numpy()
    np_mono = np.asarray(sorted(np_mono))
    print('monosemanticity mean top 100', np_mono[-100:].mean())
    print()


def log_hyperparameters(params, out_dir):
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def main(run_num, name, pre_trained_gelu_mlp=None, pre_trained_layernorm=None, dataset_path=None):
    out_dir = os.path.join("projection_out", name, str(run_num))
    os.makedirs(out_dir)

    d = 64
    G = 512
    graft_steps = 200000
    warmup_steps = 400000
    train_steps = 1000000
    batch_size = 65536
    learning_rate = 8e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyperparams = {
        "name": name,
        "run_num": run_num,
        "d": d,
        "G": G,
        "graft_steps": graft_steps,
        "warmup_steps": warmup_steps,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "original_pre_trained_gelu_mlp": pre_trained_gelu_mlp,
        "oririnal_pre_trained_layernorm": pre_trained_layernorm,
        "original_dataset_path": dataset_path,
    }
    log_hyperparameters(hyperparams, out_dir)

    # Load the existing dataset if dataset_path is provided, otherwise create a new one
    if dataset_path:
        proj = np.load(f"{dataset_path}/proj.npy")
        target_proj = np.load(f"{dataset_path}/target_proj.npy")
        dataset = ReProjectorDataset(d=d, G=G, device=device, proj=proj, target_proj=target_proj)
    else:
        dataset = ReProjectorDataset(d=d, G=G, device=device)

    # save projection matrices
    proj = dataset.proj.cpu().numpy()
    target_proj = dataset.target_proj.cpu().numpy()
    np.save(f"{out_dir}/proj.npy", proj)
    np.save(f"{out_dir}/target_proj.npy", target_proj)

    layernorm = nn.LayerNorm(d)
    gelu_mlp = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)

    # Load pre-trained models if paths are provided
    if pre_trained_layernorm:
        layernorm.load_state_dict(torch.load(pre_trained_layernorm, map_location=device))
    if pre_trained_gelu_mlp:
        gelu_mlp.load_state_dict(torch.load(pre_trained_gelu_mlp, map_location=device))

    sequential_mlp = nn.Sequential(layernorm, gelu_mlp)

    writer = SummaryWriter(log_dir=f"runs/{name}_{run_num}")

    if not pre_trained_gelu_mlp:
        train(model=sequential_mlp,
            dataset=dataset,
            writer=writer,
            name="original_gelu",
            device=device,
            num_steps=train_steps,
            warmup_steps=0,
            learning_rate=learning_rate,
            batch_size=batch_size,
            )

    # save the models
    torch.save(layernorm.state_dict(), f"{out_dir}/layernorm.pt")
    torch.save(gelu_mlp.state_dict(), f"{out_dir}/gelu_mlp.pt")

    ### graft ###

    graft_kwargs = {
        "dataset": dataset,
        "writer": writer,
        "layernorm": layernorm,
        "target_model": gelu_mlp,
        "device": device,
        "num_steps": graft_steps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    model = SoluMLP(input_size=d, hidden_size=d*4, output_size=d)
    train(model=model, name="graft_solu", **graft_kwargs)
    torch.save(model.state_dict(), f"{out_dir}/graft_solu.pt")

    model = SoluMLP(input_size=d, hidden_size=d*8, output_size=d)
    train(model=model, name="graft_big_solu", **graft_kwargs)
    torch.save(model.state_dict(), f"{out_dir}/graft_big_solu.pt")

    # model = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)
    # train(model=model, name="graft_gelu", **graft_kwargs)
    # torch.save(model.state_dict(), f"{out_dir}/graft_gelu.pt")

    # model = GeluMLP(input_size=d, hidden_size=d*8, output_size=d)
    # train(model=model, name="graft_big_gelu", **graft_kwargs)
    # torch.save(model.state_dict(), f"{out_dir}/graft_big_gelu.pt")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or analyze a model.")
    parser.add_argument("--mode", choices=["train", "analyse"], required=True, help="Choose between 'train' and 'analyse'.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs (default: 3).")
    parser.add_argument("--name", type=str, help="Name for the run. If not provided, a random name will be generated.")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory with saved checkpoints for analysis.")
    parser.add_argument("--pre_trained_gelu_mlp", type=str, help="Path to pre-trained GeluMLP model.")
    parser.add_argument("--pre_trained_layernorm", type=str, help="Path to pre-trained LayerNorm model.")
    parser.add_argument("--dataset_path", type=str, help="Path to existing dataset (proj.npy and target_proj.npy).")

    args = parser.parse_args()

    if args.mode == "train":
        run_name = args.name or str(random.randint(0, 1000000))
        for run_num in range(args.num_runs):
            main(run_num=run_num, name=run_name, pre_trained_gelu_mlp=args.pre_trained_gelu_mlp,
                 pre_trained_layernorm=args.pre_trained_layernorm, dataset_path=args.dataset_path)
        print("done")
    elif args.mode == "analyse":
        if args.checkpoint_dir is None:
            parser.error("--checkpoint_dir is required for mode 'analyse'.")
        do_analysis(args.checkpoint_dir)
