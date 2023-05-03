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
from mlp import GeluMLP, SoluMLP, GatedSoLU
from utils import measure_monosemanticity


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

        # analyse model
        print('analyzing', model_name)
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
    warmup_steps = 200000
    train_steps = 2000000
    batch_size = 65536
    learning_rate = 5e-3

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

    # experiment: replace original gelu with solu (and also try solu_plus_gelu)
    # gelu_mlp = SoluMLP(input_size=d, hidden_size=d*4, output_size=d, norm=False, pre_gelu=True)  # solu_plus_gelu
    # gelu_mlp = SoluMLP(input_size=d, hidden_size=d*4, output_size=d) # solu

    # gelu_mlp = GatedSoLU(input_size=d, hidden_size=(d*8)//3, output_size=d, norm=True, softmax=True) # gated softmax
    # gelu_mlp = GatedSoLU(input_size=d, hidden_size=(d*8)//3, output_size=d, norm=True, softmax=False) # gated solu
    # gelu_mlp = GatedSoLU(input_size=d, hidden_size=d*4, output_size=d, norm=True, softmax=False) # gated solu

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

    model = GatedSoLU(input_size=d, hidden_size=d*4, output_size=d, norm=True, softmax=False)
    train(model=model, name="graft_gate_solu", **graft_kwargs)
    torch.save(model.state_dict(), f"{out_dir}/graft_solu.pt")

    model = SoluMLP(input_size=d, hidden_size=d*4, output_size=d)
    train(model=model, name="graft_solu", **graft_kwargs)
    torch.save(model.state_dict(), f"{out_dir}/graft_solu.pt")

    model = SoluMLP(input_size=d, hidden_size=d*8, output_size=d)
    train(model=model, name="graft_big_solu", **graft_kwargs)
    torch.save(model.state_dict(), f"{out_dir}/graft_big_solu.pt")

    model = GeluMLP(input_size=d, hidden_size=d*4, output_size=d)
    train(model=model, name="graft_gelu", **graft_kwargs)
    torch.save(model.state_dict(), f"{out_dir}/graft_gelu.pt")

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
