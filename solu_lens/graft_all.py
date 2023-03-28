import os
import math
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils

from mlp import SoluMLP, GeluMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)

data_loader = transformer_lens.evals.make_pile_data_loader(tokenizer=gpt2.tokenizer, batch_size=8)
val_loader = transformer_lens.evals.make_owt_data_loader(tokenizer=gpt2.tokenizer, batch_size=8)

def train_models_for_layer(layer_to_hook):
    layernorm = copy.deepcopy(gpt2.blocks[layer_to_hook].ln2)

    h_pre = None
    h_post = None

    def mlp_pre(value, hook):
        nonlocal h_pre
        h_pre = layernorm(value).detach().clone().cpu().to(torch.float16)
        return value

    def mlp_post(value, hook):
        nonlocal h_post
        h_post = value.detach().clone().cpu().to(torch.float16)
        return value

    fwd_hooks = [
        (f"blocks.{layer_to_hook}.hook_resid_mid", mlp_pre),
        (f"blocks.{layer_to_hook}.hook_mlp_out", mlp_post)
    ]

    for name, module in gpt2.named_modules():
        print(f"{name}: {module.__class__.__name__}")

    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    d_model = 768
    solu_layer = SoluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)
    big_solu_layer = SoluMLP(input_size=d_model, hidden_size=d_model*8, output_size=d_model)
    gelu_layer = GeluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)
    interp = SoluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)
    big_interp = SoluMLP(input_size=d_model, hidden_size=d_model*8, output_size=d_model)

    orig = copy.deepcopy(gpt2.blocks[layer_to_hook].mlp)
    orig.to(device)
    print("orig: ")
    print(orig)

    models = [interp, big_interp, gelu_layer, solu_layer, big_solu_layer]
    names = ["interp", "big_interp", "gelu", "solu", "big_solu"]

    for model in models:
        model.to(device)
        model.train()

    criterion = nn.MSELoss()
    optimisers = [torch.optim.AdamW(m.parameters(), lr=5e-4) for m in models]

    writer = SummaryWriter(log_dir=f'runs/layer_{layer_to_hook}')

    def alpha_schedule(step, total_steps=100, start_alpha=0.5, end_alpha=1.0):
        if step >= total_steps:
            return end_alpha
        cos_inner = (math.pi * step) / total_steps
        alpha = start_alpha + (end_alpha - start_alpha) * (1 - math.cos(cos_inner)) / 2
        return alpha

    def get_pre_post(batch):
        with torch.no_grad():
            loss = gpt2.run_with_hooks(
                batch["tokens"].to(device),
                return_type="loss",
                fwd_hooks=fwd_hooks,
                stop_at_layer=layer_to_hook+1,
            )
        pre_batch = h_pre.to(dtype=torch.float32).to(device)
        post_batch = h_post.to(dtype=torch.float32).to(device)
        return pre_batch, post_batch

    # prepare alphas
    alpha = alpha_schedule(0)
    for idx, (model, name) in enumerate(zip(models, names)):
        if "interp" in name:
            print(f"setting alpha for {name} to {alpha}")
            model.alpha = alpha

    global_step = 0
    for epoch in range(100):
        loss_totals = [0 for _ in models]

        for bidx, batch in enumerate(data_loader):
            global_step += 1

            if global_step % 100 == 0:
                alpha = alpha_schedule(global_step // 100)
                for idx, (model, name) in enumerate(zip(models, names)):
                    if "interp" in name:
                        print(f"setting alpha for {name} to {alpha}")
                        model.alpha = alpha

            pre_batch, post_batch = get_pre_post(batch)

            for idx, (model, optimizer) in enumerate(zip(models, optimisers)):
                optimizer.zero_grad()
                res = model(pre_batch)
                loss = criterion(res, post_batch)
                loss.backward()
                optimizer.step()
                loss_totals[idx] += loss.item()

                if bidx % 100 == 0:
                    print(f"Epoch {epoch}, batch {bidx}, {names[idx]} loss: {loss.item()}")
                    writer.add_scalar(f"Loss/{names[idx]}", loss.item(), epoch * len(data_loader) + bidx)

        # evaluate
        for model in models:
            model.eval()

        val_loss_totals = [0 for _ in models]

        for idx, batch in enumerate(val_loader):
            pre_batch, post_batch = get_pre_post(batch)

            with torch.no_grad():
                for idx, model in enumerate(models):
                    loss = criterion(model(pre_batch), post_batch)
                    val_loss_totals[idx] += loss.item()
            if idx == 100:
                break

        for i, name in enumerate(names):
            writer.add_scalar(f"val loss/{name}", val_loss_totals[i] / len(val_loader), epoch)

        # train mode
        for model in models:
            model.train()

    # save all the models
    layer_str = f"layer_{layer_to_hook}"
    for model, name in zip(models, names):
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{name}_{layer_str}_ep{epoch}.pt"))

    writer.close()


if __name__ == "__main__":
    train_models_for_layer(0)
