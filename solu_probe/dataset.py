from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils


def big_owt_data_loader(tokenizer, batch_size=8):
    data = load_dataset("openwebtext", split="train[:10%]")
    dataset = tutils.tokenize_and_concatenate(data, tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


class ModelDataset(Dataset):
    """Dataset for getting mlp activations from a model.

    TODO: make layers selectable. Currently all layers are used.
    """

    def __init__(self, model, batch_size, n_random=0, device="cpu"):
        self.model = model  # gpt Transformer Lens model
        self.device = device
        self.batch_size = batch_size
        self.n_random = n_random
        self.d_model = model.cfg.d_model

        self.data_loader = big_owt_data_loader(tokenizer=model.tokenizer, batch_size=batch_size)

        num_layers = len(model.blocks)
        self.pre_hs = [None] * num_layers
        self.post_hs = [None] * num_layers

        # deepcopy layernorms
        model.to("cpu")
        self.layernorms = [deepcopy(model.blocks[layer].ln2).to(device) for layer in range(num_layers)]
        model.to(device)

        self.fwd_hooks = []
        for layer in range(num_layers):
            self.fwd_hooks += [
                (f"blocks.{layer}.hook_resid_mid", self.create_callback(layer, post=False)),
                (f"blocks.{layer}.hook_mlp_out", self.create_callback(layer, post=True)),
            ]

    def create_callback(self, layer, post=False):
        """Save in and out activations for mlp at a given layer."""

        def callback(value, hook):
            """Callback for a given layer."""
            if post:
                h = value.detach().clone().cpu().to(torch.float16)
                self.post_hs[layer] = h

            else:
                h = self.layernorms[layer](value)
                h = h.detach().clone().cpu().to(torch.float16)
                self.pre_hs[layer] = h
            return value

        return callback
    
    def __len__(self):
        return len(self.data_loader)
    
    def __getitem__(self, idx):
        raise NotImplementedError()
    
    @torch.no_grad()
    def get_pre_post(self, batch):
        self.model.run_with_hooks(
            batch["tokens"].to(self.device),
            fwd_hooks=self.fwd_hooks,
        )
    
    @torch.no_grad()
    def generate_activations(self):

        # random input to MLPs
        for i in range(self.n_random):
            pre_hs = [torch.randn(self.batch_size, self.d_model).to(self.device) for _ in range(len(self.model.blocks))]
            post_hs = []
            
            for layer, pre_h in enumerate(pre_hs):
                post_h = self.model.blocks[layer].mlp(pre_h)
                post_hs.append(post_h)
            
            yield pre_hs, post_hs

        # normal forward pass through model
        for batch in self.data_loader:
            self.get_pre_post(batch)
            yield self.pre_hs, self.post_hs
        
