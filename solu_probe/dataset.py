from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from utils import big_data_loader

import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils


class ModelDataset(Dataset):
    """Dataset for getting mlp activations from a model.

    TODO: make layers selectable. Currently all layers are used.
    """

    def __init__(self, model, batch_size, n_random=0, device="cpu", big=True, mid=False):
        self.model = model  # gpt Transformer Lens model
        self.device = device
        self.batch_size = batch_size
        self.n_random = n_random
        self.d_model = model.cfg.d_model
        self.n_ctx = model.cfg.n_ctx

        self.data_loader = big_data_loader(tokenizer=model.tokenizer, batch_size=batch_size, big=big)

        num_layers = len(model.blocks)
        self.pre_hs = [None] * num_layers
        self.post_hs = [None] * num_layers

        # deepcopy layernorms
        model.to("cpu")
        self.layernorms = [deepcopy(model.blocks[layer].ln2).to(device) for layer in range(num_layers)]
        model.to(device)

        self.fwd_hooks = []
        for layer in range(num_layers):
            # pre hook
            self.fwd_hooks.append((f"blocks.{layer}.hook_resid_mid", self.create_callback(layer, post=False)))
            # post hook
            if mid:
                self.fwd_hooks.append((f"blocks.{layer}.mlp.hook_post", self.create_callback(layer, post=True, d_post=self.d_model*4)))
            else:
                self.fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", self.create_callback(layer, post=True)))

    def create_callback(self, layer, post=False, d_post=None):
        """Save in and out activations for mlp at a given layer."""
        if d_post is None:
            d_post = self.d_model

        def callback(value, hook):
            """Callback for a given layer."""
            # h shape is batch_size, n_ctx, d_model
            # we want batch_size * n_ctx, d_model

            if post:
                h = value.detach().clone()
                h = h.reshape(-1, d_post)
                self.post_hs[layer] = h

            else:
                h = self.layernorms[layer](value)
                h = h.detach().clone()
                h = h.reshape(-1, self.d_model)
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
            pre_hs = [torch.randn(self.batch_size, self.n_ctx, self.d_model).to(self.device) for _ in range(len(self.model.blocks))]
            post_hs = []
            
            for layer, pre_h in enumerate(pre_hs):
                post_h = self.model.blocks[layer].mlp(pre_h)
                post_h = post_h.reshape(-1, self.d_model)
                post_hs.append(post_h)
            
            pre_hs = [pre_h.reshape(-1, self.d_model) for pre_h in pre_hs]
            yield pre_hs, post_hs

        # normal forward pass through model
        while True:
            for batch in self.data_loader:
                self.get_pre_post(batch)
                yield self.pre_hs, self.post_hs
        
