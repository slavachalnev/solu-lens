import os
import gc
import copy
import numpy as np
import torch
from torch import nn
import tqdm
import h5py

import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils


layer_to_hook = 0
checkpoint_dir = "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
layernorm = copy.deepcopy(gpt2.blocks[layer_to_hook].ln2)
tokenizer = gpt2.tokenizer

loader = transformer_lens.evals.make_pile_data_loader(tokenizer=tokenizer, batch_size=8)


pre = None
post = None

def mlp_pre(value, hook):
    global pre
    pre = layernorm(value).detach().clone().cpu().to(torch.float16)
    return value

def mlp_post(value, hook):
    global post
    post = value.detach().clone().cpu().to(torch.float16)
    return value

fwd_hooks = [
    # (utils.get_act_name("attn_out", layer_to_hook), mlp_pre),
    # (utils.get_act_name("mlp_out", layer_to_hook), mlp_post)
    # (f"blocks.{layer_to_hook}.mlp.hook_pre", mlp_pre),
    # (f"blocks.{layer_to_hook}.mlp.hook_post", mlp_post)
    (f"blocks.{layer_to_hook}.hook_resid_mid", mlp_pre),
    (f"blocks.{layer_to_hook}.hook_mlp_out", mlp_post)
]

pre_activations = []
post_activations = []
val_pre = []
val_post = []

for idx, batch in tqdm.tqdm(enumerate(loader)):
    with torch.no_grad():
        loss = gpt2.run_with_hooks(
            batch["tokens"].to(device),
            return_type="loss",
            fwd_hooks=fwd_hooks,
        )
    if idx % 10 == 0:
        val_pre.append(pre.numpy())
        val_post.append(post.numpy())
    else:
        pre_activations.append(pre.numpy())
        post_activations.append(post.numpy())
    if idx > 800:
        break


pre_activations_np = np.concatenate(pre_activations, axis=0)
np.save(os.path.join(checkpoint_dir, "pre_activations.npy"), pre_activations_np)
del pre_activations
del pre_activations_np

post_activations_np = np.concatenate(post_activations, axis=0)
np.save(os.path.join(checkpoint_dir, "post_activations.npy"), post_activations_np)
del post_activations
del post_activations_np

val_pre_np = np.concatenate(val_pre, axis=0)
np.save(os.path.join(checkpoint_dir, "val_pre_activations.npy"), val_pre_np)
del val_pre
del val_pre_np

val_post_np = np.concatenate(val_post, axis=0)
np.save(os.path.join(checkpoint_dir, "val_post_activations.npy"), val_post_np)
del val_post
del val_post_np
