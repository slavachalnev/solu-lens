import os
import copy
import numpy as np
import torch
from torch import nn

import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
tokenizer = gpt2.tokenizer

loader = transformer_lens.evals.make_pile_data_loader(tokenizer=tokenizer, batch_size=8)


pre = None
post = None

def mlp_pre(value, hook):
    global pre
    pre = value.detach().clone().cpu()
    return value

def mlp_post(value, hook):
    global post
    post = value.detach().clone().cpu()
    return value

layer_to_hook = 9
fwd_hooks = [
    (utils.get_act_name("attn_out", layer_to_hook), mlp_pre),
    (utils.get_act_name("mlp_out", layer_to_hook), mlp_post)
]

pre_activations = []
post_activations = []

for idx, batch in enumerate(loader):
    with torch.no_grad():
        loss = gpt2.run_with_hooks(
            batch["tokens"].to(device),
            return_type="loss",
            fwd_hooks=fwd_hooks,
        )
    pre_activations.append(pre.numpy())
    post_activations.append(post.numpy())
    if idx > 500:
        break

pre_activations_np = np.concatenate(pre_activations, axis=0)
post_activations_np = np.concatenate(post_activations, axis=0)

checkpoint_dir = "checkpoints"
np.save(os.path.join(checkpoint_dir, "pre_activations.npy"), pre_activations_np)
np.save(os.path.join(checkpoint_dir, "post_activations.npy"), post_activations_np)



