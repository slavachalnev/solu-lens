import torch
import random
import json
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple

# Import necessary libraries
import torch.nn as nn
import numpy as np

import transformer_lens
from transformer_lens import HookedTransformer

from utils import big_data_loader


def get_top_examples(model, dataset_loader, neurons: List[int], k: int):
    # hooks to extract neuron activations
    d_mlp = 2048
    n_neurons = len(neurons)
    acts_cache = [None for _ in range(len(model.blocks))]

    def get_activation_hook(layer):
        # neurons from d_mlp*layer to d_mlp*(layer+1)
        selected_for_layer = [n for n in neurons if n >= d_mlp*layer and n < d_mlp*(layer+1)]
        selected_for_layer = [n - d_mlp*layer for n in selected_for_layer]

        def hook(value, hook):
            acts_cache[layer] = value[:, :, selected_for_layer]
            return value

        return hook

    fwd_hooks = []
    for layer in range(len(model.blocks)):
        fwd_hooks.append((f"blocks.{layer}.mlp.hook_post", get_activation_hook(layer)))

    best_so_far = [[] for _ in range(n_neurons)] # list of lists of {neuron, activations, tokens, max_activation}
    for batch_idx, batch in enumerate(dataset_loader):
        with model.hooks(fwd_hooks=fwd_hooks), torch.no_grad():
            model(batch["tokens"], return_type="loss")
        
        # stack activations from all layers
        batch_activations = torch.cat(acts_cache, dim=2) # shape is (batch_size, seq_len, n_neurons)

        for i, example in enumerate(batch["tokens"]):
            for j, neuron in enumerate(neurons):
                d = {
                    "neuron": neuron,
                    "activations": batch_activations[i, :, j],
                    "tokens": example,
                    "max_activation": torch.max(batch_activations[i, :, j]).item(),
                }
                best_so_far[j].append(d)
        
        # keep only top k activations
        for j in range(n_neurons):
            best_so_far[j] = sorted(best_so_far[j], key=lambda x: x["max_activation"], reverse=True)[:k]
        
    return best_so_far


def store_results_to_json(top_k_activations: Dict[int, List[Tuple[int, torch.Tensor]]], dataset_loader, filename: str):
    results = {}
    for neuron, activation_list in top_k_activations.items():
        results[neuron] = []
        for example_idx, activation in activation_list:
            data, _ = dataset_loader.dataset[example_idx]
            results[neuron].append({
                'tokens': data.tolist(),
                'activations': activation.tolist(),
                'max_activation': torch.max(activation).item()
            })
    with open(filename, 'w') as f:
        json.dump(results, f)


# Main function to select random neurons, find top examples, and save the results

def main(model, dataset_loader):
    num_neurons = 10
    total_neurons = 2048 * len(model.blocks)
    top_k = 20

    selected_neurons = random.sample(range(total_neurons), num_neurons)
    top_k_examples = get_top_examples(model, dataset_loader, selected_neurons, top_k)
    print(top_k_examples)
    # store_results_to_json(top_k_examples, dataset_loader, 'results.json')

if __name__ == "__main__":
    # model = torch.load('path/to/model.pt')
    # model.eval()
    model = HookedTransformer.from_pretrained('gelu-4l')

    tokenizer = model.tokenizer
    dataset_loader = big_data_loader(tokenizer=tokenizer, batch_size=8, big=False)

    main(model, dataset_loader)
