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
            acts_cache[layer] = value[:, :, selected_for_layer].detach().cpu() # shape is (batch_size, seq_len, n_neurons)
            return value

        return hook

    fwd_hooks = []
    for layer in range(len(model.blocks)):
        fwd_hooks.append((f"blocks.{layer}.mlp.hook_post", get_activation_hook(layer)))

    best_so_far = [[] for _ in range(n_neurons)] # list of lists of {neuron, activations, tokens, max_activation}
    for batch_idx, batch in enumerate(dataset_loader):

        if batch_idx == 5:
            break

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

def extract_chunk(tokens, activations, chunk_size=100):
    max_index = activations.index(max(activations))
    start = max(0, max_index - chunk_size // 2)
    end = min(len(tokens), max_index + chunk_size // 2)
    return tokens[start:end], activations[start:end]


def store_results_to_json(best_examples: List[List[Dict]], tokenizer, filename: str):
    """
    Stores the top k activations for each neuron along with the corresponding input tokens in a JSON file.
    """

    results = {}
    for neuron_best_examples in best_examples:
        examples = []
        for example in neuron_best_examples:
            neuron = example["neuron"]
            tokens = example["tokens"]
            activations = example["activations"].tolist()
            max_activation = example["max_activation"]

            # Extract the chunk of tokens and activations around the max activation token
            chunk_tokens, chunk_activations = extract_chunk(tokens, activations)

            # Decode the chunk of tokens to text
            tokens_text = tokenizer.decode(chunk_tokens)

            # Create token-activation pairs for the chunk of tokens and activations
            token_activation_pairs = [(tokenizer.decode(token), activation) for token, activation in zip(chunk_tokens, chunk_activations)]

            examples.append({
                "tokens": tokens_text,
                "max_activation": max_activation,
                "token_activation_pairs": token_activation_pairs
            })

        if examples:
            results[str(neuron)] = {"examples": examples}

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

# Main function to select random neurons, find top examples, and save the results

def main(model, dataset_loader):
    num_neurons = 10
    total_neurons = 2048 * len(model.blocks)
    top_k = 20

    selected_neurons = random.sample(range(total_neurons), num_neurons)
    top_k_examples = get_top_examples(model, dataset_loader, selected_neurons, top_k)
    store_results_to_json(best_examples=top_k_examples, tokenizer=tokenizer, filename="top_examples.json")

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained('gelu-4l')

    tokenizer = model.tokenizer
    print(dir(tokenizer))
    dataset_loader = big_data_loader(tokenizer=tokenizer, batch_size=8, big=False)

    main(model, dataset_loader)
