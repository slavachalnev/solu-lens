import torch
import random
import json
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple

# Import necessary libraries
import torch.nn as nn
import numpy as np

# Import custom big_data_loader from utils.py
from utils import big_data_loader

# Define helper functions


def select_random_neurons(num_neurons: int, total_neurons: int) -> List[int]:
    return random.sample(range(total_neurons), num_neurons)


def merge_top_k_examples(current_top_k: Dict[int, List[Tuple[int, torch.Tensor]]], new_activations: Dict[int, List[Tuple[int, torch.Tensor]]], k: int) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
    merged_top_k = {}
    for neuron in current_top_k.keys():
        merged_activations = current_top_k[neuron] + new_activations[neuron]
        merged_top_k[neuron] = sorted(merged_activations, key=lambda x: torch.max(x[1]).item(), reverse=True)[:k]
    return merged_top_k


def get_example_activations(model, dataset_loader, neurons: List[int], k: int) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
    top_k_activations = {neuron: [] for neuron in neurons}
    for batch_idx, (data, target) in enumerate(dataset_loader):
        # Your implementation here to get activations for selected neurons in the current batch
        batch_activations = {} # Replace this with the actual activations for the selected neurons in the current batch
        top_k_activations = merge_top_k_examples(top_k_activations, batch_activations, k)
    return top_k_activations


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
    num_neurons = 100
    total_neurons = 2048  # Replace this with the number of MLP neurons in your model
    top_k = 20

    selected_neurons = select_random_neurons(num_neurons, total_neurons)
    top_k_examples = get_example_activations(model, dataset_loader, selected_neurons, top_k)
    store_results_to_json(top_k_examples, dataset_loader, 'results.json')

if __name__ == "__main__":
    model = torch.load('path/to/model.pt')
    model.eval()

    tokenizer = None  # Replace with your tokenizer
    dataset_loader = big_data_loader(tokenizer=tokenizer, batch_size=8, big=False)

    main(model, dataset_loader)
