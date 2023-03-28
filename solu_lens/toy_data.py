import torch
import torch.nn.functional as functional
import numpy as np
from torch.utils.data import Dataset

class ToyNeuralDataset(Dataset):
    """
    Follows https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition
    """
    def __init__(self, h=256, G=512, num_samples=1000, avg_active_features=5, lambda_decay=0.99):
        self.h = h
        self.G = G
        self.num_samples = num_samples

        # Generate ground truth features
        features = torch.randn(h, G)
        self.F = functional.normalize(features, dim=0)

        # Precompute correlated, decayed, rescaled probabilities
        self.probs = self.compute_probs(G, lambda_decay, avg_active_features)

    def compute_probs(self, G, lambda_decay, avg_active_features):
        # Generate a random covariance matrix
        cov = torch.randn(G, G)
        cov = torch.mm(cov, cov.t())
        cov = cov / cov.max()

        # Sample a correlated multivariate normal distribution
        sample = torch.distributions.MultivariateNormal(torch.zeros(G), cov).sample()

        # Calculate the standard normal cumulative distribution function
        correlated_probs = torch.tensor([torch.distributions.Normal(0, 1).cdf(val) for val in sample])

        # Apply exponential decay
        decayed_probs = correlated_probs * torch.tensor([lambda_decay**g for g in range(G)])

        # Rescale probabilities to ensure on average 5 out of G features are active
        mean_prob = decayed_probs.mean()*G
        rescale_ratio = avg_active_features / mean_prob
        rescaled_probs = decayed_probs * rescale_ratio

        return rescaled_probs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample G-dimensional binary random variable using the precomputed probabilities
        binary_rv = torch.distributions.Bernoulli(self.probs).sample()

        # Scale the binary vector with a uniform random vector
        activations = binary_rv * torch.rand(self.G)

        # Linearly combine the ground truth features using the sparse activations
        sample = torch.mv(self.F, activations)

        return sample, activations

# Usage
dataset = ToyNeuralDataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for sample, activations in data_loader:
    print(sample.shape, activations.shape)
    print('sample', sample[:10])
    print('activations', activations[:10])
    break
