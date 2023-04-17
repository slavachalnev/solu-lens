# solu-probe

## Idea
Make an MLP layer more interpretable by training a more interpretable model to mimic it.

Diagram: https://excalidraw.com/#json=UzZhSh4m_GFG_vLZUgWW6,sNxHzvMZM2cF3RO2xpv6kw

## Implementation
So far the main toy data experiment is in `projection.py` which first trains a model (see `mlp.py`) on toy projection data (see `toy_data.py`), then trains a model to mimic the first model's behaviour. The second model is either a GELU or SoLU MLP.

The real model experiment is in `graft_model.py` which trains 4 SoLU MLPs to mimic the 4 MLP layers of a pretrained `solu-4l` model from Transfomrer Lens.

I took the toy dataset setup, the monosemanticity metric, and the plot layout from [Engineering Monosemanticity in Toy Models](https://arxiv.org/pdf/2211.09169.pdf). SoLU is from [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html).

## Results
![alt text](assets/all_plots.png "One run. Sparse power-law features.")
