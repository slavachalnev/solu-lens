# solu-probe

Makes an MLP layer more interpretable by training a more interpretable model to mimic it.

So far the main experiment is in `projection.py` which first trains a model (see `mlp.py`) on toy projection data (see `toy_data.py`), then trains a model to mimic the first model's behaviour. The second model is either a GeLU or SoLU MLP.

## Results
![alt text](assets/all_plots.png "One run. Sparse power-law features.")
