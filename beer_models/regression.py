"""Sigmoid hidden layers, linear output, half-MSE, and manual backpropagation."""
import time
import numpy as np
from scipy.special import expit
from . import data as nn
def initialize(width, shared_width, hidden_layers, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    first_shared = rng.normal(0, .1, (shared_width, hidden_layers[0]))
    extra = np.random.default_rng(seed+100000).normal(0, .1, (width-shared_width, hidden_layers[0]))
    weights = {"W1": np.vstack([first_shared, extra]).astype(dtype),
               "b1": np.zeros(hidden_layers[0], dtype=dtype)}
    sizes = list(hidden_layers) + [1]
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]), 2):
        weights[f"W{i}"] = rng.normal(0, .1, (n_in, n_out)).astype(dtype)
        weights[f"b{i}"] = np.zeros(n_out, dtype=dtype)
    return weights


def forward(x, weights):
    activations = [x]
    count = len(weights)//2
    for i in range(1, count+1):
        z = activations[-1] @ weights[f"W{i}"] + weights[f"b{i}"]
        activations.append(expit(z) if i < count else z)
    return np.asarray(activations[-1]).ravel(), activations


def gradients(x, y, weights):
    prediction, activations = forward(x, weights)
    delta = ((prediction-y)/len(y)).reshape(-1, 1)
    result = {}
    for i in range(len(weights)//2, 0, -1):
        result[f"W{i}"] = np.asarray(activations[i-1].T @ delta)
        result[f"b{i}"] = delta.sum(axis=0)
        if i > 1:
            h = activations[i-1]
            delta = (delta @ weights[f"W{i}"].T) * h * (1-h)
    return result


def train(x, y, shared_width, layout, config, seed, x_val=None, y_val=None, epochs=None):
    weights = initialize(x.shape[1], shared_width, layout, seed)
    first = {k: np.zeros_like(v) for k,v in weights.items()}
    second = {k: np.zeros_like(v) for k,v in weights.items()}
    shuffle = np.random.default_rng(seed+200000)
    step = 0
    best_mae = significant_best = float("inf")
    best, best_epoch, last_progress = None, 0, 0
    history = []
    limit = epochs if epochs is not None else config["max_epochs"]
    batch_size = config["batch_size"] or len(y)
    started = time.perf_counter()
    for epoch in range(1, limit+1):
        order = shuffle.permutation(len(y)) if config["batch_size"] else None
        for start in range(0, len(y), batch_size):
            ix = order[start:start+batch_size] if order is not None else slice(None)
            grad = gradients(x[ix], y[ix], weights)
            step += 1
            for key in weights:
                if config["optimizer"] == "adam":
                    first[key] *= .9
                    first[key] += .1 * grad[key]
                    second[key] *= .999
                    second[key] += .001 * grad[key]**2
                    update = ((first[key]/(1-.9**step)) /
                              (np.sqrt(second[key]/(1-.999**step))+1e-8))
                else:
                    first[key] *= config["momentum"]
                    first[key] += grad[key]
                    update = first[key]
                weights[key] -= config["lr"]*update
        if epoch % 5 and epoch != limit:
            continue
        if not all(np.isfinite(v).all() for v in weights.values()):
            raise FloatingPointError(f"Nonfinite parameters: {layout}, {config['name']}, seed {seed}")
        fitting = nn.metrics(y, forward(x, weights)[0])
        entry = dict(epoch=epoch, train_mae=fitting["mae"], train_half_mse=fitting["half_mse"])
        if x_val is not None:
            validation = nn.metrics(y_val, forward(x_val, weights)[0])
            entry.update(val_mae=validation["mae"], val_half_mse=validation["half_mse"])
            if validation["mae"] < best_mae:
                best_mae, best_epoch = validation["mae"], epoch
                best = {k:v.copy() for k,v in weights.items()}
            if validation["mae"] < significant_best-1e-5:
                significant_best, last_progress = validation["mae"], epoch
        history.append(entry)
        if epoch % 100 == 0:
            print(f"  layout={layout} {config['name']} seed={seed} epoch={epoch} MAE={entry.get('val_mae',entry['train_mae']):.5f}", flush=True)
        if x_val is not None and epoch-last_progress >= config["patience"]:
            break
    return (best if x_val is not None else weights), dict(
        best_epoch=best_epoch if x_val is not None else limit, completed_epochs=epoch,
        stopped_early=epoch<limit, reached_epoch_limit=epoch==limit,
        seconds=round(time.perf_counter()-started,3), history=history)
