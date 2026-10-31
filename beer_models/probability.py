"""Sigmoid hidden layers, 11-class softmax, cross-entropy, and rating decoders."""
import time
import numpy as np
from scipy.special import logsumexp
from . import data as nn
from . import regression as arch

SCORES = np.arange(0, 5.01, .5)
def labels(y):
    ids = np.rint(np.asarray(y)*2).astype(int)
    assert np.all((ids >= 0) & (ids < len(SCORES)))
    assert np.array_equal(SCORES[ids], y), "Target outside declared half-point rating grid"
    return ids


def initialize(width, shared, layout, seed, dtype=np.float32):
    weights = arch.initialize(width, shared, layout, seed, dtype)
    # Match hidden initialization to the earlier experiment; use a separate output RNG.
    last = len(weights)//2
    weights[f"W{last}"] = np.random.default_rng(seed+300000).normal(
        0, .1, (layout[-1], len(SCORES))).astype(dtype)
    weights[f"b{last}"] = np.zeros(len(SCORES), dtype=dtype)
    return weights


def forward(x, weights):
    _, acts = arch.forward(x, weights)
    logits = acts[-1]
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logp), logp, acts


def gradients(x, target, weights):
    p, _, acts = forward(x, weights)
    delta = p.copy()
    delta[np.arange(len(target)), target] -= 1
    delta /= len(target)
    result = {}
    for i in range(len(weights)//2, 0, -1):
        result[f"W{i}"] = np.asarray(acts[i-1].T @ delta)
        result[f"b{i}"] = delta.sum(axis=0)
        if i > 1:
            h = acts[i-1]
            delta = (delta @ weights[f"W{i}"].T)*h*(1-h)
    return result


def decode(p):
    return dict(mode=SCORES[p.argmax(axis=1)], mean=p @ SCORES,
                median=SCORES[(p.cumsum(axis=1) >= .5).argmax(axis=1)])


def metrics(y, p, logp):
    target = labels(y)
    onehot = np.eye(len(SCORES))[target]
    decoded = decode(p)
    result = dict(nll=float(-logp[np.arange(len(y)), target].mean()),
                  brier=float(np.mean(np.sum((p-onehot)**2, axis=1))),
                  ranked_probability_score=float(np.mean((p.cumsum(1)[:, :-1]-onehot.cumsum(1)[:, :-1])**2)),
                  exact_accuracy=float(np.mean(decoded["mode"] == y)), count=len(y))
    result["decoders"] = {k: dict(nn.metrics(y, v), within_half=float(np.mean(abs(v-y) <= .5)),
                                bias=float(np.mean(v-y))) for k, v in decoded.items()}
    return result


def train(x, y, shared, layout, config, seed, xv=None, yv=None, epochs=None):
    weights = initialize(x.shape[1], shared, layout, seed)
    first = {k: np.zeros_like(v) for k, v in weights.items()}
    second = {k: np.zeros_like(v) for k, v in weights.items()}
    target = labels(y)
    shuffle = np.random.default_rng(seed+200000)
    best_loss = progress_loss = float("inf")
    best = None
    best_epoch = progress_epoch = step = 0
    history = []
    limit = epochs or config["max_epochs"]
    started = time.perf_counter()
    for epoch in range(1, limit+1):
        order = shuffle.permutation(len(y))
        for start in range(0, len(y), config["batch_size"]):
            ix = order[start:start+config["batch_size"]]
            grad = gradients(x[ix], target[ix], weights)
            step += 1
            for key in weights:
                if config["optimizer"] == "adam":
                    first[key] *= .9
                    first[key] += .1*grad[key]
                    second[key] *= .999
                    second[key] += .001*grad[key]**2
                    update = (first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
                else:
                    first[key] *= config["momentum"]
                    first[key] += grad[key]
                    update = first[key]
                weights[key] -= config["lr"]*update
        if epoch % 5 and epoch != limit:
            continue
        p, lp, _ = forward(x, weights)
        assert np.isfinite(lp).all()
        row = dict(epoch=epoch, train_nll=float(-lp[np.arange(len(y)), target].mean()))
        if xv is not None:
            pv, lv, _ = forward(xv, weights)
            vm = metrics(yv, pv, lv)
            row.update(val_nll=vm["nll"], val_accuracy=vm["exact_accuracy"],
                       val_mode_mae=vm["decoders"]["mode"]["mae"],
                       val_mean_mae=vm["decoders"]["mean"]["mae"],
                       val_median_mae=vm["decoders"]["median"]["mae"])
            if vm["nll"] < best_loss:
                best_loss, best_epoch = vm["nll"], epoch
                best = {k: v.copy() for k, v in weights.items()}
            if vm["nll"] < progress_loss-1e-5:
                progress_loss, progress_epoch = vm["nll"], epoch
        history.append(row)
        if xv is not None and epoch-progress_epoch >= config["patience"]:
            break
    return best if xv is not None else weights, dict(best_epoch=best_epoch if xv is not None else limit,
        completed_epochs=epoch, reached_epoch_limit=epoch == limit,
        seconds=time.perf_counter()-started, history=history)
