"""Original data parsing, training-only preprocessing, and evaluation metrics."""
import ast
import json
from pathlib import Path
import numpy as np
from scipy import sparse

CONTINUOUS = [0, 1, 2, 3, 4, 5, 7]
FEATURES = ["beer/ABV", "review/appearance", "review/aroma", "review/palate", "review/taste"]
CATEGORIES = ["beer/style", "user/gender", "beer/brewerId"]
def read_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Record at line {line_no} is not a dictionary")
                records.append(row)
    return records


def category(value):
    return None if value is None or str(value).strip() == "" else str(value)


def raw_arrays(records, require_target=True):
    numeric = np.empty((len(records), 8), dtype=np.float64)
    cats = [[category(r.get(key)) for r in records] for key in CATEGORIES]
    for i, row in enumerate(records):
        values = np.array([float(row[key]) for key in FEATURES])
        timestamp = float(row["review/timeUnix"])
        if not np.isfinite(values).all() or not np.isfinite(timestamp):
            raise ValueError(f"Invalid required numerical input at record {i}; no rows discarded")
        age = np.nan
        try:
            birthday = float(row["user/birthdayUnix"])
            if np.isfinite(birthday) and birthday <= timestamp:
                age = (timestamp - birthday) / (365.25 * 86400)
        except (KeyError, TypeError, ValueError):
            pass
        numeric[i] = [*values, age, float(np.isnan(age)), timestamp / 86400]
    y = None
    if require_target:
        y = np.array([float(row["review/overall"]) for row in records], dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite targets; no rows discarded")
    return numeric, cats, y


def fit_preprocessor(numeric, cats, indices):
    fit = numeric[indices].copy()
    valid_ages = fit[:, 5][np.isfinite(fit[:, 5])]
    if not len(valid_ages):
        raise ValueError("No valid training ages from which to fit the median")
    median = float(np.median(valid_ages))
    fit[~np.isfinite(fit[:, 5]), 5] = median
    means = fit[:, CONTINUOUS].mean(axis=0)
    stds = fit[:, CONTINUOUS].std(axis=0)
    stds[stds == 0] = 1
    vocab = {key: sorted({cats[j][i] for i in indices if cats[j][i] is not None})
             for j, key in enumerate(CATEGORIES)}
    return dict(age_median=median, means=means.tolist(), stds=stds.tolist(),
                vocabularies=vocab, continuous_indices=CONTINUOUS,
                numerical_columns=FEATURES + ["age_years", "age_missing", "review_days"],
                category_order=CATEGORIES, unknown_rule="last column in each categorical block",
                fit_record_count=len(indices))


def transform(numeric, cats, preprocessor, with_brewery):
    values = numeric.copy()
    values[~np.isfinite(values[:, 5]), 5] = preprocessor["age_median"]
    values[:, CONTINUOUS] = ((values[:, CONTINUOUS] - preprocessor["means"])
                            / preprocessor["stds"])
    blocks = [sparse.csr_matrix(values.astype(np.float32))]
    for j, key in enumerate(CATEGORIES[:3 if with_brewery else 2]):
        vocab = preprocessor["vocabularies"][key]
        mapping = {v: k for k, v in enumerate(vocab)}
        columns = np.array([mapping.get(v, len(vocab)) for v in cats[j]])
        blocks.append(sparse.csr_matrix((np.ones(len(values), dtype=np.float32),
                      (np.arange(len(values)), columns)), shape=(len(values), len(vocab) + 1)))
    result = sparse.hstack(blocks, format="csr", dtype=np.float32)
    assert np.isfinite(result.data).all()
    return result


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    error = prediction - y
    nonzero = y != 0
    mape = float(100 * np.mean(np.abs(error[nonzero] / y[nonzero]))) if nonzero.any() else None
    return dict(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                half_mse=float(np.mean(error**2) / 2),
                ordinary_mape_percent=mape if nonzero.all() else None,
                nonzero_target_mape_percent=mape, zero_target_count=int((~nonzero).sum()),
                count=len(y))
