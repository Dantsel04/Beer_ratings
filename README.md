# Beer ratings — Colab project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dantsel04/Beer_ratings/blob/main/Beer_Ratings_Colab.ipynb)

Open the notebook above and run its cells in order. Select a CPU runtime. The
first cell clones this repository or pulls subsequent updates with `--ff-only`.
Run `quick` mode to check the pipeline; use the default `full` mode for final results.
Full mode runs 18 selection fits plus two final neural-network refits.

## Dataset

The original `beer_50000.json` is not committed. The notebook attempts the course
download; if the server rejects it, upload your local file when prompted. The
SHA-256 check requires the exact original 50,000-record dataset. Alternatively,
change `DATA_PATH` to your dataset in mounted Drive. All records are retained.

## Models and interpretation

The notebook compares training-mean and learned linear-regression baselines with
16 → 16 sigmoid networks: a linear regression output and an 11-class softmax.
The three small modules preserve the original project's NumPy/SciPy parsing,
preprocessing, forward passes, manual backpropagation, optimizers, and decoding.
No previous results directory, Windows path, PyTorch or Intel GPU setup is needed.
The notebook shows splitting, model selection, refitting, evaluation, and plotting.

Inputs include component review scores, so this estimates overall scores from
existing review information. The fixed 32,000/8,000/10,000 split reproduces the
earlier experiments; comparisons are exploratory because that test set was
previously inspected. A random split can share beers and reviewers. This focused
notebook does not repeat the earlier architecture/feature search.

## Updating from GitHub

Push changes to `main`, then rerun the notebook's first cell. Restart the runtime
and run all cells after updating Python modules. Reopen the GitHub notebook link
to get updated notebook cells; pulling the checkout cannot alter an already-open
notebook. Local edits block the pull so they cannot be silently overwritten.

Save output ZIPs before the runtime expires. They contain metrics, predictions,
probabilities, split indices, model weights, preprocessing, histories and provenance.
Training is CPU-only; assigning a Colab GPU does not accelerate NumPy/SciPy.

## Submission

Add team names, run full mode, and put a viewable Colab link at the beginning of
the 1–2 page report. Include both MAE and MAPE and explain zero-target handling.
