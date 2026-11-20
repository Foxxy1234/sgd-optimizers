# SGD Optimizers

An educational comparison of first-order optimization methods for linear and
polynomial regression, with additional experiments on soft-margin SVMs and
gradient computation.

## What is included

- NumPy implementations of SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam,
  and Lion.
- Wrappers around PyTorch optimizers for reference comparisons, including AdamW.
  These wrappers consume gradients supplied by the training loop; they are not
  independent implementations of the PyTorch algorithms.
- Mini-batch training, learning-rate schedules, L1/L2/ElasticNet regularization,
  convergence plots, and timing/memory measurements.
- A custom forward-mode automatic differentiation implementation using dual
  numbers, central finite differences, and comparisons with PyTorch's
  reverse-mode `torch.autograd`.
- A Russian-language notebook containing explanations and saved experiment
  outputs on synthetic data and California Housing.

## Layout

```text
report.ipynb                 Experiment entry point and saved results
requirements.txt            Runtime, notebook, and linting dependencies
src/
  autodiff/                 Dual numbers, numerical gradients, benchmarks
  data/                     Synthetic generation and California Housing loader
  models/                   Linear and polynomial regression
  optimizers/               NumPy optimizer implementations
  svm/                      Soft-margin linear SVM
  training/                 Mini-batch trainer and PyTorch wrappers
  plotting.py               Plot helpers
  regularizers.py           Regularization penalties and gradients
  schedulers.py             Learning-rate schedules
```

## Setup and use

Use a Python environment compatible with the dependencies in `requirements.txt`.
The saved notebook metadata records Python 3.13.15; this is historical metadata,
not a verified compatibility requirement. Dependencies are not version-locked.

For a conventional Python environment, the following commands are a starting
point (installation and execution were not performed while preparing this copy):

```bash
cd /path/to/sgd-optimizers
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `report.ipynb` and select the Python kernel from that environment. Keep the
kernel's working directory at the project root: the first code cell adds
`src/` to the module search path. The notebook is the experiment entry point;
there is no separate command-line application.

You can inspect the saved outputs without running cells. To reproduce the
experiments, run cells in order in a fresh kernel, subject to the data caveat
below. This runs training and benchmarks and may take time.

### Data

No dataset files or local dataset cache are included. Synthetic data is generated
by the code. `src/data/real.py` calls scikit-learn's
`fetch_california_housing`, which may download the dataset and write to
scikit-learn's cache when it is not already available. Check the upstream data
source's terms and availability before fetching or redistributing data.
Offline execution of the full notebook requires an existing compatible cache.

## Results and validation

Notebook tables, figures, execution counts, timestamps, and environment metadata
are preserved from the original work. They are saved historical results, not
freshly reproduced or independently verified results. Preparing this local copy
did not execute the notebook, train models, download datasets, or verify numerical
correctness or current dependency compatibility.

## Attribution

This educational project was developed collaboratively, with **Ivan Davydov as
the primary contributor**. It is not presented as solely authored work.

The work originated in a private course repository. Course CI, assignment
materials, and template READMEs are not included in this portfolio-oriented copy.

## License

The code and report are available under the [MIT License](LICENSE).
Third-party dependencies and external datasets retain their respective licenses.
