# SF3 Machine Learning

This repository contains my work for the **SF3 Machine Learning** project, Easter Term 2026, Department of Engineering, University of Cambridge.

The project studies data-driven modelling and control of the CartPole system. The first part focuses on learning one-step dynamics models from simulator data. The later part uses these learned models to support control design and policy optimisation.

## Project Overview

The state of the CartPole system is represented as

[
X = [x, \dot{x}, \theta, \dot{\theta}]
]

and the supervised learning target is usually the one-step state change

[
\Delta X = X(T) - X(0),
]

where (T) corresponds to one call of `performAction()`.

The modelling workflow follows the standard machine-learning structure:

1. generate or load simulator trajectories;
2. split data into training, validation, and test sets;
3. train model parameters on the training set;
4. tune hyperparameters using validation error;
5. reserve the test set for final model checks;
6. evaluate both one-step prediction error and recursive rollout behaviour.

A key lesson from the project is that low one-step mean squared error is not sufficient. A model may fit local dynamics reasonably well but still fail when recursively rolled out over many steps.

## Repository Contents

The repository contains code, notebooks, figures, saved datasets, and report material for the SF3 project.

A typical structure is:

```text
SF3_Machine_Learning/
├── notebooks/          # Jupyter notebooks for experiments and task work
├── code/               # Python scripts and reusable functions
├── data/               # Saved generated data, train/validation/test splits
├── figures/            # Generated figures used in the report
├── reports/            # Interim report LaTeX/PDF files
├── results/            # Saved experimental outputs
└── README.md
```

Some folder names may differ slightly depending on the working version of the repository, but the main purpose is to keep code, generated data, figures, and report files separated.

## Main Work Completed

### Toy Experiments

Before working directly on the project tasks, I ran small experiments to understand the practical workflow behind machine learning:

* linear regression;
* ridge regularisation;
* train/validation/test splitting;
* Gaussian basis functions;
* rollout-error accumulation;
* basic PyTorch and JAX gradient usage;
* optimisation using objective functions and gradients.

These experiments helped clarify the difference between fitting a one-step model and trusting a learned model when it is iterated recursively.

### Task 1: Simulation and Linear Modelling

Task 1 involved understanding the CartPole simulator and fitting a baseline model.

The work included:

* generating zero-force CartPole trajectories;
* saving trajectory arrays and static figures;
* analysing state-response maps of (X \mapsto \Delta X);
* fitting affine one-step models;
* testing local prediction and longer-horizon rollout;
* identifying that linear models can become qualitatively wrong when rolled out.

A specific issue was angle handling. The simulator angle may evolve continuously, while some plots or learned-model iterations use remapped angles. To avoid inconsistent figures, raw simulator angles and remapped plotting/model angles were treated separately.

### Task 2: Kernel Models and Periodic Features

Task 2 extended the model class using nonlinear features and sparse Gaussian-kernel regression.

The work included:

* implementing Gaussian-kernel models;
* constructing kernel matrices such as (K_{NM}), (K_{MN}), and (K_{MM});
* using linear solves rather than explicit matrix inversion;
* tuning kernel length scales and regularisation with JAX and SciPy L-BFGS-B;
* replacing raw (\theta) with (\sin \theta) and (\cos \theta) to handle periodicity;
* comparing raw linear, sin/cos linear, and sin/cos kernel models;
* evaluating one-step MSE, theta-scan behaviour, and rollout error.

The sin/cos kernel model gave the best overall performance, although it still accumulated error over longer open-loop rollouts.

## Technical Issues Encountered

Several practical issues occurred during the project:

| Issue                              | Effect                                        | Resolution                                                             |
| ---------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| Jupyter animation unreliable       | animation was not useful as report evidence   | saved numerical arrays and static figures                              |
| angle remapping inconsistency      | repeated rollout plots could differ           | separated raw simulator angle from remapped angle                      |
| random train/validation/test split | figures were hard to reproduce                | saved split indices and fitted parameters                              |
| kernel tensor broadcasting         | silent algebra or shape errors                | checked (N \times M), (M \times N), and (M \times M) shapes explicitly |
| hyperparameter optimisation        | difficult to interpret optimisation behaviour | used validation-MSE objective and inspected rollout error separately   |

## Interim Report

The interim report summarises the first two weeks of the project. It covers:

* a brief overview of the project;
* progress on simulation, state-response maps, linear models, kernel models, and periodic features;
* key figures comparing one-step prediction and rollout performance;
* implementation issues and resolutions;
* the plan for the control stage.

The submitted report is limited to three A4 pages and follows the project requirement of staying within 750 words, including headings and captions.

## Plan for the Remaining Project

The next stage is control. The intended workflow is:

1. add force/action as an input to the learned dynamics model;
2. collect random-action one-step data;
3. fit action-conditioned linear and kernel models;
4. define a feedback policy of the form

[
u = p^T \phi(X);
]

5. construct a trajectory loss penalising pole angle, cart displacement, velocities, and control effort;
6. optimise policy parameters using JAX;
7. compare policies optimised using the true simulator with policies optimised through learned-model rollouts.

This control stage is closer to direct policy search than to purely analytic LQR. The learned model may be useful for short-horizon model-based control, but long open-loop forecasts should not be trusted without further validation.

## Software

The project uses Python and Jupyter notebooks. Main packages include:

* NumPy;
* Matplotlib;
* SciPy;
* JAX;
* PyTorch, for preparatory toy experiments.

## Notes on Reproducibility

Where possible, generated data, train/validation/test split indices, fitted parameters, and figures are saved so that experiments can be reproduced. Some figures were formatted with AI assistance, but numerical results were generated by running the project code.

## References

* SF3 Machine Learning Project Information, Easter Term 2026.
* JAX documentation: https://docs.jax.dev
* SciPy optimisation documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
* NumPy documentation: https://numpy.org/doc/stable/
