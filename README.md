# SF3 Machine Learning

This repository contains my work for the **SF3 Machine Learning** project, Easter Term 2026, Department of Engineering, University of Cambridge.

The project studies data-driven modelling and control of the CartPole system. The first stage focuses on learning one-step dynamics models from simulator data. The later stage uses these learned models to support controller design and policy optimisation.

## Project Aim

The CartPole state is represented as

$$
X = [x, \dot{x}, \theta, \dot{\theta}],
$$

where $x$ is the cart position, $\dot{x}$ is the cart velocity, $\theta$ is the pole angle, and $\dot{\theta}$ is the angular velocity.

Most modelling tasks use the one-step state change as the supervised learning target:

$$
\Delta X = X(T) - X(0),
$$

where $T$ corresponds to one call of `performAction()`.

This target is useful because it removes the near-identity part of the short-time dynamics and makes the local state change easier to model.

## Repository Structure

The repository is organised into two main folders:

```text
SF3_Machine_Learning/
├── README.md
├── SF3 Machine Learning/
│   ├── SF3 handout.pdf
│   ├── cartpole.py
│   ├── data/
│   ├── figures/
│   ├── __pycache__/
│   ├── sf3_w1_task1_1_rollouts.ipynb
│   ├── sf3_w1_task1_2_state_change_scans.ipynb
│   ├── sf3_w1_task1_3_linear_model.ipynb
│   ├── sf3_w1_task1_4_linear_rollout.ipynb
│   ├── sf3_w2_task2_1_kernel_model.ipynb
│   ├── sf3_w2_task2_2_jax_hyperparams.ipynb
│   └── sf3_w2_task2_3_sincos_features.ipynb
│
└── SF3_pretraining/
    ├── notebooks/
    ├── figures/
    └── experiment_log.md
```

The folder `SF3 Machine Learning/` contains the main project work. It includes the project handout, the supplied `cartpole.py` simulator, generated data, report figures, and the main notebooks for Week 1 and Week 2 tasks.

The folder `SF3_pretraining/` contains preparatory toy experiments used before starting the main project tasks. These experiments cover basic regression, train/validation/test splitting, regularisation, Gaussian basis functions, PyTorch/JAX usage, and rollout-error behaviour. They are not part of the final model pipeline, but they document the learning process that supported the main project work.

## Workflow

The modelling workflow follows a standard machine-learning structure:

1. generate or load CartPole simulator data;
2. split data into training, validation, and test sets;
3. fit model parameters on the training set;
4. tune hyperparameters using validation error;
5. reserve the test set for final checks;
6. evaluate both one-step prediction error and recursive rollout behaviour.

A key lesson from the project is that low one-step mean squared error is not sufficient. A model can fit local transitions reasonably well but still fail when it is recursively rolled out over many time steps.

## Preliminary Toy Experiments

Before working through the main project tasks, I ran several toy experiments to understand the practical logic of machine learning and optimisation:

* linear regression;
* ridge regularisation;
* train/validation/test splitting;
* Gaussian basis functions;
* objective functions and gradients;
* basic PyTorch usage;
* basic JAX automatic differentiation;
* rollout-error accumulation.

These experiments helped clarify the connection between regression, linear algebra, optimisation, and model validation.

## Task 1: Simulation and Linear Modelling

The first stage focused on understanding the CartPole simulator and building a baseline linear model.

Main work completed:

* generated zero-force CartPole trajectories using `cartpole.py`;
* saved numerical trajectories and static figures;
* analysed state-response maps of $X \mapsto \Delta X$;
* studied how $\theta$ and $\dot{\theta}$ affect acceleration-related outputs;
* fitted affine one-step models;
* compared local prediction with recursive rollout;
* observed that linear models can become qualitatively wrong over longer horizons.

One important issue was angle handling. The simulator angle can evolve continuously, while plotting or learned-model rollout may use a remapped angle. To improve reproducibility, raw simulator angles and remapped plotting/model angles were treated separately.

## Task 2: Kernel Models and Periodic Features

The second stage used nonlinear models to improve prediction accuracy.

The sparse Gaussian-kernel model has the form

$$
f(X) = \sum_i \alpha_i K(X, Z_i),
$$

where $Z_i$ are selected kernel centres and $\alpha_i$ are fitted coefficients.

Main work completed:

* implemented sparse Gaussian-kernel regression;
* constructed kernel matrices such as $K_{NM}$, $K_{MN}$, and $K_{MM}$;
* used `numpy.linalg.solve` rather than explicit matrix inversion;
* tuned kernel length scales and regularisation using JAX and SciPy L-BFGS-B;
* replaced raw $\theta$ with $\sin\theta$ and $\cos\theta$ to handle angular periodicity;
* compared raw linear, sin/cos linear, and sin/cos kernel models;
* evaluated models using one-step MSE, state scans, and rollout error.

The sin/cos kernel model gave the best aggregate performance, but it still accumulated error over longer open-loop rollouts.

## Technical Issues

| Issue                         | Effect                                            | Resolution                                                      |
| ----------------------------- | ------------------------------------------------- | --------------------------------------------------------------- |
| Jupyter animation unreliable  | animation was weak evidence for the report        | saved numerical arrays and static figures                       |
| angle remapping inconsistency | rollout plots could become non-reproducible       | separated raw simulator angle from remapped angle               |
| random data split             | repeated runs could give different figures        | saved split indices and fitted parameters                       |
| kernel tensor broadcasting    | possible silent shape errors                      | checked $N \times M$, $M \times N$, and $M \times M$ shapes     |
| L-BFGS-B interpretation       | unclear whether it behaved like gradient descent  | treated it as bounded quasi-Newton optimisation using gradients |
| rollout error accumulation    | low one-step MSE did not guarantee stable rollout | evaluated recursive rollout separately                          |

## Interim Report

The interim report summarises the first two weeks of the project. It covers:

* the project aim;
* simulator exploration;
* state-response maps;
* linear baseline modelling;
* kernel regression;
* JAX/SciPy hyperparameter optimisation;
* sin/cos angular features;
* rollout reliability;
* technical issues and resolutions;
* the plan for the control stage.

The report uses figures generated from the project code. Figures are stored in the `figures/` directory, with filenames beginning with `sf3_`.

## Plan for Control

The next stage is control. The planned workflow is:

1. add force/action as an input to the learned dynamics model;
2. collect random-action one-step data;
3. fit action-conditioned linear and kernel models;
4. define a feedback policy of the form

$$
u = p^T \phi(X);
$$

5. construct a trajectory loss penalising pole angle, cart displacement, velocities, and control effort;
6. optimise policy parameters using JAX;
7. compare policies optimised on the true simulator with policies optimised through learned-model rollouts.

This stage is closer to direct policy search than analytic LQR. The learned model may be useful for short-horizon model-based control, but long open-loop forecasts should not be trusted without further validation.

## Software

The project mainly uses Python and Jupyter notebooks.

Main packages include:

* NumPy;
* Matplotlib;
* SciPy;
* JAX;
* PyTorch, mainly for preliminary toy experiments.

## Reproducibility Notes

Where possible, generated data, train/validation/test split indices, fitted parameters, and figures are saved. This is important because random splitting, refitting, or inconsistent angle remapping can otherwise lead to different plots.

Some plotting layout, figure formatting, and report-formatting support was done with AI assistance. Numerical results were generated by running the project code.

## References

* SF3 Machine Learning Project Information, Easter Term 2026.
* JAX documentation: https://docs.jax.dev
* SciPy optimisation documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
* NumPy documentation: https://numpy.org/doc/stable/
