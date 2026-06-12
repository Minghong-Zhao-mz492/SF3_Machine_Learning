# Task 3.1 report notes

Run mode: `full`. Use full-run values in the final report.

Selected command action range: ±20. This corresponds to approximately ±15.2 N after tanh saturation with max_force=20.

Sparse-kernel test mean standardised MSE at this range: 0.01416.

The model input was `[x, x_dot, sin(theta), cos(theta), theta_dot, action]`, i.e. the physical 5-variable state-action input with periodic angle represented by sine and cosine.

The sparse-kernel model was fitted with centres selected from the training data and a multi-output solve; this is equivalent to four independent output models sharing centres and kernel hyperparameters.

Largest tested action bin inside the selected training range: [10, 20]; kernel standardised MSE = 0.0157.

First extrapolation bin beyond the selected training range: [20, 40]; kernel standardised MSE = 0.08948.

Suggested final-report claim: action-conditioned one-step prediction is reliable inside the sampled action range, but the stress-test bin beyond the training range should be treated as extrapolation, especially because large actions lie near the tanh saturation region.

Next notebook: optimise a linear feedback policy on the true CartPole dynamics, then compare with model-based optimisation over a short horizon.
