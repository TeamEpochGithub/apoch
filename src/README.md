# Source Code

This folder contains the base source code for future Epoch competitions.

## Structure

The `src` folder is organized as follows:

- `config`: Contains schema files for the configuration files (cv, train, submit, wandb) used in the competition.
- `modules`: Contains the main modules for the competition.
  - `modules/logger`: Contains the logger module.
  - `modules/training`: Contains training related modules that occur in `train_sys`
  - `modules/transformation`: Contains transformation related modules that occur in `x_sys` / `y_sys`
- `scoring`: Contains scoring functions for the competition, used in CV and test evaluation.
- `setup`: Contains the setup files for the competition.
- `typing`: Contains common type definitions for the competition.
- `utils`: Contains utility functions for the competition.

