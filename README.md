# Quantum-inspired Evolutionary Algorithms

This repository contains implementations of quantum-inspired evolutionary algorithms (QIEAs) applied to logistic regression. Our work extends the ideas presented in the paper [Quantum-Inspired Acromyrmex Evolutionary Algorithm](https://www.nature.com/articles/s41598-019-48409-5) to two specific tasks:

1. **Logistic Regression Optimization:**  
   - We apply QIEA to optimize the parameters of logistic regression models.
   - The performance of this quantum-inspired approach is validated against traditional gradient-based logistic regression.

2. **Hyperparameter Tuning:**  
   - We also demonstrate how QIEA can be used for automated hyperparameter tuning of logistic regression (and potentially other machine learning models).
   - This module provides an alternative to standard hyperparameter optimization techniques by leveraging quantum-inspired operators to efficiently explore the hyperparameter space.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Demos](#demos)
- [Algorithm Details](#algorithm-details)
- [Experimental Results](#experimental-results)
- [License](#license)
- [References](#references)

---

## Overview

This project explores the application of quantum-inspired evolutionary algorithms to solve optimization problems in the context of logistic regression. Our approach uses evolutionary operators inspired by quantum mechanics (such as superposition and quantum rotation gates) to find optimal weights in logistic regression tasks. In addition, we propose an automated hyperparameter tuning routine based on similar quantum-inspired principles.

Our implementation builds on the framework introduced in the Nature paper, which presented the **Quantum-Inspired Acromyrmex Evolutionary Algorithm (QIAEA)**. We adapt this core algorithm to address:
- **Parameter Optimization:** Finding the optimum weights for logistic regression.
- **Hyperparameter Optimization:** Tuning parameters (e.g., learning rate, regularization strength, and quantum rotation settings) to enhance model performance.

---

## Repository Structure

- **evaluation/**: Contains demo notebooks.
- **evo_learn/**: Hosts the QIEA-based logistic regression module.
- **paper_implementation/**: Contains code adapted from the original QIAEA paper.

---

## Demos

The demo notebooks that illustrate the functionality of the project are located at:
- `evaluation/titanic_dataset.ipynb`
- `evaluation/synthetic_dataset.ipynb`

These notebooks showcase the application of QIEA for both logistic regression optimization and hyperparameter tuning.

---

## Algorithm Details

### Quantum-Inspired Evolutionary Algorithm (QIEA)

Our implementation adapts ideas from the **Quantum-Inspired Acromyrmex Evolutionary Algorithm (QIAEA)**. In QIEA:
- **Representation:** Each solution is encoded as a quantum chromosome—a vector of qubits where the amplitudes encode probabilities.
- **Initialization:** The qubits are initialized in a uniform superposition (using Hadamard gates), ensuring a diverse starting population.
- **Measurement:** The quantum state is collapsed to obtain candidate classical solutions.
- **Evaluation:** Candidates are evaluated based on a logistic regression cost function (or another appropriate fitness metric).
- **Evolution:** Quantum rotation (using RY gates), mutation, and other operators are applied to steer the population toward higher fitness.
- **Selection:** The best individuals are retained to guide subsequent generations.

### Application to Logistic Regression

The QIEA is employed to minimize the logistic loss function by evolving the weight vector of the regression model. The fitness function inversely reflects the loss (i.e., lower loss corresponds to higher fitness).

### Hyperparameter Tuning

Parallel to weight optimization, we also use QIEA concepts to explore hyperparameter configurations (e.g., learning rate, regularization strength). This evolution-based hyperparameter tuner aims to efficiently find more optimal settings than the classical grid or random search.

---

## Experimental Results

Extensive experiments were conducted comparing:
- **QIEA vs. Gradient Descent:**  
  Demonstrating that the QIEA efficiently navigates the parameter space and avoids local minima, achieving competitive or superior accuracy in high-dimensional settings.
- **Hyperparameter Optimization:**  
  Our QIEA-driven tuner successfully identifies promising hyperparameters, improving overall model performance compared to baseline methods.

Detailed performance evaluations are available in the demo notebooks.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- Montiel, O., Rubio, Y., Olvera, C., et al. (2019). **Quantum-Inspired Acromyrmex Evolutionary Algorithm.** *Scientific Reports*, 9, Article number: 12181. [Link](https://www.nature.com/articles/s41598-019-48409-5)
