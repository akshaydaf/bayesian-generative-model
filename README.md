# Bayesian Generative Model for Handwritten Digits

A Bayesian mixture model that learns to generate handwritten digit images using a Dirichlet-Bernoulli framework. Built with PyMC and evaluated on the sklearn 8x8 digits dataset.

## Overview

The model treats each digit class as a mixture component with its own pixel-activation probabilities. A Dirichlet prior governs mixture weights, and Beta-Bernoulli likelihoods model individual pixel values. MCMC sampling (NUTS) recovers posterior distributions over class prototypes, which are then used to generate new digit images.

The project explores how the Dirichlet concentration parameter affects learned representations, comparing models with concentrations of 0.5, 10, and 25 using LOO-CV and posterior predictive MSE.

## Stack

- **PyMC** — probabilistic programming and MCMC inference
- **ArviZ** — diagnostics, model comparison (LOO), trace plots
- **scikit-learn** — digits dataset and evaluation metrics
