# 7135CEM – Task 1: Student Performance Prediction

This repository contains the Python code for Task 1 of the 7135CEM module coursework. The project investigates how Gaussian Process Regression, Gaussian Process Classification, and Bayesian Networks can be used to predict secondary school student performance using the UCI “Student Performance” dataset.

## Project overview

- **Problem**: Predict final grades and pass/fail status of secondary school students using routinely collected school data (study time, absences, parental education, support, etc.).
- **Goals**:
  - Build a Gaussian Process Regression model for continuous final grade (G3).
  - Build a Gaussian Process Classification model for binary pass/fail labels.
  - Construct a Bayesian Network to explore dependencies among student, family, and school variables.
- **Dataset**: UCI Machine Learning Repository – Student Performance (Mathematics and Portuguese subjects).

## Repository structure

- `data/`
  - `student-mat.csv` – Mathematics dataset.
  - `student-por.csv` – Portuguese dataset.
- `src/`
  - `task1_preprocessing.py` – Data loading, cleaning, encoding, and standardisation.
  - `task1_gp_regression.py` – Gaussian Process Regression with 5-fold cross-validation.
  - `task1_gp_classification.py` – Gaussian Process Classification (pass/fail) with 5-fold cross-validation.
  - `task1_bayesian_network.py` – Bayesian Network structure learning, parameter estimation, and inference.
  - `task1_visualisation.py` – Scripts to generate figures (dataset overview, correlation heatmap, etc.).
- `requirements.txt` – Python package dependencies.

## How to run

1. Install Python 3.9+ and the required packages:


