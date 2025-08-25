
# Code and Resources for "From Generalised Linear Models to Neural Networks: Modelling Insurance Claim Costs with Traditional and Modern Approaches"

[![R-CMD-check](https://github.com/elliotwatson1/Comparing-Neural-Networks-and-GLMS/workflows/R-CMD-check/badge.svg)](https://github.com/elliotwatson1/Comparing-Neural-Networks-and-GLMS/actions)
[![R Version](https://www.r-pkg.org/badges/version/insuranceData)](https://cran.r-project.org/package=insuranceData)

This repository contains all the code used to produce the results presented in the dissertation titled:

**“From Generalised Linear Models to Neural Networks: Modelling Insurance Claim Costs with Traditional and Modern Approaches”**

---

## Table of Contents

* [Dataset](#dataset)
* [Repository Structure](#repository-structure)
* [Key Scripts](#key-scripts)
* [Usage](#usage)
* [License](#license)

---

## Dataset

The study uses the `dataOhlsson` dataset from the **insuranceData** R package, which provides Swedish motor insurance claim data. It can be found at:
[View Dataset Documentation](https://cran.r-project.org/web/packages/insuranceData/insuranceData.pdf)

---

## Repository Structure

```
├── Images/                 # Figures and plots generated in the analysis
├── Statistics/             # Neural network performance metrics (large file)
├── runs/                   # Training logs for the R package tfruns
├── initial_exploration.R   # Exploratory analysis and main model file
├── more_models.R           # Code supporting the “Further Explorations” chapter
├── sweden_maps.R           # Script for generating Swedish insurance maps
├── train_model.R           # Neural network training and hyperparameter tuning
```

---

## Key Scripts

* **initial\_exploration.R** – Main script containing the majority of the research and exploratory analysis.
* **metrics\_huge\_run.csv** – CSV file containing metrics for all neural network combinations during hyperparameter tuning.
* **more\_models.R** – Code for the “Further Explorations” chapter of the dissertation.
* **sweden\_maps.R** – Generates the maps used throughout the dissertation.
* **train\_model.R** – Script used for hyperparameter tuning and training neural networks.

---

## Usage

1. Clone this repository:

```bash
git clone https://github.com/elliotwatson1/Comparing-Neural-Networks-and-GLMS.git
```

2. Open R or RStudio and set the working directory to the repository folder.
3. Install required packages (if not already installed):

```R
install.packages(c("insuranceData", "tensorflow", "tfruns", "ggplot2", "dplyr"))
```

4. Run the scripts in the order you need. Typically:

```R
source("initial_exploration.R")
source("train_model.R")
source("more_models.R")
source("sweden_maps.R")
```

---

