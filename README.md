
# Comparing Gamma Generalized Linear Models and Neural Networks for Predicting Insurance Claim Costs

[![R-CMD-check](https://github.com/elliotwatson1/Comparing-Neural-Networks-and-GLMS/workflows/R-CMD-check/badge.svg)](https://github.com/elliotwatson1/Comparing-Neural-Networks-and-GLMS/actions)
[![R Version](https://www.r-pkg.org/badges/version/insuranceData)](https://cran.r-project.org/package=insuranceData)

This repository contains all the code used to produce the results presented in the dissertation titled:

**“Comparing Gamma Generalized Linear Models and Neural Networks for Predicting Insurance Claim Costs.”**

---

## Table of Contents

* [Dataset](#dataset)
* [Repository Structure](#repository-structure)
* [Key Scripts](#key-scripts)
* [Usage](#usage)
* [License](#license)

---

## Dataset

The dataset used throughout the study is `dataOhlsson` from the R package **insuranceData**:
[View Dataset Documentation](https://cran.r-project.org/web/packages/insuranceData/insuranceData.pdf)

---

## Repository Structure

```
.
├── Images/             # All images generated throughout the study
├── Statistics/         # Large file with neural network metrics
├── runs/               # Files for use with the R library tfruns
├── initial_exploration.R   # Main research and exploratory analysis
├── metrics_huge_run.csv    # Metrics for all neural network combinations
├── more_models.R            # Code for further explorations chapter
├── sweden_maps.R            # Code for generating maps
├── train_model.R            # Hyperparameter tuning and training neural networks
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

