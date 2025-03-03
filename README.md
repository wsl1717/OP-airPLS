# OP-airPLS

## Overview

OP-airPLS is an adaptive grid search optimization algorithm designed to find the best parameters for the airPLS baseline removal method. It also includes a PCA-RF machine learning model to predict optimal airPLS parameters directly from input spectra.

This repository provides:
- Sample data used in the experiments.
- Jupyter notebooks demonstrating the optimization algorithm.
- A Jupyter notebook for making predictions using the trained PCA-RF model.

## Procedure

The overall workflow consists of the following steps:

1. **Simulating Spectra**: Generate synthetic spectra along with their true baselines.
2. **Optimizing Parameters**: Use the iterative optimization algorithm to determine the optimal airPLS parameters, $`(\lambda^*, \tau^*)`$, for each spectrum.
3. **Training the Machine Learning Model**: 
   - Use optimized parameter pairs to create a dataset.
   - Split the dataset into training, validation, and test sets.
   - Train a **PCA-RF** model to learn the relationship between spectra and optimal airPLS parameters.
4. **Evaluating Predictions**: Compare the predicted baseline (from PCA-RF) with the true baseline.

![TOC](/images/OP-airPLS-Table-of-Content.png)

The iterative optimization algorithm follows the workflow illustrated below:

![flowchart](/images/OP-airPLS-flowchart.png)

## Requirements

The code has been tested with the following software versions:

- Python 3.11.5
- NumPy 1.24.3
- Pandas 2.2.1
- Matplotlib 3.8.4
- SciPy 1.11.1
- Scikit-Learn 1.4.1

We recommend using the **Anaconda** Python distribution, which supports Windows, macOS, and Linux. To install the required packages, you can use:

## Data

Sample datasets are available in the following subdirectories:

- **Optimization data**: [`1. Optimization Part/data/`](./1.%20Optimization%20Part/data/)
- **Machine learning prediction data**: [`2. ML Part/B&E_sample_spectra.csv`](./2.%20ML%20Part/B&E_sample_spectra.csv)

The PCA-RF model also accepts other `.csv` files with the same format.

## Running the Notebooks

Before running the Jupyter notebooks, ensure that file paths in the scripts match the location of your dataset on your local machine.

### **System Requirements**
The experiments in this repository were conducted on the following systems:

- **Laptop**: Windows 11, Intel i7-10875H (2.30GHz), 32GB RAM  
- **Desktop**: Windows 11, Intel i7-13700KF (3.40GHz), 64GB RAM  

Performance may vary depending on hardware specifications.

## Contact

For any questions, issues, or collaboration opportunities, please reach out via:

📧 **Email**:  
- [jiaheng.cui@uga.edu](mailto:jiaheng.cui@uga.edu)  
- [zhao-nano-lab@uga.edu](mailto:zhao-nano-lab@uga.edu)  

💬 **GitHub**:  
- Open an **Issue** or start a discussion in **GitHub Discussions**.

We welcome feedback and potential collaborations!
