# OP-airPLS

This repository is for OP-airPLS, an adaptive grid search optimization algorithm to find best parameters for the airPLS baseline removal.

It contains sample data used in the experiments, and demonstrations of using Jupyter notebooks files to run the optimization algorithm. 

## Requirements

The code in this repo has been tested with the following software versions:
- Python 3.7.0
- PyTorch 0.4.1
- Scikit-Learn 0.20.0
- Numpy 1.15.1
- Jupyter 5.0.0
- Seaborn 0.9.0
- Matplotlib 3.0.0

We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package. 

## Data

The sample data for the experiments can be found and downloaded in the `data` subdirectory.

## Files

This repo should contain the following files:
- 1_reference_finetuning.ipynb - demonstrates fine-tuning a pre-trained CNN on the 30-isolate classification task
- 2_prediction.ipynb - demonstrates making predictions with a fine-tuned CNN
- 3_clinical_finetuning.ipynb - demonstrates fine-tuning a pre-trained CNN using clinical data and making predictions for individual patients
- config.py - contains information about the provided dataset
- datasets.py - contains code for setting up datasets and dataloaders for spectral data
- resnet.py - contains ResNet CNN model class
- training.py - contains code for training CNN and making predictions
- reference_model.ckpt - saved parameters for pre-trained CNN to be used for notebooks 1 and 2
- clinical_model.ckpt - saved parameters for pre-trained CNN to be used for demo 3

## Running the notebooks

Experiment times reported in the notebooks were achieved on a 2018 Macbook Pro. If you find any bugs or have questions, please contact `csho@stanford.edu`
