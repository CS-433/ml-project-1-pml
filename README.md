# The Higgs Boson Machine Learning Challenge

The goal of this project was to apply machine learning techniques to actual CERN particle accelerator data to recreate the process of “discovering” the Higgs particle. Using the different physical properties of the particles’ collisions, we had to determine whether or not a Higgs boson was present. To solve this binary classification task we first preprocessed the data, by removing irrelevant features and transforming others, while splitting the dataset in relevant categories. We then used different regression methods, hyper-parameter tuning and cross-validation to find the most accurate model. We achieved the best results using Least Squares, but maybe some more computer power could have allowed us to better tune our logistic models.

## Usage
### Dependencies
#### Main
* Python (only tested on >= 3.9) 
* Numpy (only tested on >= 1.21)
#### Visualization (`graphs.ipynb`)
* Matplotlib (only tested on >= 3.4)
* Seaborn (only tested on >= 0.11)
### Submission file
To create a submission file for aicrowd.com, you must run `python3 scripts/run.py <function_number>` where <function_number> is used to selected the function with which we want to run the algorithm :
* 1 -> Least Squares Gradient Descent
* 2 -> Least Squares Stochstic Gradient Descent
* 3 -> Least Squares
* 4 -> Ridge Regression
* 5 -> Logistic Regression
* 6 -> Regularized Logistic Regression 

If no number is specified, the default function that will be used is Ridge Regression.

After running the command a file called `result.csv` will be created in the `scripts` directory, and can directly be uploaded to aicrowd.com to make a submission.

## Files
The different files of the `scripts` directory are :
* `graphs.ipynb`
* `helpers_data.py`
* `implementations.py`
* `proj1_helpers.py`
* `project1.ipynb`
* `run.py`
