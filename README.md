# Housing Model Development

This repository began as a Udacity Nanodegree project working on the Boston Housing Data. Final work for this early work in machine learning can be viewed [here](doc/boston_housing.pdf). 

Later, this was reworked with a deeper emphasis on linear models. 

## Boston Redux

### Baseline Model

A [baseline](ipynb/boston_redux-00-baseline.ipynb) model was assessed against three models:

 - linear regression with no regularization (ordinary least squares)
 - linear regresison with $\ell_1$ regularization (LASSO)
 - linear regression with $\ell_2$ regularization (Ridge Regression)
 
A simple grid search was performed over the regularized models to identify an optimal coefficient for the regularization.
 
The results of this were


| alpha   | model                 | test_score | train_score  |
|:-------:|:---------------------:|:----------:|:------------:|
| NaN     | linear regression     | 0.711009   | 0.743956     |
| 0.00001 | lasso                 | 0.711009   | 0.743956     |
| 0.01000 | ridge                 | 0.711016   | 0.743956     |

### Standardized Model

A [standardized](ipynb/boston_redux-01-standardized.ipynb) model was assessed against the same three models:

 - linear regression with no regularization (ordinary least squares)
 - linear regresison with $\ell_1$ regularization (LASSO)
 - linear regression with $\ell_2$ regularization (Ridge Regression)
 
A simple grid search was performed over the regularized models to identify an optimal coefficient for the regularization.
 
The results of this were


| alpha   | model                 | test_score | train_score  |
|:-------:|:---------------------:|:----------:|:------------:|
| NaN     | linear regression     | 0.711009   | 0.743956     |
| 0.00001 | lasso                 | 0.711215   | 0.743880     |
| 0.01000 | ridge                 | 0.711298   | 0.742562     |

Note that standardization has no effect on the non-regularized linear regression.

### Skew Normal, Standardized Model

A [skew-normal, standardized](ipynb/boston_redux-02-skew_normal_standardized.ipynb) model was assessed against the same three models:

 - linear regression with no regularization (ordinary least squares)
 - linear regresison with $\ell_1$ regularization (LASSO)
 - linear regression with $\ell_2$ regularization (Ridge Regression)
 
A simple grid search was performed over the regularized models to identify an optimal coefficient for the regularization.
 
The results of this were


| alpha   | model                 | test_score | train_score  |
|:-------:|:---------------------:|:----------:|:------------:|
| NaN     | linear regression     | 0.751304   | 0.778260     |
| 0.00001 | lasso                 | 0.751307   | 0.778260     |
| 0.01000 | ridge                 | 0.751436   | 0.778242     |


Note that skew-normalization boosts both train and test performance for all three models.